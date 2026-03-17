"""Tầng 2: Ước lượng depth & thể tích từ ảnh RGB và mask.

Đây là bản được tách ra từ code NutritionVerse, dùng MiDaS để sinh depth
rồi kết hợp với mask để ước lượng volume cho từng món.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import OUTPUT_DIR, TIER2_CONFIG, ensure_directories, get_device  # type: ignore


class DepthVolumeResult:
    """
    Class lưu trữ kết quả ước lượng depth và volume cho một món ăn.
    """

    def __init__(
        self,
        class_name: str,
        depth_map: np.ndarray,
        mask: np.ndarray,
        config: Dict,
    ):
        self.class_name = class_name
        self.mask = mask

        # Extract depth values trong mask
        mask_bool = mask > 0
        depth_values = depth_map[mask_bool]

        # Lưu depth map (masked)
        self.depth_map = np.where(mask_bool, depth_map, 0).astype(np.float32)

        # Tính toán statistics
        if len(depth_values) > 0:
            self.mean_depth = float(np.mean(depth_values))
            self.std_depth = float(np.std(depth_values))
            self.min_depth = float(np.min(depth_values))
            self.max_depth = float(np.max(depth_values))
        else:
            self.mean_depth = self.std_depth = 0.0
            self.min_depth = self.max_depth = 0.0

        # Diện tích
        self.area_pixels = int(np.sum(mask_bool))

        # Chiều cao tương đối từ depth variance trong mask
        self.height_relative = self.max_depth - self.min_depth

        focal = config.get("focal_length_px", 1000.0)
        ref_dist = config.get("reference_distance_cm", 40.0)

        # Scale factor: pixels to cm tại khoảng cách tham chiếu
        pixel_to_cm = ref_dist / focal
        self.area_cm2 = self.area_pixels * (pixel_to_cm ** 2)

        if self.min_depth > 0.01 and self.max_depth > self.min_depth:
            depth_scale = config.get("depth_scale_factor", 1.0)
            self.height_cm = depth_scale * (1.0 / self.min_depth - 1.0 / self.max_depth)
            self.height_cm = max(0.5, min(self.height_cm, 20.0))
        else:
            self.height_cm = 2.5  # cm

        # Thể tích tương đối
        self.volume_relative = self.area_pixels * self.mean_depth * (1 + self.height_relative)

        # Thể tích ước lượng (cm³) với mô hình hình trụ đơn giản
        self.volume_cm3 = self.area_cm2 * self.height_cm

    def __repr__(self):
        return (
            f"DepthVolumeResult("
            f"class='{self.class_name}', "
            f"area={self.area_cm2:.1f}cm², "
            f"height={self.height_cm:.1f}cm, "
            f"volume={self.volume_cm3:.1f}cm³)"
        )


class Tier2DepthVolume:
    """Module Tier 2: load MiDaS, sinh depth map, ước lượng volume."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or TIER2_CONFIG
        self.device = get_device()

        print(f"[Tier2] Loading MiDaS model: {self.config['model_type']}")
        self._load_midas()
        print(f"[Tier2] MiDaS loaded trên device: {self.device}")

    def _load_midas(self):
        """Load MiDaS model và transform."""
        model_type = self.config["model_type"]

        # Load model từ torch hub
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()

        # Load transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def estimate_depth(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> np.ndarray:
        """
        Sinh depth map từ ảnh RGB.
        """
        # Load ảnh
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Không thể đọc ảnh: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

        orig_h, orig_w = img.shape[:2]

        # Transform
        input_batch = self.transform(img).to(self.device)

        # Inference
        prediction = self.midas(input_batch)

        # Resize về kích thước gốc
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize depth map về range [0, 1]
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-8:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map.astype(np.float32)

    def estimate_volume(
        self,
        image: Union[str, Path, np.ndarray],
        masks: List[Tuple[str, np.ndarray]],
        depth_map: Optional[np.ndarray] = None,
    ) -> List[DepthVolumeResult]:
        """
        Ước lượng thể tích cho từng món ăn dựa trên mask và depth.
        """
        if depth_map is None:
            depth_map = self.estimate_depth(image)

        results = []
        for class_name, mask in masks:
            result = DepthVolumeResult(
                class_name=class_name,
                depth_map=depth_map,
                mask=mask,
                config=self.config,
            )
            results.append(result)

        return results

    def visualize_depth(
        self,
        depth_map: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        colormap: int = cv2.COLORMAP_MAGMA,
    ) -> np.ndarray:
        """Visualize depth map với colormap."""
        depth_normalized = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)

        if output_path:
            cv2.imwrite(str(output_path), depth_colored)

        return depth_colored

    def visualize_volume(
        self,
        image: Union[str, Path, np.ndarray],
        results: List[DepthVolumeResult],
        depth_map: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Visualize kết quả volume estimation."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()

        h, w = img.shape[:2]

        depth_vis = self.visualize_depth(depth_map)
        if depth_vis.shape[:2] != (h, w):
            depth_vis = cv2.resize(depth_vis, (w, h))

        combined = np.hstack([img, depth_vis])

        y_offset = 30
        for result in results:
            text = f"{result.class_name}: V={result.volume_cm3:.1f}cm³"
            cv2.putText(
                combined, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25

        if output_path:
            cv2.imwrite(str(output_path), combined)

        return combined


def calibrate_depth_scale(
    measured_heights: List[float],
    estimated_heights: List[float],
) -> float:
    """Tính scale factor từ ground truth chiều cao."""
    if len(measured_heights) != len(estimated_heights):
        raise ValueError("Số lượng measurements không khớp")

    measured = np.array(measured_heights)
    estimated = np.array(estimated_heights)

    scale = np.sum(measured * estimated) / (np.sum(estimated ** 2) + 1e-8)

    return float(scale)


def calibrate_volume_scale(
    measured_volumes: List[float],
    estimated_volumes: List[float],
) -> float:
    """Tính scale factor cho volume từ ground truth."""
    return calibrate_depth_scale(measured_volumes, estimated_volumes)

