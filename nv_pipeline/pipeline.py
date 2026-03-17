"""NutritionVersePipeline – ghép Tier1 (seg), Tier2 (depth/volume), Tier3 (weight).

Đây là bản được gom vào package `nv_pipeline` để tách biệt khỏi code training
DeepLabV3 hiện tại. Khi cần dùng lại pipeline phân tích bữa ăn end-to-end,
import từ đây.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from .tier1_segmentation import Tier1Segmentation  # type: ignore
from .tier2_depth_volume import DepthVolumeResult, Tier2DepthVolume
from .tier3_weight_estimation import Tier3WeightEstimation, WeightEstimationResult


@dataclass
class FoodItemAnalysis:
    """Kết quả phân tích hoàn chỉnh cho một món ăn."""
    class_name: str
    confidence: float
    bbox: np.ndarray
    mask: np.ndarray
    area_pixels: int
    area_cm2: float
    height_cm: float
    volume_cm3: float
    density_g_per_cm3: float
    density_source: str
    weight_grams: float
    weight_confidence: float

    def to_dict(self) -> Dict:
        """Chuyển về dictionary để serialize."""
        return {
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            "area_pixels": int(self.area_pixels),
            "area_cm2": float(self.area_cm2),
            "height_cm": float(self.height_cm),
            "volume_cm3": float(self.volume_cm3),
            "density_g_per_cm3": float(self.density_g_per_cm3),
            "density_source": self.density_source,
            "weight_grams": float(self.weight_grams),
            "weight_confidence": float(self.weight_confidence),
        }


@dataclass
class MealAnalysisResult:
    """Kết quả phân tích toàn bộ bữa ăn."""
    image_path: str
    food_items: List[FoodItemAnalysis]
    total_weight_grams: float
    depth_map: np.ndarray

    @property
    def num_items(self) -> int:
        return len(self.food_items)

    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "num_items": self.num_items,
            "total_weight_grams": float(self.total_weight_grams),
            "food_items": [item.to_dict() for item in self.food_items],
        }

    def summary(self) -> str:
        lines = [
            f"Meal Analysis: {Path(self.image_path).name}",
            f"Total items: {self.num_items}",
            f"Total weight: {self.total_weight_grams:.1f}g",
            "-" * 40,
        ]
        for item in self.food_items:
            lines.append(
                f"  • {item.class_name}: {item.weight_grams:.1f}g "
                f"(vol={item.volume_cm3:.1f}cm³)"
            )
        return "\n".join(lines)


class NutritionVersePipeline:
    """Pipeline 3 tầng: Segmentation → Depth/Volume → Weight."""

    def __init__(
        self,
        tier1: Tier1Segmentation,
        tier2: Optional[Tier2DepthVolume] = None,
        tier3: Optional[Tier3WeightEstimation] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.tier1 = tier1
        self.tier2 = tier2 or Tier2DepthVolume()
        self.tier3 = tier3 or Tier3WeightEstimation()

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
    ) -> MealAnalysisResult:
        """Phân tích một ảnh bữa ăn."""
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
        else:
            image_path = "numpy_array"
            img_array = image

        if self.verbose:
            print(f"\n[Pipeline] Analyzing: {Path(image_path).name}")

        # Tier 1: segmentation
        if self.verbose:
            print("[Tier1] Running food segmentation...")

        seg_results = self.tier1.predict(img_array, conf_threshold=conf_threshold)  # type: ignore[arg-type]

        if self.verbose:
            print(f"[Tier1] Detected {len(seg_results)} food items")

        if len(seg_results) == 0:
            return MealAnalysisResult(
                image_path=image_path,
                food_items=[],
                total_weight_grams=0.0,
                depth_map=np.zeros(img_array.shape[:2], dtype=np.float32),
            )

        # Tier 2: depth & volume
        if self.verbose:
            print("[Tier2] Estimating depth and volume...")

        depth_map = self.tier2.estimate_depth(img_array)

        masks = [(r.class_name, r.mask) for r in seg_results]  # type: ignore[attr-defined]
        vol_results: List[DepthVolumeResult] = self.tier2.estimate_volume(
            img_array, masks, depth_map
        )

        # Tier 3: weight
        if self.verbose:
            print("[Tier3] Estimating weights...")

        items_for_weight = [(v.class_name, v.volume_cm3) for v in vol_results]
        weight_results: List[WeightEstimationResult] = self.tier3.estimate_weights_batch(
            items_for_weight
        )

        # Tổng hợp
        food_items = []
        for seg, vol, weight in zip(seg_results, vol_results, weight_results):
            item = FoodItemAnalysis(
                class_name=seg.class_name,
                confidence=seg.confidence,
                bbox=seg.bbox,
                mask=seg.mask,
                area_pixels=seg.area_pixels,
                area_cm2=vol.area_cm2,
                height_cm=vol.height_cm,
                volume_cm3=vol.volume_cm3,
                density_g_per_cm3=weight.density_g_per_cm3,
                density_source=weight.density_source,
                weight_grams=weight.weight_grams,
                weight_confidence=weight.confidence,
            )
            food_items.append(item)

        total_weight = sum(item.weight_grams for item in food_items)

        result = MealAnalysisResult(
            image_path=image_path,
            food_items=food_items,
            total_weight_grams=total_weight,
            depth_map=depth_map,
        )

        if self.verbose:
            print(f"[Pipeline] Analysis complete: {len(food_items)} items, {total_weight:.1f}g total")

        return result

