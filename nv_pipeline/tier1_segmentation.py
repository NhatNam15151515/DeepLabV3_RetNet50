"""tier1_segmentation.py — Module inference Tier 1: Segmentation + Instance Extraction.

Pipeline:
  1. Chạy model DeepLabV3+ → semantic mask
  2. Post-processing: tách từng instance theo class
     - Shape thon dài (elongated): dùng Connected Components thuần túy
     - Shape khối (block):         dùng Watershed pipeline chuẩn
  3. Trả về JSON format chuẩn cho Tier 2 (Volume) và Tier 3 (Weight)

Tuân thủ:
- KHÔNG GỘP INSTANCE cùng class
- MIN_AREA_THRESHOLD = max(0.0005 * H * W, 50)
- Watershed: Morph opening → Distance transform → Adaptive percentile (75/85)
             → Marker filter → Watershed
- Class elongated → skip watershed → chỉ CC
"""

import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("DeepLabV3_FineTune")

# ============================================================
# CÁC CLASS SHAPE THON DÀI — BỎ QUA WATERSHED
# Danh sách này cần được cấu hình theo dataset cụ thể.
# Ví dụ class_id (đã +1 do background=0) dành cho bún, mỳ, rau dài…
# ============================================================
ELONGATED_CLASS_IDS: List[int] = []  # Cần điền khi biết class map cụ thể


# ============================================================
# CORE: EXTRACT INSTANCES TỪ SEMANTIC MASK
# ============================================================
def extract_instances(
    pred_mask: np.ndarray,
    num_classes: int,
    elongated_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Tách instance từ semantic mask.

    Parameters
    ----------
    pred_mask : np.ndarray, shape (H, W), dtype uint8/int
    num_classes : int
    elongated_ids : list[int] — các class_id dùng CC thay vì watershed

    Returns
    -------
    List gồm các dict:
      {"instance_id", "class_id", "area", "bbox", "centroid"}
    """
    if elongated_ids is None:
        elongated_ids = ELONGATED_CLASS_IDS

    H, W = pred_mask.shape
    min_area = max(0.0005 * H * W, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    instances: List[Dict[str, Any]] = []
    gid = 1  # global instance counter

    for cid in range(1, num_classes):
        binary = (pred_mask == cid).astype(np.uint8) * 255
        if binary.sum() == 0:
            continue

        # Bước bắt buộc: Morphological opening giải nhiễu
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        if cid in elongated_ids:
            # ---- SHAPE THON DÀI → Connected Components thuần ----
            gid = _extract_cc(binary, cid, min_area, instances, gid)
        else:
            # ---- SHAPE KHỐI → Watershed pipeline chuẩn ----
            gid = _extract_watershed(binary, cid, min_area, H, W, instances, gid)

    return instances


# ============================================================
# CONNECTED COMPONENTS (cho class thon dài)
# ============================================================
def _extract_cc(
    binary: np.ndarray,
    cid: int,
    min_area: float,
    instances: list,
    gid: int,
) -> int:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for j in range(1, n):
        area = stats[j, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP],
                       stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[j]
        instances.append({
            "instance_id": gid,
            "class_id": cid,
            "area": int(area),
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "centroid": [float(cx), float(cy)],
        })
        gid += 1
    return gid


# ============================================================
# WATERSHED PIPELINE (cho class dạng khối)
# ============================================================
def _extract_watershed(
    binary: np.ndarray,
    cid: int,
    min_area: float,
    H: int, W: int,
    instances: list,
    gid: int,
) -> int:
    """
    Flow: opening → distance transform → adaptive peak threshold
          → marker labeling (lọc peak nhỏ) → watershed
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    active = dist[dist > 0]

    if len(active) == 0:
        # Không có gì → fallback CC
        return _extract_cc(binary, cid, min_area, instances, gid)

    # Adaptive percentile theo diện tích blob
    obj_area = np.count_nonzero(binary)
    pct = 75 if obj_area < (0.05 * H * W) else 85
    thresh_val = np.percentile(active, pct)

    _, sure_fg = cv2.threshold(dist, thresh_val, 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Marker labeling → loại peak rác
    n_markers, markers_raw, m_stats, _ = cv2.connectedComponentsWithStats(sure_fg)
    marker_min = min_area * 0.3  # lõi marker cho phép nhỏ hơn full area một chút

    clean_markers = np.zeros_like(markers_raw, dtype=np.int32)
    new_idx = 1
    for m in range(1, n_markers):
        if m_stats[m, cv2.CC_STAT_AREA] >= marker_min:
            clean_markers[markers_raw == m] = new_idx
            new_idx += 1

    if new_idx == 1:
        # Tất cả markers bị loại → fallback CC
        return _extract_cc(binary, cid, min_area, instances, gid)

    # Chuẩn bị Watershed
    # background = 1, unknown = 0, sure_fg bắt đầu từ 2
    sure_bg = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers_ws = clean_markers + 1  # shift lên 1 (BG=1)
    markers_ws[unknown == 255] = 0  # unknown region

    bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers_ws)

    # Parse kết quả
    result_labels = markers_ws - 1  # shift lại
    result_labels[result_labels < 0] = 0  # boundary = -1 → 0

    for label_val in range(1, new_idx):
        inst_mask = (result_labels == label_val).astype(np.uint8)
        area = int(inst_mask.sum())
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(inst_mask)
        M = cv2.moments(inst_mask)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = x + w / 2.0, y + h / 2.0

        instances.append({
            "instance_id": gid,
            "class_id": cid,
            "area": area,
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "centroid": [float(cx), float(cy)],
        })
        gid += 1

    return gid


# ============================================================
# INFERENCE WRAPPER — Tier 1 API
# ============================================================
class Tier1Segmentation:
    """
    Wrapper inference cho Tier 1.
    Load model → predict → extract instances → trả JSON chuẩn.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        elongated_ids: Optional[List[int]] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_classes = num_classes
        self.elongated_ids = elongated_ids or ELONGATED_CLASS_IDS

    @torch.no_grad()
    def predict(
        self,
        image_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        image_tensor : (1, 3, H, W) — đã normalise

        Returns
        -------
        dict: {"mask": np.ndarray (H,W), "instances": [...]}
        """
        x = image_tensor.to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = self.model(x)            # (1, C, H, W)

        pred = torch.argmax(logits, dim=1)    # (1, H, W)
        mask_np = pred[0].cpu().numpy().astype(np.uint8)

        instances = extract_instances(
            mask_np, self.num_classes, self.elongated_ids
        )

        return {
            "mask": mask_np,
            "instances": instances,
        }

