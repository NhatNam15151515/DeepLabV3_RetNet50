"""
postprocess.py — Post-processing pipeline cho Demo.

Pipeline:
  1. Morphological Smoothing per-class (Opening → Closing)
     - Dọn salt-pepper noise, làm mask sạch
     - Kernel nhỏ (3x3 hoặc 5x5), KHÔNG over-smooth
  2. Instance Counting qua Connected Components
     - Lọc component nhỏ (area < threshold) → bỏ noise
     - Trả về dict {class_id: count}
"""

import cv2
import numpy as np


# =====================================================================
# 1. MORPHOLOGICAL SMOOTHING
# =====================================================================
def smooth_mask(mask: np.ndarray, num_classes: int, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological Opening → Closing per-class.

    Args:
        mask:         (H, W) uint8, mỗi pixel = class_id
        num_classes:  tổng số class (bao gồm background = 0)
        kernel_size:  kích thước kernel (3 hoặc 5)

    Returns:
        cleaned mask (H, W) uint8
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    canvas = np.zeros_like(mask)

    for c in range(1, num_classes):
        bm = (mask == c).astype(np.uint8)
        if bm.sum() == 0:
            continue
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, kernel)   # xóa noise nhỏ
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, kernel)  # lấp lỗ nhỏ
        canvas[bm == 1] = c

    return canvas


def smooth_mask_batch(masks: np.ndarray, num_classes: int,
                      kernel_size: int = 5) -> np.ndarray:
    """Batch version: masks shape (B, H, W)."""
    out = np.zeros_like(masks)
    for b in range(masks.shape[0]):
        out[b] = smooth_mask(masks[b], num_classes, kernel_size)
    return out


# =====================================================================
# 2. INSTANCE COUNTING (Connected Components + Area Filter)
# =====================================================================
def _min_area(h: int, w: int) -> float:
    """Ngưỡng diện tích adaptive: 0.05% diện tích ảnh, tối thiểu 50px."""
    return max(0.0005 * h * w, 50)


def count_instances(mask: np.ndarray, num_classes: int,
                    min_area: float = None) -> dict:
    """
    Đếm số instance (connected components) cho từng class.

    Args:
        mask:        (H, W) uint8
        num_classes: tổng số class
        min_area:    ngưỡng diện tích tối thiểu, None = tự tính adaptive

    Returns:
        dict {class_id: instance_count}   (chỉ class có ≥ 1 instance)
    """
    H, W = mask.shape
    if min_area is None:
        min_area = _min_area(H, W)

    result = {}
    for c in range(1, num_classes):
        bm = (mask == c).astype(np.uint8)
        if bm.sum() == 0:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
        cnt = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area)
        if cnt > 0:
            result[c] = cnt

    return result


def count_instances_batch(masks: np.ndarray, num_classes: int,
                          min_area: float = None) -> list:
    """Batch version: trả về list[dict] cho mỗi ảnh."""
    return [count_instances(masks[b], num_classes, min_area)
            for b in range(masks.shape[0])]


# =====================================================================
# 3. FULL PIPELINE: Smooth → Count
# =====================================================================
def postprocess_and_count(mask: np.ndarray, num_classes: int,
                          kernel_size: int = 5,
                          min_area: float = None) -> tuple:
    """
    Pipeline đầy đủ cho 1 ảnh:
      1. Smooth mask
      2. Đếm instance từ mask đã smooth

    Returns:
        (cleaned_mask, instance_dict)
    """
    cleaned = smooth_mask(mask, num_classes, kernel_size)
    instances = count_instances(cleaned, num_classes, min_area)
    return cleaned, instances
