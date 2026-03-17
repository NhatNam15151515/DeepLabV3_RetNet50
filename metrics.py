"""
metrics.py — Tất cả metrics bắt buộc cho validation.

Tuân thủ:
- mIoU, Pixel Accuracy, Per-class IoU
- Boundary F1 Score
- Instance Count: Exact Match Accuracy + MACE (Mean Absolute Count Error)
- MIN_AREA_THRESHOLD = max(0.0005 * H * W, 50)  (relative, không hardcode)
- GT instance đếm hoàn toàn bằng cùng hàm CC như prediction (công bằng)

Tối ưu tốc độ:
- Các hàm nhận `preds` (đã argmax) thay vì `logits` → tránh argmax lặp.
- mIoU dùng confusion matrix → 1 pass thay vì loop N class.
- Boundary F1 chỉ xét class xuất hiện trong batch → skip class vắng mặt.
- Nếu truyền logits (4D), tự động argmax; nếu truyền preds (3D), dùng thẳng.
"""

import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger("DeepLabV3_FineTune")


def _to_preds(x: torch.Tensor) -> torch.Tensor:
    """Nếu x là logits (4D: B,C,H,W) → argmax. Nếu đã là preds (3D: B,H,W) → trả luôn."""
    if x.ndim == 4:
        return torch.argmax(x, dim=1)
    return x


def _min_area(h: int, w: int) -> float:
    """Ngưỡng diện tích adaptive theo kích thước ảnh."""
    return max(0.0005 * h * w, 50)


# =================================================================
# 1. mIoU  (bỏ qua background class 0) — confusion matrix
# =================================================================
def calculate_miou(logits_or_preds: torch.Tensor, labels: torch.Tensor,
                   num_classes: int) -> float:
    preds = _to_preds(logits_or_preds)
    # Flatten và tính confusion matrix 1 lần
    p_flat = preds.view(-1).long()
    l_flat = labels.view(-1).long()
    # Bỏ pixel ngoài range
    valid = (l_flat >= 0) & (l_flat < num_classes) & (p_flat >= 0) & (p_flat < num_classes)
    p_flat = p_flat[valid]
    l_flat = l_flat[valid]
    # Confusion matrix trên GPU
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=preds.device)
    cm.index_put_(
        (l_flat, p_flat),
        torch.ones_like(l_flat, dtype=torch.long),
        accumulate=True,
    )
    # IoU per class (skip background 0)
    ious = []
    for c in range(1, num_classes):
        tp = cm[c, c].item()
        fn = cm[c, :].sum().item() - tp
        fp = cm[:, c].sum().item() - tp
        union = tp + fn + fp
        if union > 0:
            ious.append(tp / union)
    return float(np.mean(ious)) if ious else 0.0


# =================================================================
# 2. Pixel Accuracy
# =================================================================
def calculate_pixel_accuracy(logits_or_preds: torch.Tensor,
                             labels: torch.Tensor) -> float:
    preds = _to_preds(logits_or_preds)
    return ((preds == labels).sum().float() / labels.numel()).item()


# =================================================================
# 3. Per-class IoU  (confusion matrix)
# =================================================================
def get_confusion_matrix(logits_or_preds: torch.Tensor, labels: torch.Tensor,
                         num_classes: int) -> torch.Tensor:
    """Trả về confusion matrix (num_classes, num_classes) cho 1 batch. Dùng để cộng dồn nhiều batch."""
    preds = _to_preds(logits_or_preds)
    p_flat = preds.view(-1).long()
    l_flat = labels.view(-1).long()
    valid = (l_flat >= 0) & (l_flat < num_classes) & (p_flat >= 0) & (p_flat < num_classes)
    p_flat = p_flat[valid]
    l_flat = l_flat[valid]
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=preds.device)
    cm.index_put_(
        (l_flat, p_flat),
        torch.ones_like(l_flat, dtype=torch.long),
        accumulate=True,
    )
    return cm


def per_class_iou_from_cm(cm: torch.Tensor, num_classes: int) -> dict:
    """Tính per-class IoU từ confusion matrix (bỏ qua class 0)."""
    cm = cm.cpu()
    result = {}
    for c in range(1, num_classes):
        tp = cm[c, c].item()
        fn = cm[c, :].sum().item() - tp
        fp = cm[:, c].sum().item() - tp
        union = tp + fn + fp
        if union > 0:
            result[c] = tp / union
    return result


def calculate_per_class_iou(logits_or_preds: torch.Tensor, labels: torch.Tensor,
                            num_classes: int) -> dict:
    preds = _to_preds(logits_or_preds)
    cm = get_confusion_matrix(preds, labels, num_classes)
    return per_class_iou_from_cm(cm, num_classes)


# =================================================================
# 4. Boundary F1 Score — chỉ xét class xuất hiện trong batch
# =================================================================
def calculate_boundary_f1(logits_or_preds: torch.Tensor, labels: torch.Tensor,
                          num_classes: int) -> float:
    """
    Lưu ý: Resize GT bằng nearest hoặc upsample pred trước khi so sánh.
    Hàm này giả sử logits và labels đã cùng spatial resolution.
    Tối ưu: chỉ loop qua class xuất hiện trong batch (thường << num_classes).
    """
    preds = _to_preds(logits_or_preds)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    kernel = np.ones((3, 3), np.uint8)
    f1_list = []

    # Tìm union các class xuất hiện trong batch (thường 5-10 thay vì 28)
    present_classes = set()
    for b in range(preds_np.shape[0]):
        present_classes.update(np.unique(preds_np[b]))
        present_classes.update(np.unique(labels_np[b]))
    present_classes.discard(0)  # skip background

    for b in range(preds_np.shape[0]):
        for c in present_classes:
            pm = (preds_np[b] == c).astype(np.uint8)
            lm = (labels_np[b] == c).astype(np.uint8)
            if pm.sum() == 0 and lm.sum() == 0:
                continue

            pb = cv2.dilate(pm, kernel) - cv2.erode(pm, kernel)
            lb = cv2.dilate(lm, kernel) - cv2.erode(lm, kernel)

            inter = np.logical_and(pb, lb).sum()
            prec = inter / (pb.sum() + 1e-6)
            rec = inter / (lb.sum() + 1e-6)
            if prec + rec > 0:
                f1_list.append(2 * prec * rec / (prec + rec))

    return float(np.mean(f1_list)) if f1_list else 0.0


# =================================================================
# 5. Instance Count: Exact Match + MACE — chỉ xét class xuất hiện
# =================================================================
def get_instance_count_metrics(
    logits_or_preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple:
    """
    Returns (exact_match_accuracy, mace).
    Dùng Connected Components để đếm instance.
    Cùng một hàm CC (và cùng MIN_AREA) cho cả GT và Pred → công bằng.
    """
    preds = _to_preds(logits_or_preds)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    B, H, W = preds_np.shape
    thresh = _min_area(H, W)

    # Chỉ xét class xuất hiện
    present_classes = set()
    for b in range(B):
        present_classes.update(np.unique(preds_np[b]))
        present_classes.update(np.unique(labels_np[b]))
    present_classes.discard(0)

    exact = 0
    total = 0
    abs_errors = []

    for b in range(B):
        for c in present_classes:
            pm = (preds_np[b] == c).astype(np.uint8)
            lm = (labels_np[b] == c).astype(np.uint8)
            if pm.sum() == 0 and lm.sum() == 0:
                continue

            cnt_gt = _count_cc(lm, thresh)
            cnt_pred = _count_cc(pm, thresh)

            total += 1
            abs_errors.append(abs(cnt_gt - cnt_pred))
            if cnt_gt == cnt_pred:
                exact += 1

    ema = exact / total if total > 0 else 1.0
    mace = float(np.mean(abs_errors)) if abs_errors else 0.0
    return ema, mace


def _count_cc(binary: np.ndarray, min_area: float) -> int:
    """Đếm connected components hợp lệ (area ≥ min_area)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(
        1 for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    )
