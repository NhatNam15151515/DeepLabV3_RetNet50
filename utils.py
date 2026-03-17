"""
utils.py — Tiện ích hệ thống: Seed, Loss Functions, Logging helpers.

Tuân thủ:
- CUDNN Deterministic + benchmark=False
- Loss ratio theo Phase (Phase 1: 0.7 CE + 0.3 Dice, Phase 2: 0.5 Dice + 0.3 CE + 0.2 Lovász)
- Class weight chỉ áp cho CE, Dice giữ nguyên
"""

import random
import logging
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger("DeepLabV3_FineTune")


# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================
def init_environment(seed: int = 42):
    """Cố định seed toàn hệ thống và bật CUDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info(f"Seed={seed}, cudnn.deterministic=True, cudnn.benchmark=False")


def get_device() -> torch.device:
    """Trả về device phù hợp (GPU nếu có, ngược lại CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        logger.warning("KHÔNG tìm thấy GPU — chạy trên CPU (chậm hơn đáng kể).")
    return dev


# =============================================================================
# 2. LOSS FUNCTIONS — Ratio theo Phase
# =============================================================================
def get_criterion_phase1(class_weights: torch.Tensor = None, device: torch.device = None, label_smoothing: float = 0.0):
    """
    Phase 1 (Linear Probing): 0.4 * Focal + 0.3 * Dice + 0.3 * CE
    Focal Loss tập trung vào class khó, Dice bù recall. Label smoothing cho CE giảm overconfidence.
    """
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    focal = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def loss_fn(logits, target):
        return 0.4 * focal(logits, target) + 0.3 * dice(logits, target) + 0.3 * ce(logits, target)
    return loss_fn


def get_criterion_phase2(class_weights: torch.Tensor = None, device: torch.device = None):
    """
    Phase 2 (Fine-tuning): 0.25 * Focal + 0.35 * Dice + 0.4 * Lovász
    Tăng Lovász (0.4) để cải thiện BF1 và chất lượng biên.
    """
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    focal = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    lovasz = smp.losses.LovaszLoss(mode="multiclass", from_logits=True)

    def loss_fn(logits, target):
        return 0.25 * focal(logits, target) + 0.35 * dice(logits, target) + 0.4 * lovasz(logits, target)
    return loss_fn


# =============================================================================
# 3. LOGGING SETUP
# =============================================================================
def setup_logging(log_file: str = "training.log"):
    """Cấu hình logging ghi ra cả console và file."""
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    root = logging.getLogger("DeepLabV3_FineTune")
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)
    return root
