import os
import sys

# Đảm bảo project root (chứa config.py, dataset.py, ...) nằm trên sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

"""
eval_ensemble.py — Đánh giá ensemble 3 checkpoint: average logits → argmax → mIoU / per-class IoU.

Cách dùng:
  python scripts/training/eval_ensemble.py --checkpoints runs/train/exp18/weights/best.pth ...
hoặc chỉnh LIST_CHECKPOINTS trong script.

Nếu chỉ có 1 checkpoint thì tương đương eval thường (không ensemble).
"""

import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as CFG
from utils import init_environment, get_device, setup_logging
from model_setup import create_model
from dataset import FoodSegDataset
from metrics import (
    calculate_miou, calculate_pixel_accuracy, calculate_boundary_f1,
    get_instance_count_metrics, get_confusion_matrix, per_class_iou_from_cm,
)

logger = logging.getLogger("DeepLabV3_FineTune")

# Mặc định: 3 checkpoint (điền đường dẫn sau khi train xong, ví dụ best + 2 epoch gần best)
LIST_CHECKPOINTS = [
    os.path.join(CFG.RUNS_DIR, "exp20", "weights", "best.pth"),
    # "runs/train/exp18/weights/ep30.pth",
    # "runs/train/exp18/weights/ep33.pth",
]


def parse_rare_threshold(s: str):
    """Parse \"11=0.25,12=0.3\" -> {11: 0.25, 12: 0.3}."""
    out = {}
    if not (s or "").strip():
        return out
    for part in s.strip().split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[int(k.strip())] = float(v.strip())
    return out


def apply_rare_thresholds(logits: torch.Tensor, rare_threshold: dict, num_classes: int) -> torch.Tensor:
    """
    Giảm ngưỡng quyết định cho class hiếm: nếu prob[rare] >= threshold và > prob[pred] thì gán rare.
    logits: (B, C, H, W). rare_threshold: {class_id: min_prob}.
    """
    if not rare_threshold:
        return torch.argmax(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    B, C, H, W = probs.shape
    pred_flat = pred.unsqueeze(1)
    probs_at_pred = torch.gather(probs, 1, pred_flat).squeeze(1)
    for c in sorted(rare_threshold.keys()):
        if c < 0 or c >= num_classes:
            continue
        t = rare_threshold[c]
        low_bar = probs[:, c] >= t
        better = probs[:, c] > probs_at_pred
        override = low_bar & better
        pred[override] = c
        probs_at_pred = torch.gather(probs, 1, pred.unsqueeze(1)).squeeze(1)
    return pred


def main():
    parser = argparse.ArgumentParser(description="Ensemble eval: average logits từ N checkpoint")
    parser.add_argument("--checkpoints", nargs="+", default=LIST_CHECKPOINTS, help="Đường dẫn các .pth")
    parser.add_argument("--run-dir", type=str, default=None, help="Thư mục ghi per_class_iou.csv (mặc định: runs/train/exp_ensemble)")
    parser.add_argument("--tta", action="store_true", help="Bật TTA (flip) khi predict mỗi model")
    parser.add_argument("--rare-threshold", type=str, default="", help="Ngưỡng prob cho class hiếm, VD: 11=0.25 (tofu).")
    args = parser.parse_args()

    checkpoints = [c for c in args.checkpoints if c and os.path.isfile(c)]
    if not checkpoints:
        logger.warning("Không tìm thấy file checkpoint nào. Thoát.")
        sys.exit(1)
    logger.info(f"Ensemble {len(checkpoints)} checkpoint: {checkpoints}")

    run_dir = args.run_dir or os.path.join(CFG.RUNS_DIR, "exp_ensemble")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "eval_ensemble.log")
    setup_logging(log_file=log_file)
    init_environment(seed=CFG.SEED)
    device = get_device()

    # DataLoader val
    val_ds = FoodSegDataset(
        CFG.VAL_IMG_DIR, CFG.VAL_LABEL_DIR,
        CFG.NUM_CLASSES, raw_to_seq=getattr(CFG, "RAW_TO_SEQ", None),
        split="val", img_size=CFG.IMG_SIZE,
        skip_resize=getattr(CFG, "USE_RESIZED_DATA", False),
    )
    val_batch = getattr(CFG, "VAL_BATCH_SIZE", 8)
    val_loader = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        num_workers=getattr(CFG, "NUM_WORKERS", 4),
        pin_memory=(device.type == "cuda"),
    )

    # Tạo N model, load N checkpoint
    use_imagenet = getattr(CFG, "USE_IMAGENET", True)
    models = []
    for path in checkpoints:
        m = create_model(CFG.NUM_CLASSES, use_imagenet=True) if use_imagenet else create_model(
            CFG.NUM_CLASSES, pretrained_weights_path=CFG.PRETRAINED_PATH, old_num_classes=CFG.OLD_NUM_CLASSES
        )
        state = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(state, strict=True)
        m = m.to(device).eval()
        models.append(m)

    def predict_batch(images):
        """Average logits từ tất cả model. Option TTA: mỗi model dùng flip average."""
        with torch.no_grad():
            if args.tta:
                logits_list = []
                for m in models:
                    l0 = m(images)
                    l1 = torch.flip(m(torch.flip(images, dims=[3])), dims=[3])
                    l2 = torch.flip(m(torch.flip(images, dims=[2])), dims=[2])
                    logits_list.append((l0 + l1 + l2) / 3.0)
            else:
                logits_list = [m(images) for m in models]
            return torch.stack(logits_list, dim=0).mean(dim=0)

    all_miou, all_px, all_bf1 = [], [], []
    all_ema, all_mace = [], []
    total_cm = torch.zeros(CFG.NUM_CLASSES, CFG.NUM_CLASSES, dtype=torch.long)

    rare_threshold = parse_rare_threshold(args.rare_threshold)
    if rare_threshold:
        logger.info(f"Rare-class thresholds (inference): {rare_threshold}")

    for images, masks in tqdm(val_loader, desc="Ensemble Eval", bar_format="{l_bar}{bar:30}{r_bar}"):
        images, masks = images.to(device), masks.to(device)
        logits = predict_batch(images)
        preds = apply_rare_thresholds(logits, rare_threshold, CFG.NUM_CLASSES).cpu()
        masks_np = masks.cpu()
        all_miou.append(calculate_miou(preds, masks_np, CFG.NUM_CLASSES))
        all_px.append(calculate_pixel_accuracy(preds, masks_np))
        all_bf1.append(calculate_boundary_f1(preds, masks_np, CFG.NUM_CLASSES))
        total_cm += get_confusion_matrix(preds, masks_np, CFG.NUM_CLASSES)
        ema, mace = get_instance_count_metrics(preds, masks_np, CFG.NUM_CLASSES)
        all_ema.append(ema)
        all_mace.append(mace)

    logger.info(f"  mIoU            = {np.mean(all_miou):.4f}")
    logger.info(f"  Pixel Accuracy  = {np.mean(all_px):.4f}")
    logger.info(f"  Boundary F1     = {np.mean(all_bf1):.4f}")
    logger.info(f"  Exact Match Acc = {np.mean(all_ema):.4f}")
    logger.info(f"  MACE            = {np.mean(all_mace):.4f}")

    per_class = per_class_iou_from_cm(total_cm, CFG.NUM_CLASSES)
    class_names = getattr(CFG, "CLASS_NAMES", [f"class_{i}" for i in range(CFG.NUM_CLASSES)])
    per_class_file = os.path.join(run_dir, "per_class_iou_ensemble.csv")
    with open(per_class_file, "w", encoding="utf-8") as f:
        f.write("class_id,class_name,iou\n")
        for c in range(1, CFG.NUM_CLASSES):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            iou = per_class.get(c, 0.0)
            f.write(f"{c},{name},{iou:.4f}\n")
    logger.info(f"  Per-class IoU đã lưu: {per_class_file}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()

