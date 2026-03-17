import os
import sys

# Đảm bảo project root (chứa config.py, dataset.py, ...) nằm trên sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

"""
finetune_from_checkpoint.py

Fine-tune chắc chắn từ 1 checkpoint tốt nhất, KHÔNG đụng train.py/config.py.

Ví dụ (sau khi gom script vào thư mục scripts):
  python scripts/training/finetune_from_checkpoint.py
"""

import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import config as CFG
from utils import init_environment, get_device, get_criterion_phase2, setup_logging
from model_setup import create_model
from dataset import FoodSegDataset, calculate_class_weights
from metrics import (
    calculate_miou,
    calculate_pixel_accuracy,
    calculate_boundary_f1,
    get_instance_count_metrics,
    get_confusion_matrix,
    per_class_iou_from_cm,
)

logger = logging.getLogger("DeepLabV3_FineTune")


def _count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def log_model_info(model: torch.nn.Module) -> None:
    """
    In/log đầy đủ thông tin model trước khi train:
    - Kiến trúc model (repr)
    - Tổng params / trainable params
    - Breakdown encoder/decoder/head nếu có (SMP)
    """
    total, trainable = _count_params(model)
    logger.info("=" * 60)
    logger.info("MODEL SUMMARY (trước khi train)")
    logger.info("=" * 60)
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Total params: {total:,} | Trainable params: {trainable:,} ({(100.0*trainable/max(total,1)):.2f}%)")

    # SMP modules (nếu có)
    for name in ("encoder", "decoder", "segmentation_head"):
        if hasattr(model, name):
            mod = getattr(model, name)
            t, tr = _count_params(mod)
            logger.info(f"{name:17s}: total={t:,} trainable={tr:,}")
        else:
            logger.warning(f"Model không có attribute '{name}' (nếu đổi backbone/arch cần kiểm tra optimizer groups).")


def log_run_info(
    *,
    checkpoint_path: str,
    out_dir: str,
    resume_mode: bool,
    start_epoch: int,
    epochs: int,
    lr_encoder: float,
    lr_decoder: float,
    lr_head: float,
    weight_decay: float,
    use_ema: bool,
    ema_decay: float,
    boost: Dict[int, float],
    oversample_rare: bool,
    strong_rare_aug: bool,
    tta_scales: List[float],
    tta_hflip: bool,
    tta_vflip: bool,
) -> None:
    logger.info("=" * 60)
    logger.info("RUN CONFIG (trước khi train)")
    logger.info("=" * 60)
    logger.info(f"Resume mode      : {resume_mode}")
    logger.info(f"Checkpoint input : {checkpoint_path}")
    logger.info(f"Output dir       : {out_dir}")
    logger.info(f"Epoch range      : {start_epoch} -> {epochs}")
    logger.info(f"LR encoder       : {lr_encoder:g}")
    logger.info(f"LR decoder       : {lr_decoder:g}")
    logger.info(f"LR head          : {lr_head:g}")
    logger.info(f"Weight decay     : {weight_decay:g}")
    logger.info(f"EMA              : {use_ema} (decay={ema_decay:g})")
    logger.info(f"Boost (clamp)    : {boost if boost else '{}'}")
    logger.info(f"Oversample rare  : {oversample_rare}")
    logger.info(f"Strong rare aug  : {strong_rare_aug}")
    logger.info(f"TTA scales       : {tta_scales}")
    logger.info(f"TTA hflip/vflip  : {tta_hflip}/{tta_vflip}")
    logger.info("=" * 60)


def setup_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    exp_num = 1
    while True:
        exp_dir = os.path.join(base_dir, f"ft{exp_num}")
        if not os.path.exists(exp_dir):
            os.path.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
            return exp_dir
        exp_num += 1


def resolve_output_root_from_checkpoint(checkpoint_path: str) -> str:
    """
    Lưu fine-tune ngay trong folder run của checkpoint:
      runs/train/exp19/weights/best.pth  ->  runs/train/exp19/finetune/
    """
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))          # .../weights
    run_dir = os.path.dirname(ckpt_dir)                                   # .../exp19
    return os.path.join(run_dir, "finetune")


def find_latest_best_checkpoint() -> str:
    """
    Tự tìm best.pth mới nhất theo expN trong runs/train/expN/weights/best.pth.
    """
    base = os.path.join(CFG.PROJECT_ROOT, "runs", "train")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Không thấy runs/train tại: {base}")
    exps = []
    for name in os.listdir(base):
        if name.startswith("exp") and name[3:].isdigit():
            exps.append((int(name[3:]), os.path.join(base, name)))
    exps.sort(reverse=True, key=lambda x: x[0])
    for _, exp_dir in exps:
        cand = os.path.join(exp_dir, "weights", "best.pth")
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("Không tìm thấy best.pth nào trong runs/train/exp*/weights/.")


def parse_boost(s: str) -> Dict[int, float]:
    """
    Parse string dạng: "1=2,4=1.5,25=2"
    """
    out: Dict[int, float] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Boost token '{p}' không đúng format id=factor")
        k, v = p.split("=", 1)
        out[int(k.strip())] = float(v.strip())
    return out


def clamp_boost(boost: Dict[int, float], lo: float = 1.0, hi: float = 2.0) -> Dict[int, float]:
    return {int(k): max(lo, min(hi, float(v))) for k, v in boost.items()}


def _strip_wrapping_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", "\"")):
        return s[1:-1].strip()
    return s


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    path: str,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> None:
    """Lưu confusion matrix dạng ảnh heatmap để trực quan hóa nhầm lẫn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib không có → bỏ qua lưu heatmap confusion matrix.")
        return
    n = cm.shape[0]
    if num_classes is not None:
        n = min(n, num_classes)
    cm = cm[:n, :n].astype(np.float64)
    cm_vis = np.log1p(cm)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), max(8, n * 0.3)))
    im = ax.imshow(cm_vis, aspect="auto", cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if class_names and len(class_names) >= n:
        ticks = list(range(n))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([class_names[i][:12] for i in ticks], rotation=45, ha="right")
        ax.set_yticklabels([class_names[i][:12] for i in ticks])
    plt.colorbar(im, ax=ax, label="log1p(count)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"  Confusion matrix heatmap: {path}")


def build_rare_oversampler(
    label_dir: str,
    image_dir: str,
    num_classes: int,
    rare_classes: List[int],
) -> Tuple[WeightedRandomSampler, np.ndarray]:
    """
    Oversample ảnh có chứa rare_classes.
    Trả về sampler + weights theo ảnh (để debug).
    """
    import cv2

    alpha = 2.0
    images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    weights = np.ones(len(images), dtype=np.float64)
    rare_set = set(int(c) for c in rare_classes)

    for i, img_name in enumerate(images):
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(label_dir, mask_name)
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim > 2:
            m = m[:, :, 0]
        m = np.clip(m.astype(np.int64), 0, num_classes - 1)
        uniq = np.unique(m)
        present = sum((int(u) in rare_set) for u in uniq)
        if present > 0:
            weights[i] = 1.0 + alpha * present

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, weights


def tta_predict(
    model,
    images: torch.Tensor,
    use_amp: bool,
    scales: List[float],
    hflip: bool,
    vflip: bool,
    align: int = 16,
) -> torch.Tensor:
    """
    TTA cho DeepLabV3+ (SMP): đảm bảo input H,W chia hết cho 16 khi scale != 1.0
    """
    _, _, H, W = images.shape
    logits_list = []

    with torch.cuda.amp.autocast(enabled=use_amp):
        for s in scales:
            if s == 1.0:
                inp = images
            else:
                new_h = max(align, int(round(H * s / align) * align))
                new_w = max(align, int(round(W * s / align) * align))
                inp = torch.nn.functional.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

            out = model(inp)
            if s != 1.0:
                out = torch.nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            logits_list.append(out)

            if s == 1.0:
                if hflip:
                    logits_list.append(torch.flip(model(torch.flip(inp, dims=[3])), dims=[3]))
                if vflip:
                    logits_list.append(torch.flip(model(torch.flip(inp, dims=[2])), dims=[2]))

    return torch.stack(logits_list, dim=0).mean(dim=0)


def run_finetune(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device: torch.device,
    num_classes: int,
    epochs: int,
    use_amp: bool,
    use_ema: bool,
    ema_decay: float,
    out_dir: str,
    tta_scales: List[float],
    tta_hflip: bool,
    tta_vflip: bool,
    start_epoch: int = 1,
    initial_best_miou: float = -1.0,
    ema_state_init: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_miou = initial_best_miou

    ema_state = None
    if use_ema:
        if ema_state_init is not None:
            ema_state = {k: v.to(device) for k, v in ema_state_init.items()}
        else:
            ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    results_csv = os.path.join(out_dir, "finetune_results.csv")
    write_header = not (start_epoch > 1 and os.path.isfile(results_csv))
    with open(results_csv, "a" if start_epoch > 1 else "w", encoding="utf-8") as f:
        if write_header:
            f.write("epoch,train_loss,val_miou,val_pix_acc,val_bf1\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"[FT] Epoch {epoch}/{epochs}", bar_format="{l_bar}{bar:30}{r_bar}", leave=True)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
            loss = criterion(logits.float(), masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(CFG, "GRAD_CLIP_MAX_NORM", 0.5))
            scaler.step(optimizer)
            scaler.update()

            if use_ema and ema_state is not None:
                for k, v in model.state_dict().items():
                    if v.dtype in (torch.float32, torch.float16):
                        ema_state[k] = ema_decay * ema_state[k] + (1 - ema_decay) * v.detach().float()
                    else:
                        ema_state[k] = v.detach().clone()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(len(train_loader), 1)

        model.eval()
        model_save_state = None
        if use_ema and ema_state is not None:
            model_save_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state, strict=True)

        total_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        val_mious, val_px, val_bf1 = [], [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="  Val", bar_format="{l_bar}{bar:20}{r_bar}", leave=False):
                images = images.to(device)
                masks = masks.to(device)
                if tta_scales and (len(tta_scales) > 1 or tta_hflip or tta_vflip):
                    logits = tta_predict(model, images, use_amp, tta_scales, tta_hflip, tta_vflip)
                else:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(images)
                preds = torch.argmax(logits, dim=1)
                val_mious.append(calculate_miou(preds, masks, num_classes))
                val_px.append(calculate_pixel_accuracy(preds, masks))
                val_bf1.append(calculate_boundary_f1(preds, masks, num_classes))
                total_cm += get_confusion_matrix(preds, masks, num_classes).cpu()

        if model_save_state is not None:
            model.load_state_dict(model_save_state, strict=True)

        miou = float(np.mean(val_mious)) if val_mious else 0.0
        pix = float(np.mean(val_px)) if val_px else 0.0
        bf1 = float(np.mean(val_bf1)) if val_bf1 else 0.0

        logger.info(f"[FT] Epoch {epoch}/{epochs} | TrainLoss={train_loss:.4f} | mIoU={miou:.4f} PixAcc={pix:.4f} BF1={bf1:.4f}")

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{miou:.6f},{pix:.6f},{bf1:.6f}\n")

        per_class = per_class_iou_from_cm(total_cm, num_classes)
        class_names = getattr(CFG, "CLASS_NAMES", [f"class_{i}" for i in range(num_classes)])
        per_class_path = os.path.join(out_dir, f"per_class_iou_ft_ep{epoch:02d}.csv")
        with open(per_class_path, "w", encoding="utf-8") as f:
            f.write("class_id,class_name,iou\n")
            for c in range(1, num_classes):
                name = class_names[c] if c < len(class_names) else f"class_{c}"
                f.write(f"{c},{name},{per_class.get(c, 0.0):.4f}\n")

        cm_np = total_cm.numpy()
        np.save(os.path.join(out_dir, f"confusion_matrix_ft_ep{epoch:02d}.npy"), cm_np)
        save_confusion_matrix_heatmap(
            cm_np,
            os.path.join(out_dir, f"confusion_matrix_ft_ep{epoch:02d}.png"),
            class_names=class_names,
            num_classes=num_classes,
        )

        if miou > best_miou:
            best_miou = miou
            best_path = os.path.join(out_dir, "weights", "best_ft.pth")
            if use_ema and ema_state is not None:
                torch.save({k: v.cpu().clone() for k, v in ema_state.items()}, best_path)
            else:
                torch.save(model.state_dict(), best_path)
            logger.info(f"  → Best FT mIoU={best_miou:.4f} — saved {best_path}")

    last_path = os.path.join(out_dir, "weights", "last_ft.pth")
    last_ckpt = {
        "epoch": epoch,
        "best_miou": best_miou,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if use_ema and ema_state is not None:
        last_ckpt["ema_state"] = {k: v.cpu().clone() for k, v in ema_state.items()}
    torch.save(last_ckpt, last_path)
    logger.info(f"Saved last (resumable): {last_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune từ checkpoint tốt nhất (an toàn, không sửa train/config).")
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn checkpoint .pth (bỏ qua nếu dùng --resume).")
    parser.add_argument("--resume", type=str, default="", help="Tiếp tục từ run trước: folder finetune (VD: runs/train/exp19/finetune/ft1).")
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-decoder", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)

    parser.add_argument("--ema", dest="ema", action="store_true", help="Bật EMA khi fine-tune (default: bật).")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="Tắt EMA.")
    parser.set_defaults(ema=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    cfg_boost = getattr(CFG, "CLASS_WEIGHT_BOOST", None) or {}
    default_boost_str = ",".join(f"{k}={v}" for k, v in sorted(cfg_boost.items())) if cfg_boost else ""
    parser.add_argument(
        "--boost",
        type=str,
        default=default_boost_str,
        help="Boost class weight: \"11=2,...\" (clamp 1.0–2.0). Default: from config.CLASS_WEIGHT_BOOST.",
    )
    parser.add_argument("--oversample-rare", action="store_true", help="Oversample ảnh chứa class trong boost.")
    parser.add_argument("--strong-rare-aug", action="store_true", help="Augmentation mạnh hơn cho ảnh chứa class hiếm.")

    parser.add_argument("--tta-scales", type=str, default="1.0", help="VD: \"1.0\" hoặc \"0.9,1.0,1.1\" (default: 1.0)")
    parser.add_argument("--hflip", dest="hflip", action="store_true", help="Bật TTA horizontal flip (default: bật).")
    parser.add_argument("--no-hflip", dest="hflip", action="store_false", help="Tắt TTA horizontal flip.")
    parser.add_argument("--vflip", dest="vflip", action="store_true", help="Bật TTA vertical flip (default: bật).")
    parser.add_argument("--no-vflip", dest="vflip", action="store_false", help="Tắt TTA vertical flip.")
    parser.set_defaults(hflip=True, vflip=True)

    args = parser.parse_args()

    resume_mode = bool(args.resume.strip())
    if resume_mode:
        out_dir = os.path.abspath(args.resume.strip())
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(f"--resume folder không tồn tại: {out_dir}")
        last_ckpt_path = os.path.join(out_dir, "weights", "last_ft.pth")
        if not os.path.isfile(last_ckpt_path):
            raise FileNotFoundError(f"Không tìm thấy {last_ckpt_path} để resume.")
    else:
        if not args.checkpoint:
            args.checkpoint = find_latest_best_checkpoint()
            logger.info(f"Tự chọn checkpoint mới nhất: {args.checkpoint}")
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError("Cần --checkpoint (file .pth) hoặc --resume (folder finetune).")
        out_root = resolve_output_root_from_checkpoint(args.checkpoint)
        out_dir = setup_run_dir(out_root)

    setup_logging(log_file=os.path.join(out_dir, "finetune.log"))
    init_environment(seed=getattr(CFG, "SEED", 42))
    device = get_device()

    use_amp = bool(getattr(CFG, "USE_AMP", False)) and device.type == "cuda"

    tta_scales_raw = _strip_wrapping_quotes(args.tta_scales)
    scales = [float(_strip_wrapping_quotes(x.strip())) for x in tta_scales_raw.split(",") if x.strip()]
    if not scales:
        scales = [1.0]

    class_weights = calculate_class_weights(
        label_dir=CFG.TRAIN_LABEL_DIR,
        image_dir=CFG.TRAIN_IMG_DIR,
        num_classes=CFG.NUM_CLASSES,
        raw_to_seq=getattr(CFG, "RAW_TO_SEQ", None),
    )

    boost_str = _strip_wrapping_quotes(args.boost)
    if not boost_str:
        cfg_boost = getattr(CFG, "CLASS_WEIGHT_BOOST", None) or {}
        if cfg_boost:
            boost_str = ",".join(f"{k}={v}" for k, v in sorted(cfg_boost.items()))
    boost = clamp_boost(parse_boost(boost_str), 1.0, 2.0)
    if boost:
        w = class_weights.cpu().numpy().astype(np.float64)
        for c, f in boost.items():
            if 0 <= c < CFG.NUM_CLASSES:
                w[c] = w[c] * f
        class_weights = torch.tensor(w, dtype=torch.float32)
        logger.info(f"Boost class weights (clamp 1.0–2.0): {boost}")

    rare_ids = list(boost.keys()) if boost else []
    train_ds = FoodSegDataset(
        CFG.TRAIN_IMG_DIR, CFG.TRAIN_LABEL_DIR,
        CFG.NUM_CLASSES, raw_to_seq=getattr(CFG, "RAW_TO_SEQ", None),
        split="train", img_size=CFG.IMG_SIZE,
        skip_resize=getattr(CFG, "USE_RESIZED_DATA", False),
        rare_class_ids=rare_ids,
        strong_rare_aug=bool(args.strong_rare_aug),
    )
    val_ds = FoodSegDataset(
        CFG.VAL_IMG_DIR, CFG.VAL_LABEL_DIR,
        CFG.NUM_CLASSES, raw_to_seq=getattr(CFG, "RAW_TO_SEQ", None),
        split="val", img_size=CFG.IMG_SIZE,
        skip_resize=getattr(CFG, "USE_RESIZED_DATA", False),
    )

    sampler = None
    if args.oversample_rare and boost:
        sampler, img_w = build_rare_oversampler(
            label_dir=CFG.TRAIN_LABEL_DIR,
            image_dir=CFG.TRAIN_IMG_DIR,
            num_classes=CFG.NUM_CLASSES,
            rare_classes=sorted(boost.keys()),
        )
        np.save(os.path.join(out_dir, "oversample_image_weights.npy"), img_w)
        logger.info("Oversampling rare images: enabled.")

    train_loader = DataLoader(
        train_ds,
        batch_size=getattr(CFG, "BATCH_SIZE", 4),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=getattr(CFG, "NUM_WORKERS", 4),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=getattr(CFG, "VAL_BATCH_SIZE", 8),
        shuffle=False,
        num_workers=getattr(CFG, "NUM_WORKERS", 4),
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(CFG.NUM_CLASSES, use_imagenet=True if getattr(CFG, "USE_IMAGENET", True) else False)
    model = model.to(device)

    start_epoch = 1
    initial_best_miou = -1.0
    ema_state_init = None

    if resume_mode:
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt["epoch"]) + 1
        initial_best_miou = float(ckpt.get("best_miou", -1.0))
        ema_state_init = ckpt.get("ema_state")
        logger.info(f"Resume từ {last_ckpt_path} → epoch tiếp {start_epoch}, best_miou={initial_best_miou:.4f}")
    else:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    log_model_info(model)
    log_run_info(
        checkpoint_path=(last_ckpt_path if resume_mode else args.checkpoint),
        out_dir=out_dir,
        resume_mode=resume_mode,
        start_epoch=start_epoch,
        epochs=args.epochs,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        use_ema=bool(args.ema),
        ema_decay=args.ema_decay,
        boost=boost,
        oversample_rare=bool(args.oversample_rare),
        strong_rare_aug=bool(args.strong_rare_aug),
        tta_scales=scales,
        tta_hflip=bool(args.hflip),
        tta_vflip=bool(args.vflip),
    )

    opt = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder},
            {"params": model.segmentation_head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    if resume_mode and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        logger.info("Loaded optimizer state để resume.")

    criterion = get_criterion_phase2(class_weights, device)

    if not resume_mode:
        meta = {
            "checkpoint": args.checkpoint,
            "epochs": args.epochs,
            "lr_encoder": args.lr_encoder,
            "lr_decoder": args.lr_decoder,
            "lr_head": args.lr_head,
            "weight_decay": args.weight_decay,
            "ema": args.ema,
            "ema_decay": args.ema_decay,
            "boost": boost,
            "oversample_rare": args.oversample_rare,
            "strong_rare_aug": args.strong_rare_aug,
            "tta_scales": scales,
            "tta_hflip": args.hflip,
            "tta_vflip": args.vflip,
        }
        with open(os.path.join(out_dir, "finetune_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    run_finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=opt,
        criterion=criterion,
        device=device,
        num_classes=CFG.NUM_CLASSES,
        epochs=args.epochs,
        use_amp=use_amp,
        use_ema=args.ema,
        ema_decay=args.ema_decay,
        out_dir=out_dir,
        tta_scales=scales,
        tta_hflip=args.hflip,
        tta_vflip=args.vflip,
        start_epoch=start_epoch,
        initial_best_miou=initial_best_miou,
        ema_state_init=ema_state_init,
    )

    logger.info(f"DONE. Outputs: {out_dir}")


if __name__ == "__main__":
    main()


