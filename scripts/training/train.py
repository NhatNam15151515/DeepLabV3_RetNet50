import os
import sys

# Đảm bảo project root (chứa config.py, dataset.py, ...) nằm trên sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as CFG
from utils import init_environment, get_device, get_criterion_phase1, get_criterion_phase2, setup_logging
from model_setup import create_model, freeze_encoder, unfreeze_encoder, log_trainable_params
from dataset import FoodSegDataset, calculate_class_weights
from metrics import (
    calculate_miou, calculate_pixel_accuracy,
    calculate_boundary_f1, get_instance_count_metrics,
    get_confusion_matrix, per_class_iou_from_cm,
)

logger = logging.getLogger("DeepLabV3_FineTune")


# =================================================================
# RUN DIRECTORY
# =================================================================
def setup_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    exp_num = 1
    while True:
        exp_dir = os.path.join(base_dir, f"exp{exp_num}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            return exp_dir
        exp_num += 1


# TRAINING LOOP
def run_phase(
    phase_name: str,
    model, train_loader, val_loader,
    optimizer, criterion,
    num_epochs: int,
    device: torch.device,
    num_classes: int,
    patience: int,
    save_path: str,
    scheduler=None,
    scheduler_step_with_metric: bool = True,
    use_amp: bool = True,
    csv_file: str = None,
    use_ema: bool = False,
    ema_decay: float = 0.999,
):
    """Train-validate cho 1 phase. Returns best_miou. Nếu use_ema=True thì validate bằng EMA và lưu best = EMA."""
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_miou = 0.0
    patience_counter = 0

    # EMA: shadow copy của weights (chỉ dùng khi use_ema, thường P2)
    ema_state = None
    if use_ema:
        ema_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, num_epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        grad_norms = []

        pbar = tqdm(
            train_loader,
            desc=f"[{phase_name}] Epoch {epoch}/{num_epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            leave=True,
        )

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
            
            # Compute loss in float32 to prevent NaN issues with AMP
            loss = criterion(logits.float(), masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP_MAX_NORM)
            grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else float(gn))

            scaler.step(optimizer)
            scaler.update()

            # Cập nhật EMA sau mỗi step (chỉ params float; buffer như BN copy từ model)
            if use_ema and ema_state is not None:
                for k, v in model.state_dict().items():
                    if v.dtype in (torch.float32, torch.float16):
                        ema_state[k] = ema_decay * ema_state[k].to(device) + (1 - ema_decay) * v.detach().float()
                        ema_state[k] = ema_state[k].cpu()
                    else:
                        ema_state[k] = v.detach().cpu().clone()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", gn=f"{gn:.1f}")

        avg_loss = running_loss / max(len(train_loader), 1)
        avg_gn = float(np.mean(grad_norms)) if grad_norms else 0.0
        scale_val = scaler.get_scale() if use_amp else 1.0

        # ---------- VALIDATE ----------
        model.eval()
        # Swap EMA vào model để validation; sau val bắt buộc trả lại weights gốc để tiếp tục training
        model_save_state = None
        if use_ema and ema_state is not None:
            model_save_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(device) for k, v in ema_state.items()}, strict=True)

        val_mious, val_px_accs, val_bf1s = [], [], []
        compute_bf1 = (epoch % 5 == 0) or (epoch == num_epochs)

        try:
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"  Val", bar_format="{l_bar}{bar:20}{r_bar}", leave=False):
                    images = images.to(device)
                    masks = masks.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    val_mious.append(calculate_miou(preds, masks, num_classes))
                    val_px_accs.append(calculate_pixel_accuracy(preds, masks))
                    if compute_bf1:
                        val_bf1s.append(calculate_boundary_f1(preds, masks, num_classes))
        finally:
            # Luôn trả lại weights gốc sau val để bước train tiếp theo dùng đúng params
            if model_save_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in model_save_state.items()}, strict=True)

        val_miou = float(np.mean(val_mious)) if val_mious else 0.0
        val_px = float(np.mean(val_px_accs)) if val_px_accs else 0.0
        val_bf1 = float(np.mean(val_bf1s)) if val_bf1s else -1.0

        bf1_str = f"{val_bf1:.4f}" if compute_bf1 else "—"
        logger.info(
            f"[{phase_name}] Epoch {epoch}/{num_epochs} | "
            f"Loss={avg_loss:.4f}  GradNorm={avg_gn:.2f}  Scale={scale_val:.0f} | "
            f"mIoU={val_miou:.4f}  PixAcc={val_px:.4f}  BF1={bf1_str}"
        )

        if scheduler is not None:
            if scheduler_step_with_metric:
                scheduler.step(val_miou)
            else:
                scheduler.step()

        if csv_file:
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write("phase,epoch,train_loss,grad_norm,val_miou,val_pix_acc,val_bf1\n")
                bf1_csv = f"{val_bf1:.4f}" if compute_bf1 else ""
                f.write(f"{phase_name},{epoch},{avg_loss:.4f},{avg_gn:.2f},{val_miou:.4f},{val_px:.4f},{bf1_csv}\n")

        # --- Early stopping & best model ---
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            if use_ema and ema_state is not None:
                torch.save({k: v.cpu().clone() for k, v in ema_state.items()}, save_path)
            else:
                torch.save(model.state_dict(), save_path)
            logger.info(f"  → Best mIoU={best_miou:.4f} — saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  → Early stopping tại epoch {epoch}.")
                break

    return best_miou


# =================================================================
# MAIN
# =================================================================
def main():
    run_dir = setup_run_dir(CFG.RUNS_DIR)
    save_path = os.path.join(run_dir, "weights", "best.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")
    csv_file = os.path.join(run_dir, "results.csv")

    setup_logging(log_file=log_file)
    logger.info(f"Khởi tạo thư mục chạy: {run_dir}")
    init_environment(seed=CFG.SEED)
    device = get_device()

    # --- Class weights ---
    class_weights = calculate_class_weights(
        label_dir=CFG.TRAIN_LABEL_DIR,
        image_dir=CFG.TRAIN_IMG_DIR,
        num_classes=CFG.NUM_CLASSES,
        raw_to_seq=getattr(CFG, "RAW_TO_SEQ", None),
    )
    # Tinh chỉnh theo per_class_iou.csv: class IoU thấp có thể tăng weight qua CLASS_WEIGHT_BOOST
    # Factor giới hạn [1.0, 2.0]. Chuyển sang numpy để nhân an toàn (list/tensor → array).
    boost = getattr(CFG, "CLASS_WEIGHT_BOOST", {})
    if boost:
        if isinstance(class_weights, torch.Tensor):
            w = class_weights.cpu().numpy().astype(np.float64)
        else:
            w = np.asarray(class_weights, dtype=np.float64)
        for c, factor in boost.items():
            if 0 <= c < CFG.NUM_CLASSES:
                f = max(1.0, min(2.0, float(factor)))
                w[c] = w[c] * f
        class_weights = torch.tensor(w, dtype=torch.float32)
        logger.info(f"Đã áp dụng CLASS_WEIGHT_BOOST (factor clamp 1.0–2.0) cho class: {list(boost.keys())}")

    # --- DataLoaders ---
    train_ds = FoodSegDataset(
        CFG.TRAIN_IMG_DIR, CFG.TRAIN_LABEL_DIR,
        CFG.NUM_CLASSES, raw_to_seq=getattr(CFG, "RAW_TO_SEQ", None),
        split="train", img_size=CFG.IMG_SIZE,
        skip_resize=getattr(CFG, "USE_RESIZED_DATA", False),
    )
    val_ds = FoodSegDataset(
        CFG.VAL_IMG_DIR, CFG.VAL_LABEL_DIR,
        CFG.NUM_CLASSES, raw_to_seq=getattr(CFG, "RAW_TO_SEQ", None),
        split="val", img_size=CFG.IMG_SIZE,
        skip_resize=getattr(CFG, "USE_RESIZED_DATA", False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # tránh batch size = 1 gây lỗi BatchNorm trong ASPP
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_batch = getattr(CFG, "VAL_BATCH_SIZE", CFG.BATCH_SIZE)
    val_loader = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --- Model ---
    use_imagenet = getattr(CFG, "USE_IMAGENET", False)
    if use_imagenet:
        model = create_model(CFG.NUM_CLASSES, use_imagenet=True)
    else:
        model = create_model(
            CFG.NUM_CLASSES,
            pretrained_weights_path=CFG.PRETRAINED_PATH,
            old_num_classes=CFG.OLD_NUM_CLASSES,
        )
    model = model.to(device)

    # --- AMP toggle ---
    use_amp = getattr(CFG, "USE_AMP", True)

    # ======================== PHASE 1 ========================
    logger.info("=" * 60)
    logger.info("PHASE 1 — LINEAR PROBING  (Freeze Encoder)")
    logger.info("=" * 60)
    freeze_encoder(model)
    log_trainable_params(model)

    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.P1_LR,
    )
    label_smoothing = getattr(CFG, "LABEL_SMOOTHING", 0.0)
    loss1 = get_criterion_phase1(class_weights, device, label_smoothing=label_smoothing)

    best1 = run_phase(
        "P1", model, train_loader, val_loader,
        opt1, loss1,
        num_epochs=CFG.P1_EPOCHS,
        device=device,
        num_classes=CFG.NUM_CLASSES,
        patience=CFG.P1_PATIENCE,
        save_path=save_path,
        csv_file=csv_file,
        use_amp=use_amp,
    )
    logger.info(f"Phase 1 xong — Best mIoU = {best1:.4f}")

    # ======================== PHASE 2 ========================
    logger.info("=" * 60)
    logger.info("PHASE 2 — DIFFERENTIAL FINE-TUNING  (Unfreeze All)")
    logger.info("=" * 60)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        logger.info("Loaded best Phase 1 weights.")

    unfreeze_encoder(model)
    log_trainable_params(model)

    opt2 = torch.optim.Adam([
        {"params": model.encoder.parameters(),          "lr": CFG.P2_LR_ENCODER},
        {"params": model.decoder.parameters(),          "lr": CFG.P2_LR_DECODER},
        {"params": model.segmentation_head.parameters(),"lr": CFG.P2_LR_HEAD},
    ], weight_decay=CFG.P2_WEIGHT_DECAY)

    use_cosine = getattr(CFG, "P2_USE_COSINE", False)
    warmup_epochs = getattr(CFG, "P2_WARMUP_EPOCHS", 2)
    if use_cosine:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(opt2, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(opt2, T_max=CFG.P2_EPOCHS - warmup_epochs, eta_min=1e-6)
        sched2 = SequentialLR(opt2, [warmup, cosine], milestones=[warmup_epochs])
        sched_step_with_metric = False
        logger.info(f"P2 Scheduler: Warmup {warmup_epochs} ep + CosineAnnealing")
    else:
        sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt2, mode="max",
            patience=CFG.P2_SCHEDULER_PATIENCE,
            factor=CFG.P2_SCHEDULER_FACTOR,
        )
        sched_step_with_metric = True

    loss2 = get_criterion_phase2(class_weights, device)

    p2_use_ema = getattr(CFG, "P2_USE_EMA", False)
    p2_ema_decay = getattr(CFG, "P2_EMA_DECAY", 0.999)
    best2 = run_phase(
        "P2", model, train_loader, val_loader,
        opt2, loss2,
        num_epochs=CFG.P2_EPOCHS,
        device=device,
        num_classes=CFG.NUM_CLASSES,
        patience=CFG.P2_PATIENCE,
        save_path=save_path,
        scheduler=sched2,
        scheduler_step_with_metric=sched_step_with_metric,
        csv_file=csv_file,
        use_amp=use_amp,
        use_ema=p2_use_ema,
        ema_decay=p2_ema_decay,
    )
    logger.info(f"Phase 2 xong — Best mIoU = {best2:.4f}")

    # ======================== FINAL EVAL + TTA ========================
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION trên tập Validation (với TTA)")
    logger.info("=" * 60)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()

    all_miou, all_px, all_bf1 = [], [], []
    all_ema, all_mace = [], []
    total_cm = torch.zeros(CFG.NUM_CLASSES, CFG.NUM_CLASSES, dtype=torch.long)

    tta_scales = getattr(CFG, "TTA_SCALES", [1.0])
    if not tta_scales:
        tta_scales = [1.0]
    tta_hflip = getattr(CFG, "TTA_HFLIP", True)
    tta_vflip = getattr(CFG, "TTA_VFLIP", True)

    # SMP (DeepLabV3+) yêu cầu H, W chia hết cho 16
    TTA_ALIGN = 16

    def tta_predict(model, images, device, use_amp_flag):
        """TTA tham số hóa: scales + hflip/vflip. Mỗi biến thể 1 forward trên cả batch → gom logits → mean."""
        B, _, H, W = images.shape
        logits_list = []

        with torch.amp.autocast("cuda", enabled=use_amp_flag and device.type == "cuda"):
            for scale in tta_scales:
                if scale == 1.0:
                    inp = images
                else:
                    # Resize nhưng làm tròn H,W về bội của 16 để SMP không báo lỗi
                    new_h = max(TTA_ALIGN, int(round(H * scale / TTA_ALIGN) * TTA_ALIGN))
                    new_w = max(TTA_ALIGN, int(round(W * scale / TTA_ALIGN) * TTA_ALIGN))
                    inp = torch.nn.functional.interpolate(
                        images, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                # Original (hoặc scale khác 1.0)
                out = model(inp)
                if scale != 1.0:
                    out = torch.nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
                logits_list.append(out)
                # Flips chỉ tại scale 1.0 để đổi tốc độ/độ chính xác
                if scale == 1.0:
                    if tta_hflip:
                        logits_list.append(torch.flip(model(torch.flip(inp, dims=[3])), dims=[3]))
                    if tta_vflip:
                        logits_list.append(torch.flip(model(torch.flip(inp, dims=[2])), dims=[2]))

            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return logits

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Final Eval (TTA)", bar_format="{l_bar}{bar:30}{r_bar}"):
            images, masks = images.to(device), masks.to(device)
            logits = tta_predict(model, images, device, use_amp)
            preds = torch.argmax(logits, dim=1)
            all_miou.append(calculate_miou(preds, masks, CFG.NUM_CLASSES))
            all_px.append(calculate_pixel_accuracy(preds, masks))
            all_bf1.append(calculate_boundary_f1(preds, masks, CFG.NUM_CLASSES))
            total_cm += get_confusion_matrix(preds, masks, CFG.NUM_CLASSES).cpu()
            ema, mace = get_instance_count_metrics(preds, masks, CFG.NUM_CLASSES)
            all_ema.append(ema)
            all_mace.append(mace)

    logger.info(f"  mIoU            = {np.mean(all_miou):.4f}")
    logger.info(f"  Pixel Accuracy  = {np.mean(all_px):.4f}")
    logger.info(f"  Boundary F1     = {np.mean(all_bf1):.4f}")
    logger.info(f"  Exact Match Acc = {np.mean(all_ema):.4f}")
    logger.info(f"  MACE            = {np.mean(all_mace):.4f}")

    # Per-class IoU (từ confusion matrix tích lũy)
    per_class = per_class_iou_from_cm(total_cm, CFG.NUM_CLASSES)
    class_names = getattr(CFG, "CLASS_NAMES", [f"class_{i}" for i in range(CFG.NUM_CLASSES)])
    per_class_file = os.path.join(run_dir, "per_class_iou.csv")
    with open(per_class_file, "w", encoding="utf-8") as f:
        f.write("class_id,class_name,iou\n")
        for c in range(1, CFG.NUM_CLASSES):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            iou = per_class.get(c, 0.0)
            f.write(f"{c},{name},{iou:.4f}\n")
            logger.info(f"  IoU class {c} ({name}) = {iou:.4f}")
    logger.info(f"  Per-class IoU đã lưu: {per_class_file}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()

