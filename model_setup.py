"""
model_setup.py — Khởi tạo DeepLabV3+ (SMP), load pretrained 104-class,
replace segmentation head (Kaiming), freeze/unfreeze, log params.

Tuân thủ:
- In missing_keys / unexpected_keys và assert chỉ head mismatch
- Auto detect in_channels decoder (không hardcode 256)
- Kaiming Normal init cho head mới
- Log trainable vs total params
"""

import logging
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger("DeepLabV3_FineTune")


def _kaiming_init(module: nn.Module):
    """Khởi tạo Kaiming Normal cho Conv2d layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def _extract_state_dict(raw) -> dict:
    """
    Tự động extract model weights từ checkpoint.
    Hỗ trợ:
      - raw state_dict (key = 'conv1.weight', ...)
      - full training checkpoint (key = 'model_state_dict' hoặc 'state_dict')
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint không phải dict, mà là {type(raw)}")

    # Nếu có 'model_state_dict' hoặc 'state_dict' → extract
    for candidate in ("model_state_dict", "state_dict"):
        if candidate in raw:
            logger.info(f"Checkpoint là training state → extract '{candidate}'")
            return raw[candidate]

    # Kiểm tra xem đây có phải raw state_dict không
    # (raw state_dict thường có key = tên layer, value = tensor)
    sample_val = next(iter(raw.values()), None)
    if isinstance(sample_val, torch.Tensor):
        logger.info("Checkpoint là raw state_dict.")
        return raw

    raise ValueError(
        f"Không nhận diện được format checkpoint. Top-level keys: {list(raw.keys())[:10]}"
    )


def _auto_remap_keys(ckpt_sd: dict, model_sd: dict) -> dict:
    """
    Tự động map key names từ checkpoint (framework gốc) sang SMP.

    Chiến lược:
      1. Thử load thẳng (nếu key names trùng → trả về ngay).
      2. Thử strip common prefix từ checkpoint keys (ví dụ 'model.').
      3. Fallback: suffix matching nếu cả 2 đều không khớp.
    """
    model_keys = set(model_sd.keys())

    # --- Chiến lược 1: key names trùng trực tiếp ---
    overlap = set(ckpt_sd.keys()) & model_keys
    if len(overlap) > len(model_keys) * 0.5:
        logger.info(f"Key names trùng trực tiếp: {len(overlap)}/{len(model_sd)}")
        return ckpt_sd

    # --- Chiến lược 2: Strip common prefix ---
    # Tìm prefix chung nhất của checkpoint keys mà model keys không có
    # Ví dụ: ckpt = 'model.encoder.conv1.weight' → strip 'model.' → 'encoder.conv1.weight'
    prefixes_to_try = ["model.", "module.", "backbone.", "net."]
    for prefix in prefixes_to_try:
        stripped = {}
        match_count = 0
        for ck, cv in ckpt_sd.items():
            new_key = ck[len(prefix):] if ck.startswith(prefix) else ck
            if new_key in model_keys:
                match_count += 1
            stripped[new_key] = cv

        if match_count > len(model_keys) * 0.5:
            logger.info(f"Strip prefix '{prefix}': {match_count}/{len(model_sd)} keys matched!")
            # Validate shapes
            mapped = {}
            for sk, sv in stripped.items():
                if sk in model_sd:
                    if model_sd[sk].shape == sv.shape:
                        mapped[sk] = sv
                    else:
                        logger.warning(f"Shape mismatch: {sk} ckpt={sv.shape} vs model={model_sd[sk].shape}")
                # Keys không có trong model (ví dụ num_batches_tracked) → bỏ qua
            logger.info(f"Sau shape check: {len(mapped)} keys sẵn sàng load.")
            return mapped

    # --- Chiến lược 3: Suffix matching (fallback) ---
    logger.info("Thử suffix matching (fallback)...")
    model_suffix_idx = {}
    for mk in model_sd.keys():
        parts = mk.split(".")
        if len(parts) > 1:
            suf = ".".join(parts[1:])
            model_suffix_idx[suf] = mk

    mapped = {}
    for ck, cv in ckpt_sd.items():
        parts = ck.split(".")
        # Thử bỏ 1, 2 prefix levels
        for skip in range(1, min(3, len(parts))):
            suf = ".".join(parts[skip:])
            if suf in model_suffix_idx:
                target = model_suffix_idx[suf]
                if model_sd[target].shape == cv.shape:
                    mapped[target] = cv
                break

    logger.info(f"Suffix matching: {len(mapped)}/{len(ckpt_sd)} keys mapped.")
    return mapped


def create_model(
    num_classes: int,
    pretrained_weights_path: str = None,
    old_num_classes: int = 104,
    use_imagenet: bool = False,
) -> nn.Module:
    """
    Tạo DeepLabV3Plus ResNet50.

    Hỗ trợ 2 mode:
      A) use_imagenet=True  → encoder từ ImageNet, head mới cho num_classes
      B) pretrained_weights_path → load checkpoint FoodSeg103 rồi thay head

    Flow A (khuyến nghị):
      1. Build model (num_classes) + encoder_weights="imagenet"
      2. Done! Encoder đã có ImageNet features, head mới sẵn.

    Flow B (giữ lại tương thích):
      1. Build model (old_num_classes) + encoder_weights=None
      2. Load checkpoint → remap keys → thay head
    """
    if use_imagenet:
        # --- Mode A: ImageNet pretrained (1.2M ảnh) ---
        logger.info("Khởi tạo ResNet50 + ImageNet pretrained encoder.")
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        # Head đã đúng num_classes, init Kaiming cho head
        model.segmentation_head.apply(_kaiming_init)
        logger.info(f"ImageNet model ready → out_channels={num_classes} (Kaiming init)")
        return model

    # --- Mode B: Load custom checkpoint (flow cũ) ---
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=old_num_classes,
    )

    # --- Bước 2: Load pretrained nếu có ---
    if pretrained_weights_path:
        logger.info(f"Loading pretrained checkpoint: {pretrained_weights_path}")
        raw = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)

        # 2a. Extract state_dict thật sự
        ckpt_sd = _extract_state_dict(raw)
        logger.info(f"Checkpoint state_dict: {len(ckpt_sd)} keys")

        # 2b. Auto remap keys
        mapped_sd = _auto_remap_keys(ckpt_sd, model.state_dict())

        # 2c. Load vào model
        result = model.load_state_dict(mapped_sd, strict=False)

        if result.missing_keys:
            logger.info(f"missing_keys ({len(result.missing_keys)}): "
                        f"{result.missing_keys[:10]}{'...' if len(result.missing_keys) > 10 else ''}")
        if result.unexpected_keys:
            logger.warning(f"unexpected_keys ({len(result.unexpected_keys)}): "
                           f"{result.unexpected_keys[:10]}")

        # Assert: missing keys chỉ được là segmentation_head
        non_head_missing = [k for k in result.missing_keys
                            if not k.startswith("segmentation_head")]
        if non_head_missing:
            logger.error(
                f"CÓ {len(non_head_missing)} keys KHÔNG thuộc segmentation_head bị missing!\n"
                f"  Ví dụ: {non_head_missing[:5]}\n"
                "  → Encoder/Decoder KHÔNG được load pretrained weights!\n"
                "  → Kiểm tra lại checkpoint hoặc key mapping."
            )
            raise RuntimeError(
                f"Load checkpoint thất bại: {len(non_head_missing)} keys ngoài "
                f"segmentation_head bị missing. Xem log để biết chi tiết."
            )

        loaded_count = len(mapped_sd) - len(result.unexpected_keys)
        logger.info(f"✓ Load thành công {loaded_count} keys. "
                    f"Chỉ segmentation_head ({len(result.missing_keys)} keys) bị missing (đúng kỳ vọng).")

    # --- Bước 3: Detect in_channels của head gốc ---
    old_head = model.segmentation_head
    if len(old_head) > 0 and isinstance(old_head[0], nn.Conv2d):
        head_in_channels = old_head[0].in_channels
    else:
        head_in_channels = 256
    logger.info(f"Decoder out → head in_channels = {head_in_channels}")

    # --- Bước 4: Replace head mới + Kaiming init ---
    new_head = smp.base.SegmentationHead(
        in_channels=head_in_channels,
        out_channels=num_classes,
        kernel_size=1,
        activation=None,
        upsampling=4,
    )
    new_head.apply(_kaiming_init)
    model.segmentation_head = new_head
    logger.info(f"Segmentation head replaced → out_channels={num_classes} (Kaiming init)")

    return model


def freeze_encoder(model: nn.Module):
    """Đóng băng encoder — dùng cho Phase 1."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    logger.info("Encoder FROZEN.")


def unfreeze_encoder(model: nn.Module):
    """Mở khoá encoder — dùng cho Phase 2."""
    for p in model.encoder.parameters():
        p.requires_grad = True
    logger.info("Encoder UNFROZEN.")


def log_trainable_params(model: nn.Module):
    """In tổng quan params trainable / total."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100 if total > 0 else 0
    logger.info(f"Trainable: {trainable:,} / Total: {total:,} ({pct:.2f}%)")
    return trainable
