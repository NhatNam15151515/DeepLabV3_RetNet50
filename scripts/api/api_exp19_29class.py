import os
import sys

# Đảm bảo project root nằm trên sys.path để import config, model_setup, postprocess
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

"""
api_exp19_29class.py — FastAPI server dùng checkpoint exp19 (29 class cũ).

Chạy (sau khi gom script):
    uvicorn scripts.api.api_exp19_29class:app --host 0.0.0.0 --port 8000

Ảnh đầu vào: bất kỳ kích thước nào. Không ép resize hay bóp méo.
"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config as CFG
from model_setup import create_model
from postprocess import count_instances
from nv_pipeline.tier2_depth_volume import Tier2DepthVolume
from nv_pipeline.tier3_weight_estimation import Tier3WeightEstimation


LEGACY_CATEGORY_INFO_PATH = os.path.join(
    CFG.PROJECT_ROOT, "data", "FoodSemSeg", "category_info.json"
)
with open(LEGACY_CATEGORY_INFO_PATH, "r", encoding="utf-8", errors="replace") as f:
    _legacy_info = json.load(f)
_legacy_categories = sorted(_legacy_info["categories"], key=lambda x: x["id"])
LEGACY_NUM_CLASSES = _legacy_info["num_classes"]
LEGACY_CLASS_NAMES = ["background"] + [c["name"] for c in _legacy_categories]

API19_BEST_PATH = os.path.join(
    CFG.RUNS_DIR, "exp19", "weights", "best.pth"
)

FOOD_DENSITIES = {
    "milk": 1.03, "banana": 0.95, "steak": 1.08, "pork": 1.05,
    "chicken duck": 1.05, "fish": 1.02, "shrimp": 0.85, "bread": 0.35,
    "noodles": 0.90, "rice": 1.15, "tofu": 0.95, "potato": 1.09,
    "tomato": 0.99, "lettuce": 0.35, "cucumber": 0.96, "carrot": 1.04,
    "broccoli": 0.60, "cabbage": 0.45, "onion": 1.05, "pepper": 0.55,
    "other": 0.90, "soup": 1.01, "rice+main": 1.05, "fruit": 0.85,
    "drink": 1.00, "dessert": 0.80, "utensil": 0.0, "background-other": 0.0,
    "egg": 1.03,
}

# Thư mục lưu kết quả API exp19 (29 class)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
API19_SAVE_DIR = os.path.join(SCRIPTS_DIR, "api_results_exp19")
os.makedirs(API19_SAVE_DIR, exist_ok=True)

# Kích thước tối đa cạnh dài (model train 512)
MAX_INPUT_SIDE = 512

# Normalize ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


CLASS_COLORS = [
    (0,   0,   0),      # 0  background
    (255, 255, 255),    # 1  milk
    (255, 215, 0),      # 2  banana
    (205, 92,  92),     # 3  steak
    (210, 105, 30),     # 4  pork
    (255, 165, 0),      # 5  chicken duck
    (70,  130, 180),    # 6  fish
    (255, 140, 0),      # 7  shrimp
    (222, 184, 135),    # 8  bread
    (255, 228, 196),    # 9  noodles
    (238, 232, 170),    # 10 rice
    (245, 245, 220),    # 11 tofu
    (210, 180, 140),    # 12 potato
    (255, 99,  71),     # 13 tomato
    (144, 238, 144),    # 14 lettuce
    (50,  205, 50),     # 15 cucumber
    (255, 165, 79),     # 16 carrot
    (34,  139, 34),     # 17 broccoli
    (154, 205, 50),     # 18 cabbage
    (255, 255, 0),      # 19 onion
    (255, 69,  0),      # 20 pepper
    (255, 182, 193),    # 21 other
    (135, 206, 235),    # 22 soup
    (176, 196, 222),    # 23 rice+main
    (199, 21,  133),    # 24 fruit
    (0,   206, 209),    # 25 drink
    (219, 112, 147),    # 26 dessert
    (139, 69,  19),     # 27 utensil
    (112, 128, 144),    # 28 background-other
]


def draw_labels_on_regions(
    img: np.ndarray,
    preds: np.ndarray,
    present_classes: list,
    class_names: list,
    weight_map: dict = None,
) -> np.ndarray:
    """Vẽ tên class + trọng lượng trực tiếp lên centroid của mỗi vùng mask."""
    out = img.copy()
    ih, iw = out.shape[:2]
    base_dim = max(ih, iw)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, base_dim / 900.0)
    thickness = max(1, int(base_dim / 600.0))

    for cid in present_classes:
        if cid == 0:
            continue
        mask = (preds == cid).astype(np.uint8)
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask > 0)
        cx, cy = int(xs.mean()), int(ys.mean())

        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        if weight_map and cid in weight_map:
            w_g = weight_map[cid]
            label = f"{name} {w_g:.0f}g" if w_g >= 1 else f"{name} <1g"
        else:
            label = name

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = max(0, min(cx - tw // 2, iw - tw))
        ty = max(th, min(cy + th // 2, ih))

        cv2.putText(out, label, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def preprocess_any_size(image: np.ndarray, max_side: int = MAX_INPUT_SIDE):
    """
    Tiền xử lý ảnh bất kỳ kích thước, không bóp méo.
    - Scale để cạnh dài = max_side (giữ tỷ lệ)
    - Pad về max_side x max_side (pad bottom-right, value=0)
    Trả về: (tensor CxHxW, content_h, content_w) để crop mask sau inference.
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, new_h, new_w


def preds_to_original_size(preds: np.ndarray, content_h: int, content_w: int, orig_h: int, orig_w: int):
    """Crop vùng nội dung (bỏ padding), resize về kích thước ảnh gốc."""
    cropped = preds[:content_h, :content_w].copy()
    out = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return out


class LegacyModelSingleton:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(LEGACY_NUM_CLASSES, use_imagenet=True)
        state = torch.load(API19_BEST_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        print(f"[API-EXP19] Seg model ready on {self.device} | classes={LEGACY_NUM_CLASSES}")

        print("[API-EXP19] Loading Tier2 (MiDaS) + Tier3 (Weight)...")
        self.tier2 = Tier2DepthVolume()
        self.tier3 = Tier3WeightEstimation()
        for name, density in FOOD_DENSITIES.items():
            self.tier3.density_db.add_density(name, density)
        print("[API-EXP19] Tier2 + Tier3 ready")

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = LegacyModelSingleton()
        return cls._instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    LegacyModelSingleton.get()
    yield


app = FastAPI(title="FoodSegmentation API (exp19, 29-class legacy)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh.")

    ih, iw = image.shape[:2]

    ms = LegacyModelSingleton.get()
    tensor, content_h, content_w = preprocess_any_size(image, MAX_INPUT_SIDE)
    tensor = tensor.to(ms.device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(ms.device.type == "cuda")):
        logits = ms.model(tensor)
        preds = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    # Map mask về đúng kích thước ảnh gốc (crop padding rồi resize)
    preds_orig = preds_to_original_size(preds, content_h, content_w, ih, iw)

    # Color mask theo kích thước gốc
    color_mask = np.zeros((ih, iw, 3), dtype=np.uint8)
    for cid, color in enumerate(CLASS_COLORS):
        if cid >= LEGACY_NUM_CLASSES:
            continue
        color_mask[preds_orig == cid] = color

    overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    inst_dict = count_instances(preds_orig, LEGACY_NUM_CLASSES)

    # Tier 2-3: ước lượng trọng lượng
    weight_map = {}
    present_classes = [cid for cid in inst_dict if cid > 0]
    if present_classes:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_map = ms.tier2.estimate_depth(img_rgb)
        for cid in present_classes:
            cls_mask = (preds_orig == cid).astype(np.uint8)
            cls_name = LEGACY_CLASS_NAMES[cid] if cid < len(LEGACY_CLASS_NAMES) else f"class_{cid}"
            dv_results = ms.tier2.estimate_volume(img_rgb, [(cls_name, cls_mask)], depth_map)
            if dv_results:
                we = ms.tier3.estimate_weight(cls_name, dv_results[0].volume_cm3)
                weight_map[cid] = round(we.weight_grams, 1)

    if present_classes:
        color_mask = draw_labels_on_regions(color_mask, preds_orig, present_classes, LEGACY_CLASS_NAMES, weight_map)
        overlay = draw_labels_on_regions(overlay, preds_orig, present_classes, LEGACY_CLASS_NAMES, weight_map)

    detections = []
    total_weight = 0.0
    for cid, cnt in sorted(inst_dict.items()):
        if cid >= len(LEGACY_CLASS_NAMES):
            name = f"class_{cid}"
        else:
            name = LEGACY_CLASS_NAMES[cid]
        if cid < len(CLASS_COLORS):
            r, g, b = CLASS_COLORS[cid]
        else:
            r, g, b = (255, 255, 255)
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        w_g = weight_map.get(cid, 0.0)
        total_weight += w_g
        detections.append(
            {
                "class_id": cid,
                "class_name": name,
                "instance_count": int(cnt),
                "estimated_weight_g": w_g,
                "color_rgb": [int(r), int(g), int(b)],
                "color_hex": color_hex,
            }
        )

    response = {
        "image_info": {
            "width": int(iw),
            "height": int(ih),
        },
        "total_estimated_weight_g": round(total_weight, 1),
        "detections": detections,
    }

    try:
        filename = file.filename or "upload"
        stem, _ = os.path.splitext(os.path.basename(filename))
        safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
        out_base = os.path.join(API19_SAVE_DIR, safe_stem)
        cv2.imwrite(out_base + "_orig.png", image)
        cv2.imwrite(out_base + "_mask.png", color_mask)
        cv2.imwrite(out_base + "_overlay.png", overlay)
        with open(out_base + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[API-EXP19] Warning: failed to save api_results_exp19:", repr(e))

    return response


@app.get("/health")
async def health():
    return {"status": "ok", "num_classes": LEGACY_NUM_CLASSES}
