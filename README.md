# DeepLabV3+ RetNet50 – Food Semantic Segmentation

Phân vùng ngữ nghĩa thực phẩm với DeepLabV3+ (ResNet50) trên dataset FoodSemSeg, kèm pipeline 3 tầng (seg → depth → weight) để ước lượng trọng lượng.

## Cấu trúc thư mục

```
├── config.py, dataset.py, model_setup.py, metrics.py, utils.py, postprocess.py
├── config/              # category_info mẫu (22 class, 29 class)
├── data/                # Dataset (không push nội dung)
├── checkpoints/         # Pretrained (không push)
├── runs/                # Kết quả train (không push)
├── reports/             # Báo cáo .docx (không push)
├── weights/             # best.pth — tải từ Releases
├── nv_pipeline/         # Pipeline 3 tầng
└── scripts/
    ├── training/        # train, finetune, eval_ensemble
    ├── api/             # FastAPI (22 class, 29 class)
    ├── data_tools/      # compute_class_stats, visualize, offline_resize
    ├── nv_pipeline/     # calibrate, test pipeline
    └── generate_report.py
```

## Chạy API
API sẽ tìm checkpoint theo thứ tự:

runs/train/exp21/weights/best.pth (nếu có)
weights/best.pth (nếu không có)
Quy trình cho người dùng:

Clone repo
Copy thư mục runs/ của vào project
pip install -r requirements.txt
uvicorn scripts.api.api:app --host 0.0.0.0 --port 8000
Không cần tải từ Releases nếu đã có runs/.
```bash
pip install -r requirements.txt
```

1. Tải thư mục `runs/` từ [Google Drive](https://drive.google.com/drive/folders/1kIm6Y9wQF1dEbHKUn86htZ2UsWq7GnQD?usp=sharing) → giải nén vào thư mục gốc project (có `runs/train/exp21/weights/best.pth`)
2. `uvicorn scripts.api.api:app --host 0.0.0.0 --port 8000`
3. `python scripts/api/test_api.py`


## Training

Tạo `data/FoodSemSeg_512x512/` với `category_info.json` (copy từ `config/category_info_22.json`), `train/images`, `train/masks`, `test/images`, `test/masks`.

```bash
python scripts/training/train.py
python scripts/training/finetune_from_checkpoint.py --checkpoint runs/train/exp21/weights/best.pth
python scripts/training/eval_ensemble.py --checkpoints runs/train/exp21/weights/best.pth --tta
```

## API khác

- **Legacy 29 class:** `uvicorn scripts.api.api_exp19_29class:app --port 8001`
- **Đổi checkpoint:** Sửa `API_BEST_PATH` trong `scripts/api/api.py`

## Pipeline 3 tầng

| Tầng | Input | Output |
|------|-------|--------|
| Tier 1 | Ảnh RGB | Semantic mask + instance |
| Tier 2 | Ảnh + mask | MiDaS depth → volume (cm³) |
| Tier 3 | Volume + class | Trọng lượng (g) |

## Data tools & Báo cáo

- `compute_class_stats.py`, `visualize_all_data.py`, `offline_resize.py`
- `python scripts/generate_report.py` → `reports/BaoCao_DeepLabV3_FoodSeg.docx`
