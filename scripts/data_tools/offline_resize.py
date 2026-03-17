"""
offline_resize.py — Resize trước ảnh + mask semantic về kích thước cố định.

Tạo folder FoodSemSeg_resized chứa data đã resize sẵn,
giúp training nhanh hơn (không resize on-the-fly mỗi batch).

Cách dùng (sau khi gom script):
    python scripts/data_tools/offline_resize.py --size 512
"""

import os
import cv2
import shutil
import argparse
import numpy as np
from tqdm import tqdm


def resize_semantic_pair(img_path, mask_path, out_img_path, out_mask_path,
                         target_h=512, target_w=512):
    """
    Resize ảnh và mask giữ nguyên tỷ lệ (aspect ratio).
    - Scale theo cạnh dài nhất cho vừa target.
    - Pad phần dư bằng background (ảnh: đen, mask: 0).
    - INTER_AREA cho ảnh, INTER_NEAREST cho mask.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] Không đọc được ảnh: {img_path}")
        return False

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    pad_y1 = (target_h - new_h) // 2
    pad_y2 = target_h - new_h - pad_y1
    pad_x1 = (target_w - new_w) // 2
    pad_x2 = target_w - new_w - pad_x1

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_img = cv2.copyMakeBorder(
        img_resized, pad_y1, pad_y2, pad_x1, pad_x2,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    cv2.imwrite(out_img_path, padded_img)

    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"  [SKIP] Không đọc được mask: {mask_path}")
            return False

        if mask.ndim > 2:
            mask = mask[:, :, 0]

        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded_mask = cv2.copyMakeBorder(
            mask_resized, pad_y1, pad_y2, pad_x1, pad_x2,
            cv2.BORDER_CONSTANT, value=0
        )
        cv2.imwrite(out_mask_path, padded_mask)

    return True


def resize_split(in_img_dir, in_mask_dir, out_img_dir, out_mask_dir,
                 target_h, target_w, split_name):
    """Resize toàn bộ 1 split (train hoặc test)."""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    if not os.path.exists(in_img_dir):
        print(f"  [SKIP] Không tìm thấy: {in_img_dir}")
        return 0, 0

    images = sorted(
        f for f in os.listdir(in_img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    success = 0
    fail = 0

    for img_name in tqdm(images, desc=f"Resize {split_name}"):
        img_path = os.path.join(in_img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(in_mask_dir, mask_name)

        out_img = os.path.join(out_img_dir, img_name)
        out_mask = os.path.join(out_mask_dir, mask_name)

        ok = resize_semantic_pair(img_path, mask_path, out_img, out_mask,
                                  target_h, target_w)
        if ok:
            success += 1
        else:
            fail += 1

    return success, fail


def main():
    parser = argparse.ArgumentParser(description="Offline resize FoodSemSeg data")
    parser.add_argument("--size", type=int, default=512,
                        help="Target size (cả width và height, default=512)")
    parser.add_argument("--width", type=int, default=None,
                        help="Target width (nếu muốn khác height)")
    parser.add_argument("--height", type=int, default=None,
                        help="Target height (nếu muốn khác width)")
    args = parser.parse_args()

    target_h = args.height or args.size
    target_w = args.width or args.size

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "data", "FoodSemSeg")
    out_dir = os.path.join(project_root, "data", f"FoodSemSeg_{target_h}x{target_w}")

    print("=" * 60)
    print(f"OFFLINE RESIZE — FoodSemSeg")
    print(f"  Input : {base_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Size  : {target_w}x{target_h}")
    print("=" * 60)

    total_ok = 0
    total_fail = 0

    for split in ["train", "test"]:
        in_img = os.path.join(base_dir, split, "images")
        in_mask = os.path.join(base_dir, split, "masks")
        out_img = os.path.join(out_dir, split, "images")
        out_mask = os.path.join(out_dir, split, "masks")

        ok, fail = resize_split(in_img, in_mask, out_img, out_mask,
                                target_h, target_w, split)
        total_ok += ok
        total_fail += fail
        print(f"  {split}: {ok} OK, {fail} FAIL")

    src_json = os.path.join(base_dir, "category_info.json")
    dst_json = os.path.join(out_dir, "category_info.json")
    if os.path.exists(src_json):
        shutil.copy2(src_json, dst_json)
        print(f"\nCopied category_info.json → {dst_json}")

    print(f"\n✓ DONE — {total_ok} ảnh resize thành công, {total_fail} lỗi.")
    print(f"  Output: {out_dir}")
    print(f"\nĐể dùng data đã resize, bật USE_RESIZED_DATA = True trong config.py")


if __name__ == "__main__":
    main()

