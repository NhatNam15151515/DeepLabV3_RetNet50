"""Test script cho FastAPI /predict endpoint (22-class API mặc định)."""

import sys
import json
import requests

URL = "http://localhost:8000/predict"
"""
http://localhost:8000/docs để test trên trình duyệt
"""
IMG = r"c:\Nhat Nam\do an chuyen nganh\DeepLabV3_RetNet50\data\FoodSemSeg_512x512\test\images\00000048.jpg"

print(f"Sending: {IMG}")
print(f"To: {URL}")
print()

try:
    r = requests.post(URL, files={"file": ("test.jpg", open(IMG, "rb"), "image/jpeg")}, timeout=60)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print(f"HTTP Status: {r.status_code}")

if r.status_code != 200:
    print(f"Error: {r.text[:500]}")
    sys.exit(1)

d = r.json()

print(f"Response keys: {list(d.keys())}")
print()
print(f"image_info: {d.get('image_info')}")
print()
print("detections:")
for det in d.get("detections", []):
    print(f"  [{det['class_id']:>2}] {det['class_name']:<20} x{det['instance_count']}  {det['color_hex']}")
print()

oi = d.get("original_image", {})
mi = d.get("mask_image", {})
ov = d.get("overlay_image", {})
print(f"original_image: format={oi.get('format')}, base64_len={len(oi.get('data', ''))}")
print(f"mask_image:     format={mi.get('format')}, base64_len={len(mi.get('data', ''))}")
print(f"overlay_image:  format={ov.get('format')}, base64_len={len(ov.get('data', ''))}")
print()
print("ALL TESTS PASSED!")

