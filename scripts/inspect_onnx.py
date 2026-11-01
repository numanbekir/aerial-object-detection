"""Show what goes into and comes out of a YOLO ONNX model.

Prints the input and output names and shapes, runs the model once on one image
(letterboxed to the input size) and shows the strongest raw candidates.

Usage:
    python scripts/inspect_onnx.py --model runs/visdrone_yolo11s_640_baseline/weights/best.onnx \
        --image data/VisDrone/images/val/0000295_02400_d_0000033.jpg
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aerial_object_detection.decode import letterbox  # noqa: E402

parser = argparse.ArgumentParser(description="Inspect a YOLO ONNX model.")
parser.add_argument("--model", required=True, help="path to a .onnx file")
parser.add_argument("--image", required=True, help="one image to run")
parser.add_argument("--top", type=int, default=5, help="how many candidates to print")
args = parser.parse_args()

session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
print("inputs:")
for i in session.get_inputs():
    print(f"  {i.name}  shape {i.shape}  type {i.type}")
print("outputs:")
for o in session.get_outputs():
    print(f"  {o.name}  shape {o.shape}  type {o.type}")

size = session.get_inputs()[0].shape[2]
img = Image.open(args.image)
x, (scale, pad_x, pad_y) = letterbox(img, size)
new_w, new_h = round(img.width * scale), round(img.height * scale)
print(f"\nimage {img.width} x {img.height} -> {new_w} x {new_h}, padded to {size} x {size}")
print(f"input array shape {x.shape}, values {x.min():.2f} to {x.max():.2f}")

start = time.perf_counter()
out = session.run(None, {session.get_inputs()[0].name: x})[0]
ms = (time.perf_counter() - start) * 1000
print(f"\noutput array shape {out.shape}  ({ms:.0f} ms on CPU)")

pred = out[0]                 # (4 + classes, candidates)
boxes = pred[:4]              # cx, cy, w, h in input pixels
scores = pred[4:]             # one score per class
best_class = scores.argmax(axis=0)
best_score = scores.max(axis=0)
print(f"candidates: {pred.shape[1]}, with best score >= 0.25: {(best_score >= 0.25).sum()}")

print(f"\nstrongest {args.top} candidates (column, cx, cy, w, h, class, score):")
for j in np.argsort(-best_score)[: args.top]:
    cx, cy, w, h = boxes[:, j]
    print(f"  {j:>5}  {cx:7.1f} {cy:7.1f} {w:6.1f} {h:6.1f}  class {best_class[j]}  {best_score[j]:.3f}")
