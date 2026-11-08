"""Time one YOLO model on VisDrone val images with different runtimes.

The pipeline is the same for every runtime, only the model call changes
(runtimes.py): letterbox -> model -> select_candidates -> batched_nms -> boxes_to_original.

Two times are reported per image, as medians:
- model: the model call alone (for the GPU, until the result is back in memory)
- everything: from a decoded PIL image to the final boxes

Each image is read and decoded before its clock starts, so disk reads are not
timed. The first --warmup images are run but not timed.

--runtime: onnx-cpu, torch-cpu, torch-gpu, torch-gpu-fp16, trt (see runtimes.py).
--preprocess: where the letterbox runs; gpu and gpu-aa need a GPU runtime.

With --out, the boxes of every image (warm-up included) are also saved as COCO
results JSON, so detection-eval's scripts/eval_coco.py can score the runtime.
For mAP, use all val images and a low threshold:
    --count 548 --warmup 0 --conf 0.001 --out runs/coco_eval/<name>/detections.json

Usage (Ultralytics venv, from the repo root):
    python scripts/benchmark.py --runtime onnx-cpu --model runs/visdrone_yolo11s_640_baseline/weights/best.onnx
    python scripts/benchmark.py --runtime trt --model runs/trt_640_fp16/best.engine --preprocess gpu-aa
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aerial_object_detection.runtimes import PREPARES, RUNTIMES, detect, load_runtime, make_prepare  # noqa: E402
from detection_eval.coco import IMAGE_SUFFIXES, detections_to_coco, save_json  # noqa: E402

parser = argparse.ArgumentParser(description="Time a YOLO model with different runtimes.")
parser.add_argument("--runtime", required=True, choices=RUNTIMES)
parser.add_argument("--model", required=True, help=".onnx for onnx-cpu, .pt for torch-*, .engine for trt")
parser.add_argument("--imgsz", type=int, default=640, help="input size for torch-* (ONNX and TensorRT read it from the file)")
parser.add_argument("--images", default=str(ROOT / "data/VisDrone/images/val"), help="image folder")
parser.add_argument("--count", type=int, default=100, help="timed images")
parser.add_argument("--warmup", type=int, default=10, help="untimed images before timing")
parser.add_argument("--conf", type=float, default=0.25, help="score threshold (as in live use, not mAP)")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
parser.add_argument("--preprocess", default="cpu", choices=PREPARES, help="where the letterbox runs")
parser.add_argument("--out", help="also save the boxes as COCO results JSON")
args = parser.parse_args()
if args.preprocess != "cpu" and args.runtime in ("onnx-cpu", "torch-cpu"):
    parser.error("--preprocess gpu / gpu-aa needs a GPU runtime")

run, size = load_runtime(args.runtime, args.model, args.imgsz)
prepare = make_prepare(args.preprocess, size)

paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
paths = paths[: args.warmup + args.count]

model_ms, total_ms, box_counts, results = [], [], [], []
for i, path in enumerate(paths):
    with Image.open(path) as img:
        image = img.convert("RGB")
    start = time.perf_counter()
    xyxy, scores, classes, model_seconds = detect(run, prepare, image, args.conf, args.iou, args.max_det)
    total = time.perf_counter() - start
    if args.out:
        # same ids as read_yolo_dataset: position in the sorted folder, from 1
        results.extend(detections_to_coco(i + 1, xyxy, scores, classes))
    if i < args.warmup:
        continue
    model_ms.append(model_seconds * 1000)
    total_ms.append(total * 1000)
    box_counts.append(len(xyxy))

model_med = statistics.median(model_ms)
total_med = statistics.median(total_ms)
print(f"{args.runtime}, {args.model}, letterbox {args.preprocess}, input {size}, {len(model_ms)} images after {args.warmup} warm-up")
print(f"model       median {model_med:6.1f} ms  ({1000 / model_med:5.1f} FPS)   min {min(model_ms):.1f}  max {max(model_ms):.1f}")
print(f"everything  median {total_med:6.1f} ms  ({1000 / total_med:5.1f} FPS)   min {min(total_ms):.1f}  max {max(total_ms):.1f}")
print(f"boxes per image (conf {args.conf}): {np.mean(box_counts):.1f}")
if args.out:
    save_json(results, args.out)
    print(f"{len(paths)} images, {len(results)} detections -> {args.out}")
