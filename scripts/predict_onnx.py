"""Run a YOLO ONNX model on a folder of images without Ultralytics.

letterbox -> ONNX Runtime -> select_candidates -> batched_nms -> boxes_to_original,
saved as COCO results JSON in the same format as scripts/predict_coco.py, so
detection-eval's scripts/eval_coco.py can score both.

Usage:
    python scripts/predict_onnx.py --model runs/visdrone_yolo11s_640_baseline/weights/best.onnx \
        --out runs/coco_eval/yolo11s_640_onnx_val/detections.json
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aerial_object_detection.onnx_detector import OnnxDetector  # noqa: E402
from detection_eval.coco import IMAGE_SUFFIXES, detections_to_coco, save_json  # noqa: E402

parser = argparse.ArgumentParser(description="ONNX Runtime detections as COCO results JSON.")
parser.add_argument("--model", required=True, help="path to a .onnx file")
parser.add_argument("--images", default=str(ROOT / "data/VisDrone/images/val"), help="image folder")
parser.add_argument("--conf", type=float, default=0.001, help="score threshold (low, as in mAP evaluation)")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
parser.add_argument("--out", required=True, help="output JSON path")
args = parser.parse_args()

detect = OnnxDetector(args.model, args.conf, args.iou, args.max_det)
paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

results = []
start = time.perf_counter()
for image_id, path in enumerate(paths, start=1):  # same ids as read_yolo_dataset
    with Image.open(path) as img:
        xyxy, scores, classes = detect(img)
    results.extend(detections_to_coco(image_id, xyxy, scores, classes))

total = time.perf_counter() - start
save_json(results, args.out)
print(f"{len(paths)} images, {len(results)} detections -> {args.out}")
print(f"per image (CPU): model {detect.model_seconds / len(paths) * 1000:.0f} ms, everything {total / len(paths) * 1000:.0f} ms")
