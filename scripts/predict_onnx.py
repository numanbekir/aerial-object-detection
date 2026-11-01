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

import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from detection_eval.boxes import cxcywh_to_xyxy  # noqa: E402
from detection_eval.coco import IMAGE_SUFFIXES, detections_to_coco, save_json  # noqa: E402
from aerial_object_detection.decode import boxes_to_original, letterbox, select_candidates  # noqa: E402
from detection_eval.nms import batched_nms  # noqa: E402

parser = argparse.ArgumentParser(description="ONNX Runtime detections as COCO results JSON.")
parser.add_argument("--model", required=True, help="path to a .onnx file")
parser.add_argument("--images", default=str(ROOT / "data/VisDrone/images/val"), help="image folder")
parser.add_argument("--conf", type=float, default=0.001, help="score threshold (low, as in mAP evaluation)")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
parser.add_argument("--out", required=True, help="output JSON path")
args = parser.parse_args()

session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]

paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
results = []
model_ms = 0.0
start = time.perf_counter()
for image_id, path in enumerate(paths, start=1):  # same ids as read_yolo_dataset
    with Image.open(path) as img:
        x, (scale, pad_x, pad_y) = letterbox(img, size)
        img_w, img_h = img.size

    t = time.perf_counter()
    pred = session.run(None, {input_name: x})[0][0]
    model_ms += (time.perf_counter() - t) * 1000

    boxes, scores, classes = select_candidates(pred, args.conf)
    keep = batched_nms(cxcywh_to_xyxy(boxes), scores, classes, args.iou, args.max_det)
    xyxy = boxes_to_original(boxes[keep], scale, pad_x, pad_y, img_w, img_h)
    results.extend(detections_to_coco(image_id, xyxy, scores[keep], classes[keep]))

total = time.perf_counter() - start
save_json(results, args.out)
print(f"{len(paths)} images, {len(results)} detections -> {args.out}")
print(f"model {model_ms / len(paths):.0f} ms per image (CPU), everything {total / len(paths) * 1000:.0f} ms per image")
