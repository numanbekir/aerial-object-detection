"""Sliced inference with a YOLO ONNX model: the whole image plus overlapping tiles.

Every image is run once at the model's input size and once per tile at full
resolution; the tile boxes are shifted back to image pixels and everything is
merged with class-aware NMS (aerial_object_detection.slicing). Saved as COCO
results JSON like scripts/predict_onnx.py, so detection-eval's eval_coco.py
scores both the same way.

Usage:
    python scripts/predict_sliced.py --model runs/visdrone_yolo11s_640_baseline/weights/best.onnx \
        --out runs/coco_eval/yolo11s_640_sliced_val/detections.json

    # several merge methods from one model pass: {merge} in --out is replaced by each name
    python scripts/predict_sliced.py --model ... --merge nms edge ios \
        --out runs/coco_eval/yolo11s_640_sliced_{merge}_val/detections.json
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aerial_object_detection.onnx_detector import OnnxDetector  # noqa: E402
from aerial_object_detection.slicing import MERGES, merge_ios, sliced_candidates, tile_origins  # noqa: E402
from detection_eval.coco import IMAGE_SUFFIXES, detections_to_coco, save_json  # noqa: E402

parser = argparse.ArgumentParser(description="Sliced ONNX Runtime detections as COCO results JSON.")
parser.add_argument("--model", required=True, help="path to a .onnx file")
parser.add_argument("--images", default=str(ROOT / "data/VisDrone/images/val"), help="image folder")
parser.add_argument("--tile", type=int, help="tile size in pixels (default: the model's input size)")
parser.add_argument("--overlap", type=int, default=128, help="pixels shared by neighbouring tiles")
parser.add_argument("--no-full-image", action="store_true", help="run the tiles only")
parser.add_argument("--merge", nargs="+", default=["nms"], choices=sorted(MERGES),
                    help="how tile boxes are merged: nms, edge (drop boxes at inner tile borders) or ios (SAHI)")
parser.add_argument("--conf", type=float, default=0.001, help="score threshold (low, as in mAP evaluation)")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold, inside a tile and for merging")
parser.add_argument("--max-det", type=int, default=300, help="max detections per tile and per image")
parser.add_argument("--limit", type=int, help="only the first N images (a quick check; not for scoring)")
parser.add_argument("--out", required=True, help="output JSON path; with several --merge values it must contain {merge}")
args = parser.parse_args()
if len(args.merge) > 1 and "{merge}" not in args.out:
    parser.error("--out must contain {merge} when several merge methods are given")

detect = OnnxDetector(args.model, args.conf, args.iou, args.max_det)
tile = args.tile or detect.size
paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
if args.limit:
    paths = paths[: args.limit]

results = {m: [] for m in args.merge}
merge_seconds = {m: 0.0 for m in args.merge}
model_runs = 0
start = time.perf_counter()
for image_id, path in enumerate(paths, start=1):  # same ids as read_yolo_dataset
    with Image.open(path) as img:
        img = img.convert("RGB")
        candidates = sliced_candidates(img, detect, tile, args.overlap, not args.no_full_image)
        model_runs += len(tile_origins(img.width, img.height, tile, args.overlap)) + (not args.no_full_image)
    for m in args.merge:
        t = time.perf_counter()
        if m == "ios":
            xyxy, scores, classes = merge_ios(*candidates, max_det=args.max_det)
        else:
            xyxy, scores, classes = MERGES[m](*candidates, iou=args.iou, max_det=args.max_det)
        merge_seconds[m] += time.perf_counter() - t
        results[m].extend(detections_to_coco(image_id, xyxy, scores, classes))

total = time.perf_counter() - start
print(f"{len(paths)} images, tile {tile}, overlap {args.overlap}, "
      f"full image {'no' if args.no_full_image else 'yes'}: {model_runs / len(paths):.1f} model runs per image")
print(f"per image (CPU): model {detect.model_seconds / len(paths) * 1000:.0f} ms, everything {total / len(paths) * 1000:.0f} ms")
for m in args.merge:
    out = args.out.replace("{merge}", m)
    save_json(results[m], out)
    print(f"  {m:5s} merge {merge_seconds[m] / len(paths) * 1000:5.1f} ms per image, "
          f"{len(results[m])} detections -> {out}")
