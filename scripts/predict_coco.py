"""Run a YOLO model on a dataset split and save its detections as COCO results JSON.

Image ids follow the sorted file names, the same order read_yolo_dataset uses,
so the file matches the ground truth built by detection-eval's scripts/eval_coco.py.

Usage (needs the train dependency group, i.e. ultralytics):
    python scripts/predict_coco.py \
        --weights runs/visdrone_yolo11s_640_baseline/weights/best.pt --imgsz 640 \
        --out runs/coco_eval/yolo11s_640_val/detections.json
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from detection_eval.coco import IMAGE_SUFFIXES, detections_to_coco, save_json  # noqa: E402

parser = argparse.ArgumentParser(description="Save YOLO detections as COCO results JSON.")
parser.add_argument("--weights", required=True, help="path to a .pt file")
parser.add_argument("--images", default=str(ROOT / "data/VisDrone/images/val"), help="image folder")
parser.add_argument("--imgsz", type=int, default=640, help="input image size")
parser.add_argument("--conf", type=float, default=0.001, help="score threshold (low, as in mAP evaluation)")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (Ultralytics val default)")
parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
parser.add_argument("--out", required=True, help="output JSON path")
args = parser.parse_args()

paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
model = YOLO(args.weights)

results = []
for image_id, path in enumerate(paths, start=1):
    r = model.predict(
        str(path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=args.max_det, verbose=False,
    )[0]
    boxes = r.boxes
    results += detections_to_coco(
        image_id,
        boxes.xyxy.cpu().numpy(),  # pixels of the original image
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy(),
    )

save_json(results, args.out)
print(f"{len(paths)} images, {len(results)} detections -> {args.out}")
