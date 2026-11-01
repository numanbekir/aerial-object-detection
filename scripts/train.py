"""Train a YOLO11 model on VisDrone2019-DET.

Usage (from any directory):
    python scripts/train.py                                       # 640 baseline
    python scripts/train.py --imgsz 1280 --batch 4 --name visdrone_yolo11s_1280
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # repo root

parser = argparse.ArgumentParser(description="Train YOLO11 on VisDrone2019-DET.")
parser.add_argument("--imgsz", type=int, default=640, help="input image size")
parser.add_argument(
    "--batch", type=int, default=16,
    help="images per batch (lower it if GPU memory runs out; nbs=64 keeps the update size fixed)",
)
parser.add_argument("--epochs", type=int, default=50, help="passes over the training set")
parser.add_argument(
    "--name", default="visdrone_yolo11s_640_baseline", help="run folder under runs/"
)
args = parser.parse_args()

# Start from COCO-pretrained weights (transfer learning)
model = YOLO(ROOT / "data/weights/yolo11s.pt")

model.train(
    data=str(ROOT / "data/VisDrone.yaml"),
    imgsz=args.imgsz,
    epochs=args.epochs,
    batch=args.batch,
    workers=4,
    seed=0,  # Ultralytics default, written out so runs stay comparable
    project=str(ROOT / "runs"),
    name=args.name,
)
