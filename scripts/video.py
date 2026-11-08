"""Run a YOLO model on a video: boxes on every frame, time per stage, FPS.

Stages, timed per frame (medians printed at the end):
- decode: read the next frame from the file (OpenCV + FFmpeg, on the CPU)
- detect: letterbox + model + candidates + NMS + boxes back (runtimes.detect)
- draw: boxes, the class counts and the FPS line on the frame
- write: resize and encode the frame into the output file (only with --out)
- show: resize and show the frame in a window (only with --show)

FPS = 1000 / (decode + detect + draw), on screen as an average of the last 30
frames. Writing and showing are left out: the file exists only for the
recording, and a window costs differently on every machine.

Usage (Ultralytics venv, from the repo root):
    python scripts/video.py --video data/videos/bridge_4k60.mp4 \
        --model runs/trt_640_fp16/best.engine --out runs/video/bridge_640.mp4
"""
import argparse
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aerial_object_detection.runtimes import PREPARES, RUNTIMES, detect, load_runtime, make_prepare  # noqa: E402

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# one BGR colour per class
COLORS = [(60, 60, 255), (0, 165, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
          (255, 128, 0), (0, 255, 255), (128, 0, 255), (255, 255, 255), (200, 0, 120)]
RUNTIME_LABELS = {"onnx-cpu": "ONNX Runtime CPU", "torch-cpu": "PyTorch CPU", "torch-gpu": "PyTorch GPU",
                  "torch-gpu-fp16": "PyTorch GPU FP16", "trt": "TensorRT"}


def draw(frame, xyxy, classes, header):
    """Boxes coloured by class, the header line at the top, class counts at the bottom."""
    height, width = frame.shape[:2]
    thick = max(1, width // 960)
    for (x1, y1, x2, y2), c in zip(xyxy.astype(int), classes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[c], thick)

    counts = np.bincount(classes, minlength=len(CLASS_NAMES))
    footer = "   ".join(f"{CLASS_NAMES[c]} {n}" for c, n in enumerate(counts) if n)
    scale = width / 1600
    for text, y in ((header, int(50 * scale)), (footer, height - int(25 * scale))):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 * thick)
        cv2.rectangle(frame, (0, y - th - int(15 * scale)), (tw + int(30 * scale), y + int(15 * scale)), (0, 0, 0), -1)
        cv2.putText(frame, text, (int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2 * thick, cv2.LINE_AA)


parser = argparse.ArgumentParser(description="YOLO on a video, with time per stage and FPS.")
parser.add_argument("--video", required=True, help="input video file")
parser.add_argument("--model", required=True, help=".engine for trt (default), .pt or .onnx for the others")
parser.add_argument("--runtime", default="trt", choices=RUNTIMES)
parser.add_argument("--preprocess", default="gpu-aa", choices=PREPARES, help="where the letterbox runs")
parser.add_argument("--imgsz", type=int, default=640, help="input size for torch-*")
parser.add_argument("--conf", type=float, default=0.25, help="score threshold")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0: all)")
parser.add_argument("--warmup", type=int, default=10, help="frames left out of the summary")
parser.add_argument("--out", help="write the annotated video here (.mp4)")
parser.add_argument("--out-width", type=int, default=1920, help="width of the written video")
parser.add_argument("--show", action="store_true", help="show the frames in a window (q to stop)")
parser.add_argument("--label", help="first part of the line on the frame (default: runtime and input size)")
args = parser.parse_args()

run, size = load_runtime(args.runtime, args.model, args.imgsz)
prepare = make_prepare(args.preprocess, size, bgr=True)  # OpenCV gives BGR frames

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    sys.exit(f"cannot open {args.video}")
video_fps = cap.get(cv2.CAP_PROP_FPS)
src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if args.out:
    out_size = (args.out_width, round(src_h * args.out_width / src_w))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, out_size)

precision = " FP16" if args.runtime == "trt" and "fp16" in str(args.model).lower() else ""
name = args.label or f"{RUNTIME_LABELS[args.runtime]}{precision}, input {size}"
times = {"decode": [], "detect": [], "draw": [], "write": [], "show": []}
recent = deque(maxlen=30)  # decode + detect + draw of the last frames, for the on-screen FPS
boxes_per_frame = []
frame_index = 0

while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        break
    t1 = time.perf_counter()
    xyxy, scores, classes, _ = detect(run, prepare, frame, args.conf, args.iou)
    t2 = time.perf_counter()
    live_ms = 1000 * (t2 - t0) if not recent else statistics.fmean(recent)
    header = (f"{name} | {1000 / live_ms:.0f} FPS | decode {1000 * (t1 - t0):.1f} ms"
              f"  detect {1000 * (t2 - t1):.1f} ms | {len(xyxy)} objects")
    draw(frame, xyxy, classes, header)
    t3 = time.perf_counter()
    recent.append(1000 * (t3 - t0))

    if writer is not None:
        writer.write(cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA))
    t4 = time.perf_counter()
    if args.show:
        cv2.imshow(name, cv2.resize(frame, (1280, round(src_h * 1280 / src_w)), interpolation=cv2.INTER_AREA))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    t5 = time.perf_counter()

    if frame_index >= args.warmup:
        for key, seconds in (("decode", t1 - t0), ("detect", t2 - t1), ("draw", t3 - t2),
                             ("write", t4 - t3), ("show", t5 - t4)):
            times[key].append(1000 * seconds)
        boxes_per_frame.append(len(xyxy))
    frame_index += 1
    if args.max_frames and frame_index >= args.max_frames:
        break

cap.release()
if writer is not None:
    writer.release()
if args.show:
    cv2.destroyAllWindows()

n = len(times["decode"])
if n == 0:
    sys.exit("not enough frames after the warm-up")
print(f"{name}, {args.preprocess} letterbox, video {src_w}x{src_h} at {video_fps:.0f} FPS, "
      f"{n} frames after {args.warmup} warm-up")
used = ["decode", "detect", "draw"] + ["write"] * bool(args.out) + ["show"] * args.show
for key in used:
    print(f"  {key:7s} median {statistics.median(times[key]):6.1f} ms")
live = statistics.median([a + b + c for a, b, c in zip(times["decode"], times["detect"], times["draw"])])
print(f"  decode + detect + draw  {live:.1f} ms -> {1000 / live:.0f} FPS "
      f"({'keeps up with' if 1000 / live >= video_fps else 'slower than'} the video's {video_fps:.0f} FPS)")
print(f"  objects per frame: {np.mean(boxes_per_frame):.1f}")
if args.out:
    print(f"  written: {args.out} ({out_size[0]}x{out_size[1]}, mp4v)")
