"""Time one YOLO model on VisDrone val images with different runtimes.

The pipeline is the same for every runtime, only the model call changes:
letterbox -> model -> select_candidates -> batched_nms -> boxes_to_original.

Two times are reported per image, as medians:
- model: the model call alone (for the GPU, until the result is back in memory)
- everything: from a decoded PIL image to the final boxes

Each image is read and decoded before its clock starts, so disk reads are not
timed. The first --warmup images are run but not timed.

--preprocess decides where the letterbox runs (GPU options for torch-gpu*, trt):
    cpu     PIL resize on the CPU (decode.letterbox), the default
    gpu     the 0-255 image goes to the GPU, bilinear resize there, no antialias
    gpu-aa  the same with antialias (averages every pixel it covers, as PIL does)

Runtimes:
    onnx-cpu        ONNX Runtime on the CPU (.onnx)
    torch-cpu       PyTorch on the CPU (.pt, needs Ultralytics to load the weights)
    torch-gpu       PyTorch on CUDA, FP32 (.pt, needs Ultralytics)
    torch-gpu-fp16  PyTorch on CUDA, FP16 weights and input (.pt, needs Ultralytics)
    trt             TensorRT engine on CUDA (.engine from Ultralytics export,
                    FP32 or FP16 as it was built)

With --out, the boxes of every image (warm-up included) are also saved as COCO
results JSON, so detection-eval's scripts/eval_coco.py can score the runtime.
For mAP, use all val images and a low threshold:
    --count 548 --warmup 0 --conf 0.001 --out runs/coco_eval/<name>/detections.json

Usage (Ultralytics venv, from the repo root):
    python scripts/benchmark.py --runtime onnx-cpu --model runs/visdrone_yolo11s_640_baseline/weights/best.onnx
    python scripts/benchmark.py --runtime torch-gpu --model runs/visdrone_yolo11s_640_baseline/weights/best.pt
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
from aerial_object_detection.decode import boxes_to_original, letterbox, select_candidates  # noqa: E402
from detection_eval.boxes import cxcywh_to_xyxy  # noqa: E402
from detection_eval.coco import IMAGE_SUFFIXES, detections_to_coco, save_json  # noqa: E402
from detection_eval.nms import batched_nms  # noqa: E402


def onnx_model(path):
    """Returns (run, size): run takes the letterboxed array, returns (14, N)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]

    def run(x):
        return session.run(None, {model_input.name: x})[0][0]

    return run, model_input.shape[2]


def torch_model(path, device, size, half=False):
    """Same as onnx_model, with the PyTorch model inside the .pt file."""
    import torch
    from ultralytics import YOLO

    # fuse() folds batch norm into the conv layers, as the ONNX export does
    model = YOLO(str(path)).model.fuse().to(device).eval()
    dtype = torch.float16 if half else torch.float32
    model = model.to(dtype)

    @torch.inference_mode()
    def run(x):
        # x is a NumPy array (cpu letterbox) or already a GPU tensor (gpu letterbox)
        out = model(torch.as_tensor(x).to(device, dtype))[0][0]
        # .cpu() waits for the GPU to finish, so the time includes the work.
        # The rest of the pipeline is NumPy in FP32, as for the other runtimes.
        return out.float().cpu().numpy()

    return run, size


def trt_model(path):
    """Same as onnx_model, with a TensorRT engine built by Ultralytics export.

    FP32 or FP16 is decided when the engine is built, not here.
    """
    import json

    import tensorrt as trt
    import torch

    torch.cuda.init()
    with open(path, "rb") as f:
        data = f.read()
    # Ultralytics writes a JSON metadata block before the engine:
    # 4 bytes with its length, then the JSON. Engines without it are kept as they are.
    meta_len = int.from_bytes(data[:4], "little")
    try:
        json.loads(data[4 : 4 + meta_len])
        data = data[4 + meta_len :]
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(data)
    context = engine.create_execution_context()

    # One GPU tensor per engine input / output; the engine reads and writes them in place
    buffers, input_name, output_name = {}, None, None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = torch.float16 if engine.get_tensor_dtype(name) == trt.float16 else torch.float32
        buffers[name] = torch.empty(tuple(engine.get_tensor_shape(name)), dtype=dtype, device="cuda")
        context.set_tensor_address(name, buffers[name].data_ptr())
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name
    # TensorRT wants its own stream; on the default one it adds extra waits
    stream = torch.cuda.Stream()

    def run(x):
        x = torch.as_tensor(x)
        stream.wait_stream(torch.cuda.current_stream())  # a GPU letterbox must be finished
        with torch.cuda.stream(stream):
            buffers[input_name].copy_(x)
            context.execute_async_v3(stream.cuda_stream)
            # .cpu() waits for the engine to finish, as for torch-gpu
            return buffers[output_name][0].float().cpu().numpy()

    return run, buffers[input_name].shape[2]


def letterbox_gpu(image, size, antialias):
    """decode.letterbox on the GPU. Returns (x, (scale, pad_x, pad_y)), x a CUDA tensor."""
    import torch
    import torch.nn.functional as F

    # same size and padding as decode.letterbox, so boxes_to_original undoes it
    scale = min(size / image.width, size / image.height)
    new_w, new_h = round(image.width * scale), round(image.height * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

    # (H, W, 3) 0-255 integers, 1 byte per value, copied to the GPU as they are
    pixels = torch.from_numpy(np.array(image)).to("cuda")
    x = pixels.permute(2, 0, 1)[None].float() / 255.0  # (1, 3, H, W), 0-1
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=antialias)
    canvas = torch.full((1, 3, size, size), 114 / 255.0, device="cuda")
    canvas[:, :, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = x
    # the GPU runs this in the background; wait, so the model time does not include it
    torch.cuda.synchronize()
    return canvas, (scale, pad_x, pad_y)


def detect(run, prepare, image, conf, iou, max_det):
    """One image through the whole pipeline. prepare is the letterbox to use.

    Returns (xyxy, scores, classes, model seconds).
    """
    x, (scale, pad_x, pad_y) = prepare(image)
    start = time.perf_counter()
    pred = run(x)
    model_seconds = time.perf_counter() - start

    boxes, scores, classes = select_candidates(pred, conf)
    keep = batched_nms(cxcywh_to_xyxy(boxes), scores, classes, iou, max_det)
    xyxy = boxes_to_original(boxes[keep], scale, pad_x, pad_y, image.width, image.height)
    return xyxy, scores[keep], np.asarray(classes[keep], dtype=int), model_seconds


parser = argparse.ArgumentParser(description="Time a YOLO model with different runtimes.")
parser.add_argument("--runtime", required=True, choices=["onnx-cpu", "torch-cpu", "torch-gpu", "torch-gpu-fp16", "trt"])
parser.add_argument("--model", required=True, help=".onnx for onnx-cpu, .pt for torch-*, .engine for trt")
parser.add_argument("--imgsz", type=int, default=640, help="input size for torch-* (ONNX reads it from the file)")
parser.add_argument("--images", default=str(ROOT / "data/VisDrone/images/val"), help="image folder")
parser.add_argument("--count", type=int, default=100, help="timed images")
parser.add_argument("--warmup", type=int, default=10, help="untimed images before timing")
parser.add_argument("--conf", type=float, default=0.25, help="score threshold (as in live use, not mAP)")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
parser.add_argument("--preprocess", default="cpu", choices=["cpu", "gpu", "gpu-aa"], help="where the letterbox runs")
parser.add_argument("--out", help="also save the boxes as COCO results JSON")
args = parser.parse_args()
if args.preprocess != "cpu" and args.runtime in ("onnx-cpu", "torch-cpu"):
    parser.error("--preprocess gpu / gpu-aa needs a GPU runtime")

if args.runtime == "onnx-cpu":
    run, size = onnx_model(args.model)
elif args.runtime == "trt":
    run, size = trt_model(args.model)
else:
    device = "cpu" if args.runtime == "torch-cpu" else "cuda"
    run, size = torch_model(args.model, device, args.imgsz, half=args.runtime == "torch-gpu-fp16")

if args.preprocess == "cpu":
    def prepare(image):
        return letterbox(image, size)
else:
    def prepare(image):
        return letterbox_gpu(image, size, antialias=args.preprocess == "gpu-aa")

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
