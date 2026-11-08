"""One YOLO model behind different runtimes, with a CPU or GPU letterbox.

Every runtime gives a run(x) function: the letterboxed input in, the raw
(14, N) output out as a NumPy array. Around it the pipeline is always the same:
letterbox -> run -> select_candidates -> batched_nms -> boxes_to_original.

Runtimes:
    onnx-cpu        ONNX Runtime on the CPU (.onnx)
    torch-cpu       PyTorch on the CPU (.pt, needs Ultralytics to load the weights)
    torch-gpu       PyTorch on CUDA, FP32 (.pt, needs Ultralytics)
    torch-gpu-fp16  PyTorch on CUDA, FP16 weights and input (.pt, needs Ultralytics)
    trt             TensorRT engine on CUDA (.engine from Ultralytics export,
                    FP32 or FP16 as it was built)

Letterbox (prepare):
    cpu     Pillow resize on the CPU (decode.letterbox)
    gpu     the 0-255 image goes to the GPU, bilinear resize there, no antialias
    gpu-aa  the same with antialias (averages every pixel it covers, as Pillow does)

torch, tensorrt and onnxruntime are imported only by the runtime that needs them.
"""
import time

import numpy as np
from PIL import Image

from aerial_object_detection.decode import boxes_to_original, letterbox, select_candidates
from detection_eval.boxes import cxcywh_to_xyxy
from detection_eval.nms import batched_nms

RUNTIMES = ["onnx-cpu", "torch-cpu", "torch-gpu", "torch-gpu-fp16", "trt"]
PREPARES = ["cpu", "gpu", "gpu-aa"]


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


def load_runtime(runtime, model_path, imgsz=640):
    """Returns (run, size) for one of RUNTIMES. imgsz is used by torch-* only."""
    if runtime == "onnx-cpu":
        return onnx_model(model_path)
    if runtime == "trt":
        return trt_model(model_path)
    if runtime in ("torch-cpu", "torch-gpu", "torch-gpu-fp16"):
        device = "cpu" if runtime == "torch-cpu" else "cuda"
        return torch_model(model_path, device, imgsz, half=runtime == "torch-gpu-fp16")
    raise ValueError(f"unknown runtime {runtime!r}, expected one of {RUNTIMES}")


def image_size(image):
    """(width, height) of a PIL image or an (H, W, 3) array."""
    if isinstance(image, Image.Image):
        return image.size
    return image.shape[1], image.shape[0]


def letterbox_gpu(image, size, antialias, bgr=False):
    """decode.letterbox on the GPU. Returns (x, (scale, pad_x, pad_y)), x a CUDA tensor.

    image: a PIL image or an (H, W, 3) uint8 array, RGB (BGR with bgr=True,
    as OpenCV reads video frames; the channels are swapped on the GPU).
    """
    import torch
    import torch.nn.functional as F

    # same size and padding as decode.letterbox, so boxes_to_original undoes it
    width, height = image_size(image)
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

    # (H, W, 3) 0-255 integers, 1 byte per value, copied to the GPU as they are
    pixels = torch.from_numpy(np.array(image) if isinstance(image, Image.Image) else image).to("cuda")
    x = pixels.permute(2, 0, 1)[None].float() / 255.0  # (1, 3, H, W), 0-1
    if bgr:
        x = x[:, [2, 1, 0]]
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=antialias)
    canvas = torch.full((1, 3, size, size), 114 / 255.0, device="cuda")
    canvas[:, :, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = x
    # the GPU runs this in the background; wait, so the model time does not include it
    torch.cuda.synchronize()
    return canvas, (scale, pad_x, pad_y)


def make_prepare(kind, size, bgr=False):
    """Returns prepare(image) -> (x, (scale, pad_x, pad_y)) for one of PREPARES."""
    if kind == "cpu":

        def prepare(image):
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]) if bgr else image)
            return letterbox(image, size)

        return prepare
    if kind in ("gpu", "gpu-aa"):

        def prepare(image):
            return letterbox_gpu(image, size, antialias=kind == "gpu-aa", bgr=bgr)

        return prepare
    raise ValueError(f"unknown letterbox {kind!r}, expected one of {PREPARES}")


def detect(run, prepare, image, conf=0.25, iou=0.7, max_det=300):
    """One image through the whole pipeline.

    Returns (xyxy, scores, classes, model seconds); boxes in the pixels of image.
    """
    x, (scale, pad_x, pad_y) = prepare(image)
    start = time.perf_counter()
    pred = run(x)
    model_seconds = time.perf_counter() - start

    boxes, scores, classes = select_candidates(pred, conf)
    keep = batched_nms(cxcywh_to_xyxy(boxes), scores, classes, iou, max_det)
    width, height = image_size(image)
    xyxy = boxes_to_original(boxes[keep], scale, pad_x, pad_y, width, height)
    return xyxy, scores[keep], np.asarray(classes[keep], dtype=int), model_seconds
