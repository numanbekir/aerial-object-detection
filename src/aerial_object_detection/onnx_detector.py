"""A YOLO ONNX model as a function: PIL image in, boxes in image pixels out."""
import time

import numpy as np
import onnxruntime as ort

from aerial_object_detection.decode import boxes_to_original, letterbox, select_candidates
from detection_eval.boxes import cxcywh_to_xyxy
from detection_eval.nms import batched_nms


class OnnxDetector:
    """letterbox -> ONNX Runtime -> select_candidates -> batched_nms -> boxes_to_original.

    Calling the detector on a PIL image returns (xyxy, scores, classes):
    xyxy (K, 4) in the pixels of that image, scores (K,), classes (K,) int,
    highest score first.

    model_seconds adds up the time spent inside ONNX Runtime, without the
    resize and the decoding around it.
    """

    def __init__(self, model_path, conf=0.001, iou=0.7, max_det=300):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.size = model_input.shape[2]  # square input: (1, 3, size, size)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.model_seconds = 0.0

    def __call__(self, image):
        x, (scale, pad_x, pad_y) = letterbox(image, self.size)
        start = time.perf_counter()
        pred = self.session.run(None, {self.input_name: x})[0][0]
        self.model_seconds += time.perf_counter() - start

        boxes, scores, classes = select_candidates(pred, self.conf)
        keep = batched_nms(cxcywh_to_xyxy(boxes), scores, classes, self.iou, self.max_det)
        xyxy = boxes_to_original(boxes[keep], scale, pad_x, pad_y, image.width, image.height)
        return xyxy, scores[keep], np.asarray(classes[keep], dtype=int)
