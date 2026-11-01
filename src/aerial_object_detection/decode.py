"""Turn the raw output of a YOLO ONNX model into detections.

The model returns one array of shape (1, 4 + C, N) for one image:
N candidates (8400 at 640), each a column of 4 box numbers (cx, cy, w, h in
input pixels) followed by C class scores.

Steps for one image: letterbox -> model -> select_candidates -> NMS ->
boxes_to_original.
"""
import numpy as np
from PIL import Image

from detection_eval.boxes import cxcywh_to_xyxy


def letterbox(image, size):
    """Fit a PIL image into a size x size square for the model.

    The long side is scaled to size, the aspect ratio is kept and the rest is
    filled with grey (114), with the image in the middle.

    Returns:
        (x, (scale, pad_x, pad_y))
        x: (1, 3, size, size) float32, RGB, values 0-1.
        scale, pad_x, pad_y: what boxes_to_original needs to undo this.
    """
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    new_w, new_h = round(image.width * scale), round(image.height * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    # (H, W, 3) 0-255 -> (1, 3, H, W) 0-1
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, (scale, pad_x, pad_y)


def boxes_to_original(boxes, scale, pad_x, pad_y, img_w, img_h):
    """cx, cy, w, h in the letterboxed square -> x1, y1, x2, y2 in the original image.

    Boxes are clipped to the image: a box can reach into the grey padding.
    """
    xyxy = cxcywh_to_xyxy(boxes)
    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y
    xyxy /= scale
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img_h)
    return xyxy


def select_candidates(pred, conf_threshold=0.25):
    """Keep the candidates whose best class score is at least conf_threshold.

    Args:
        pred: (4 + C, N) raw output of one image (the model output without
            its first axis).
        conf_threshold: minimum score to keep a candidate.

    Returns:
        (boxes, scores, classes)
        boxes: (K, 4) float, cx, cy, w, h, one row per kept candidate.
        scores: (K,) float, the best class score of each kept candidate.
        classes: (K,) int, the class with that score.
        Kept candidates stay in their original column order.
    """
    boxes = pred[:4]
    class_scores = pred[4:]

    # Best class of every candidate: down the rows of each column.
    classes = class_scores.argmax(axis=0)
    scores = class_scores.max(axis=0)

    keep = scores >= conf_threshold
    # The model gives one box per column; return one box per row.
    return boxes[:, keep].T, scores[keep], classes[keep]
