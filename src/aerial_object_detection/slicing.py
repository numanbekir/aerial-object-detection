"""Sliced inference: run the model on overlapping tiles instead of one shrunk image.

At imgsz 640 a 1360 x 765 drone image is scaled to 640 x 360, so a 20-pixel
pedestrian becomes about 9 pixels. Cutting the image into 640 x 640 tiles keeps
every object at its original size. Neighbouring tiles overlap, so an object on
a tile border is fully inside at least one tile; the duplicates this creates
are removed with NMS after the boxes are shifted back to image pixels.

The whole image is also run once at the normal size, as SAHI does by default:
large objects that do not fit in one tile are still found there.
"""
import numpy as np

from detection_eval.nms import batched_nms


def tile_starts(length, tile, overlap):
    """Start positions of the tiles along one side of the image.

    Tiles are `tile` pixels long and step forward by `tile - overlap`. The last
    tile is pushed back so that it ends exactly at the image edge instead of
    hanging over it. A side not longer than one tile gets a single tile at 0.

    Example: length 1360, tile 640, overlap 128 -> step 512 -> [0, 512, 720].
    (A third step would start at 1024 and end at 1664, past the edge, so the
    last tile starts at 1360 - 640 = 720 instead.)

    Args:
        length: image width or height in pixels.
        tile: tile size in pixels.
        overlap: pixels shared by two neighbouring tiles (0 <= overlap < tile).

    Returns:
        list of int start positions, increasing, first one 0.
    """
    if length <= tile:
        return [0]

    step = tile - overlap
    starts = []
    start = 0
    while start + tile <= length:
        starts.append(start)
        start += step

    # The tiles that fit may stop short of the edge; one more tile, flush
    # with the edge, covers the rest.
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_origins(width, height, tile, overlap):
    """Top-left corner (x0, y0) of every tile, row by row."""
    xs = tile_starts(width, tile, overlap)
    ys = tile_starts(height, tile, overlap)
    return [(x0, y0) for y0 in ys for x0 in xs]


def predict_sliced(image, detect, tile=640, overlap=128, full_image=True, iou=0.7, max_det=300):
    """Detections on overlapping tiles, merged into one set for the whole image.

    Args:
        image: PIL image.
        detect: function, PIL image -> (xyxy, scores, classes) in the pixels of
            the image it was given (for example an OnnxDetector).
        tile: tile size in pixels.
        overlap: pixels shared by two neighbouring tiles.
        full_image: also run detect on the whole image.
        iou: NMS IoU threshold for merging.
        max_det: at most this many detections for the whole image.

    Returns:
        (xyxy, scores, classes) in the pixels of image, highest score first.
    """
    all_boxes, all_scores, all_classes = [], [], []

    if full_image:
        xyxy, scores, classes = detect(image)
        all_boxes.append(xyxy)
        all_scores.append(scores)
        all_classes.append(classes)

    for x0, y0 in tile_origins(image.width, image.height, tile, overlap):
        # A side shorter than one tile gives a smaller crop, never a crop past the edge.
        x1 = min(x0 + tile, image.width)
        y1 = min(y0 + tile, image.height)
        xyxy, scores, classes = detect(image.crop((x0, y0, x1, y1)))
        # Tile pixels -> image pixels: move every box by the tile's corner.
        all_boxes.append(xyxy + np.array([x0, y0, x0, y0], dtype=float))
        all_scores.append(scores)
        all_classes.append(classes)

    xyxy = np.concatenate([np.reshape(b, (-1, 4)) for b in all_boxes]).astype(float)
    scores = np.concatenate(all_scores).astype(float)
    classes = np.concatenate(all_classes).astype(int)

    keep = batched_nms(xyxy, scores, classes, iou, max_det)
    return xyxy[keep], scores[keep], classes[keep]
