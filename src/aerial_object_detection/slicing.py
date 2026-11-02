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


def sliced_candidates(image, detect, tile=640, overlap=128, full_image=True, edge_margin=2):
    """Boxes from the whole image and from every tile, all in image pixels.

    Args:
        image: PIL image.
        detect: function, PIL image -> (xyxy, scores, classes) in the pixels of
            the image it was given (for example an OnnxDetector).
        tile: tile size in pixels.
        overlap: pixels shared by two neighbouring tiles.
        full_image: also run detect on the whole image.
        edge_margin: a tile box this close (in pixels) to an inner tile border
            counts as touching it.

    Returns:
        (xyxy, scores, classes, at_edge). at_edge is True for a tile box that
        touches a border between two tiles (not the image border): the object
        may be cut there, so the box may be only part of it.
    """
    all_boxes, all_scores, all_classes, all_edge = [], [], [], []

    def add(xyxy, scores, classes, at_edge):
        all_boxes.append(np.reshape(np.asarray(xyxy, dtype=float), (-1, 4)))
        all_scores.append(np.asarray(scores, dtype=float))
        all_classes.append(np.asarray(classes, dtype=int))
        all_edge.append(at_edge)

    if full_image:
        xyxy, scores, classes = detect(image)
        add(xyxy, scores, classes, np.zeros(len(scores), dtype=bool))

    for x0, y0 in tile_origins(image.width, image.height, tile, overlap):
        # A side shorter than one tile gives a smaller crop, never a crop past the edge.
        x1 = min(x0 + tile, image.width)
        y1 = min(y0 + tile, image.height)
        xyxy, scores, classes = detect(image.crop((x0, y0, x1, y1)))
        xyxy = np.reshape(np.asarray(xyxy, dtype=float), (-1, 4))

        # Borders shared with another tile; the image's own border is not one.
        at_edge = np.zeros(len(xyxy), dtype=bool)
        if x0 > 0:
            at_edge |= xyxy[:, 0] <= edge_margin
        if y0 > 0:
            at_edge |= xyxy[:, 1] <= edge_margin
        if x1 < image.width:
            at_edge |= xyxy[:, 2] >= (x1 - x0) - edge_margin
        if y1 < image.height:
            at_edge |= xyxy[:, 3] >= (y1 - y0) - edge_margin

        # Tile pixels -> image pixels: move every box by the tile's corner.
        add(xyxy + np.array([x0, y0, x0, y0], dtype=float), scores, classes, at_edge)

    return (np.concatenate(all_boxes), np.concatenate(all_scores),
            np.concatenate(all_classes), np.concatenate(all_edge))


def box_ios(a, b):
    """Intersection over the smaller box, for every pair: (N, 4), (M, 4) -> (N, M).

    1.0 when one box lies completely inside the other, whatever their sizes.
    """
    a = np.asarray(a, dtype=float)[:, None, :]
    b = np.asarray(b, dtype=float)[None, :, :]
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    smaller = np.minimum(area_a, area_b)
    return np.divide(w * h, smaller, out=np.zeros_like(smaller), where=smaller > 0)


def merge_nms(xyxy, scores, classes, at_edge, iou=0.7, max_det=300):
    """Class-aware NMS over all boxes. Partial boxes from tile borders can survive."""
    keep = batched_nms(xyxy, scores, classes, iou, max_det)
    return xyxy[keep], scores[keep], classes[keep]


def merge_edge(xyxy, scores, classes, at_edge, iou=0.7, max_det=300):
    """Drop tile boxes that touch an inner tile border, then class-aware NMS.

    A box at an inner border lies in the overlap, so a small object there is
    whole in the neighbouring tile, and a large one is found on the whole image.
    """
    xyxy, scores, classes = xyxy[~at_edge], scores[~at_edge], classes[~at_edge]
    keep = batched_nms(xyxy, scores, classes, iou, max_det)
    return xyxy[keep], scores[keep], classes[keep]


def merge_ios(xyxy, scores, classes, at_edge, ios=0.5, max_det=300):
    """Greedy merging by IoS, the SAHI default (GREEDYNMM, IOS 0.5).

    From the highest score down: a box takes every remaining box of its class
    whose IoS with it is above `ios`, grows to the box that covers them all,
    and keeps its own score.
    """
    order = np.argsort(-scores, kind="stable")
    used = np.zeros(len(scores), dtype=bool)
    out_boxes, out_scores, out_classes = [], [], []

    for i in order:
        if used[i]:
            continue
        used[i] = True
        rest = np.nonzero(~used & (classes == classes[i]))[0]
        matched = rest[box_ios(xyxy[[i]], xyxy[rest])[0] > ios]
        used[matched] = True

        group = xyxy[np.concatenate([[i], matched])]
        out_boxes.append([group[:, 0].min(), group[:, 1].min(), group[:, 2].max(), group[:, 3].max()])
        out_scores.append(scores[i])
        out_classes.append(classes[i])
        if len(out_scores) == max_det:
            break

    return (np.reshape(np.array(out_boxes, dtype=float), (-1, 4)),
            np.array(out_scores, dtype=float), np.array(out_classes, dtype=int))


MERGES = {"nms": merge_nms, "edge": merge_edge, "ios": merge_ios}


def predict_sliced(image, detect, tile=640, overlap=128, full_image=True, iou=0.7, max_det=300,
                   merge="nms"):
    """Detections on overlapping tiles, merged into one set for the whole image.

    merge: "nms" (class-aware NMS at `iou`), "edge" (drop boxes at inner tile
    borders, then NMS) or "ios" (SAHI-style greedy merging, IoS 0.5).

    Returns:
        (xyxy, scores, classes) in the pixels of image, highest score first.
    """
    candidates = sliced_candidates(image, detect, tile, overlap, full_image)
    if merge == "ios":
        return merge_ios(*candidates, max_det=max_det)
    return MERGES[merge](*candidates, iou=iou, max_det=max_det)
