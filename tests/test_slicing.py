import numpy as np
import pytest
from PIL import Image

from aerial_object_detection.slicing import predict_sliced, tile_origins, tile_starts


# ---- the VisDrone example ------------------------------------------------------

def test_visdrone_width():
    # step 512: 0, 512, then 1024 would end at 1664 > 1360, so the last tile ends at the edge
    assert tile_starts(1360, 640, 128) == [0, 512, 720]


def test_visdrone_height():
    # 512 + 640 = 1152 > 765, so the second tile is pushed back to 765 - 640 = 125
    assert tile_starts(765, 640, 128) == [0, 125]


# ---- short sides ---------------------------------------------------------------

def test_side_equal_to_tile():
    assert tile_starts(640, 640, 128) == [0]


def test_side_shorter_than_tile():
    assert tile_starts(500, 640, 128) == [0]


# ---- edges -----------------------------------------------------------------------

def test_exact_fit_adds_no_extra_tile():
    # 0 and 512: the second tile ends at 1152, exactly the edge. No third tile.
    assert tile_starts(1152, 640, 128) == [0, 512]


def test_one_pixel_past_exact_fit():
    assert tile_starts(1153, 640, 128) == [0, 512, 513]


def test_no_overlap():
    assert tile_starts(1280, 640, 0) == [0, 640]


def test_returns_plain_ints():
    starts = tile_starts(1360, 640, 128)
    assert all(type(s) is int for s in starts)


# ---- properties for many sizes -------------------------------------------------

@pytest.mark.parametrize("length", list(range(600, 4001, 37)))
@pytest.mark.parametrize("tile, overlap", [(640, 128), (640, 0), (512, 100), (1024, 256)])
def test_tiles_cover_the_side(length, tile, overlap):
    starts = tile_starts(length, tile, overlap)
    assert starts[0] == 0
    assert starts == sorted(set(starts)), "increasing, no duplicates"
    if length <= tile:
        assert starts == [0]
        return
    assert starts[-1] + tile == length, "last tile ends at the edge"
    for a, b in zip(starts, starts[1:]):
        assert b - a <= tile - overlap, "neighbours share at least `overlap` pixels"
    assert all(s + tile <= length for s in starts), "no tile hangs over the edge"


# ---- tile_origins -------------------------------------------------------------

def test_visdrone_origins():
    assert tile_origins(1360, 765, 640, 128) == [
        (0, 0), (512, 0), (720, 0),
        (0, 125), (512, 125), (720, 125),
    ]


# ---- predict_sliced with a fake detector -----------------------------------------

class WhiteSquareDetector:
    """Stand-in for the model: one box around the white pixels of the image it gets."""

    def __init__(self):
        self.sizes = []  # (width, height) of every image it was called on

    def __call__(self, image):
        self.sizes.append(image.size)
        white = np.asarray(image.convert("L")) > 128
        if not white.any():
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        ys, xs = np.nonzero(white)
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return np.array([box], dtype=float), np.array([0.9]), np.array([0])


def scene(width, height, square=None):
    """Black image with one white square (x1, y1, x2, y2), or none."""
    image = Image.new("RGB", (width, height))
    if square:
        image.paste((255, 255, 255), square)
    return image


def test_calls_full_image_and_every_tile():
    detector = WhiteSquareDetector()
    predict_sliced(scene(1360, 765), detector)
    assert detector.sizes == [(1360, 765)] + [(640, 640)] * 6


def test_without_full_image():
    detector = WhiteSquareDetector()
    predict_sliced(scene(1360, 765), detector, full_image=False)
    assert detector.sizes == [(640, 640)] * 6


def test_object_in_overlap_found_once():
    # x 560-580 is inside tiles 0-640 and 512-1152, y 100-120 only in the top row:
    # the full image and two tiles find it, NMS keeps one box.
    xyxy, scores, classes = predict_sliced(scene(1360, 765, (560, 100, 580, 120)), WhiteSquareDetector())
    assert xyxy.tolist() == [[560, 100, 580, 120]]
    assert classes.tolist() == [0]


def test_tile_boxes_are_shifted_to_image_pixels():
    # bottom-right corner, only in the tile that starts at (720, 125)
    square = (1300, 700, 1320, 720)
    xyxy, _, _ = predict_sliced(scene(1360, 765, square), WhiteSquareDetector(), full_image=False)
    assert xyxy.tolist() == [list(square)]


def test_cut_object_leaves_a_partial_box():
    # x 630-650 crosses the right edge of tile 0-640. That tile sees only 630-640
    # (IoU 0.5 with the whole box), which is below the NMS threshold 0.7, so the
    # partial box survives next to the whole one.
    xyxy, _, _ = predict_sliced(scene(1360, 765, (630, 100, 650, 120)), WhiteSquareDetector())
    assert sorted(xyxy.tolist()) == [[630, 100, 640, 120], [630, 100, 650, 120]]


def test_image_smaller_than_tile():
    detector = WhiteSquareDetector()
    xyxy, _, _ = predict_sliced(scene(500, 400, (10, 10, 30, 30)), detector)
    assert detector.sizes == [(500, 400), (500, 400)]
    assert xyxy.tolist() == [[10, 10, 30, 30]]


def test_nothing_found():
    xyxy, scores, classes = predict_sliced(scene(1360, 765), WhiteSquareDetector())
    assert xyxy.shape == (0, 4)
    assert len(scores) == 0 and len(classes) == 0


def test_max_det():
    def many(image):
        n = 50
        boxes = np.array([[i * 10, 0, i * 10 + 5, 5] for i in range(n)], dtype=float)
        return boxes, np.linspace(0.9, 0.1, n), np.zeros(n, dtype=int)

    xyxy, scores, _ = predict_sliced(scene(1360, 765), many, max_det=20)
    assert len(xyxy) == 20
    assert list(scores) == sorted(scores, reverse=True)
