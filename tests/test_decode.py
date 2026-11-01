import numpy as np
from PIL import Image

from aerial_object_detection.decode import boxes_to_original, letterbox, select_candidates

# 4 candidates (columns), 3 classes: car, bus, pedestrian.
EXAMPLE = np.array([
    # col 0  col 1  col 2  col 3
    [100.0, 200.0, 300.0, 400.0],  # cx
    [50.0, 60.0, 70.0, 80.0],      # cy
    [10.0, 20.0, 30.0, 40.0],      # w
    [5.0, 6.0, 7.0, 8.0],          # h
    [0.90, 0.10, 0.30, 0.05],      # car
    [0.20, 0.70, 0.20, 0.10],      # bus
    [0.10, 0.10, 0.20, 0.15],      # pedestrian
])


def test_example():
    boxes, scores, classes = select_candidates(EXAMPLE, 0.25)
    # Column 3: best score 0.15 (pedestrian) is below 0.25, dropped.
    assert boxes.tolist() == [[100, 50, 10, 5], [200, 60, 20, 6], [300, 70, 30, 7]]
    assert scores.tolist() == [0.90, 0.70, 0.30]
    assert classes.tolist() == [0, 1, 0]


def test_shapes_are_rows():
    boxes, scores, classes = select_candidates(EXAMPLE, 0.25)
    assert boxes.shape == (3, 4)
    assert scores.shape == (3,) and classes.shape == (3,)


def test_classes_are_integers():
    _, _, classes = select_candidates(EXAMPLE, 0.25)
    assert np.issubdtype(classes.dtype, np.integer)


def test_threshold_is_inclusive():
    _, scores, _ = select_candidates(EXAMPLE, 0.30)
    assert scores.tolist() == [0.90, 0.70, 0.30]


def test_low_threshold_keeps_all():
    _, _, classes = select_candidates(EXAMPLE, 0.0)
    assert classes.tolist() == [0, 1, 0, 2]


def test_nothing_kept():
    boxes, scores, classes = select_candidates(EXAMPLE, 0.95)
    assert boxes.shape == (0, 4)
    assert len(scores) == 0 and len(classes) == 0


def test_model_sized_input():
    rng = np.random.default_rng(0)
    pred = rng.random((14, 8400))
    boxes, scores, classes = select_candidates(pred, 0.9)
    expected = pred[4:].max(axis=0) >= 0.9
    assert boxes.shape == (expected.sum(), 4)
    assert np.array_equal(boxes, pred[:4, expected].T)
    assert classes.max() <= 9


# ---- letterbox and boxes_to_original ------------------------------------------

GREY = 114 / 255


def test_letterbox_wide_image():
    # 200 x 100 red image into a 100 x 100 square: scale 0.5, 100 x 50, 25 px grey above and below.
    x, (scale, pad_x, pad_y) = letterbox(Image.new("RGB", (200, 100), (255, 0, 0)), 100)
    assert x.shape == (1, 3, 100, 100) and x.dtype == np.float32
    assert (scale, pad_x, pad_y) == (0.5, 0, 25)
    assert np.allclose(x[0, :, 10, 50], GREY)          # top padding
    assert np.allclose(x[0, :, 50, 50], [1, 0, 0])     # image, red
    assert np.allclose(x[0, :, 90, 50], GREY)          # bottom padding


def test_letterbox_tall_image():
    _, (scale, pad_x, pad_y) = letterbox(Image.new("RGB", (100, 200)), 100)
    assert (scale, pad_x, pad_y) == (0.5, 25, 0)


def test_letterbox_scales_up_small_images():
    _, (scale, pad_x, pad_y) = letterbox(Image.new("RGB", (50, 25)), 100)
    assert (scale, pad_x, pad_y) == (2.0, 0, 25)


def test_boxes_to_original():
    # Same setting as the wide image: scale 0.5, pad_y 25.
    # In the square: centre (50, 50), 10 x 10 -> corners 45, 45, 55, 55.
    # Minus padding: 45, 20, 55, 30. Divided by 0.5: 90, 40, 110, 60.
    out = boxes_to_original(np.array([[50.0, 50, 10, 10]]), 0.5, 0, 25, 200, 100)
    assert out.tolist() == [[90, 40, 110, 60]]


def test_boxes_to_original_clips_to_image():
    # A box that reaches into the top padding and past the right edge.
    out = boxes_to_original(np.array([[98.0, 28, 10, 10]]), 0.5, 0, 25, 200, 100)
    assert out.tolist() == [[186, 0, 200, 16]]


def test_boxes_to_original_empty():
    assert boxes_to_original(np.zeros((0, 4)), 0.5, 0, 25, 200, 100).shape == (0, 4)


def test_round_trip_on_visdrone_size():
    # 1360 x 765 -> 640: scale 640/1360, 640 x 360, 140 px above and below.
    _, (scale, pad_x, pad_y) = letterbox(Image.new("RGB", (1360, 765)), 640)
    assert (pad_x, pad_y) == (0, 140)
    original = np.array([[700.0, 600, 760, 700]])            # x1, y1, x2, y2
    cx, cy = (700 + 760) / 2 * scale + pad_x, (600 + 700) / 2 * scale + pad_y
    w, h = 60 * scale, 100 * scale
    back = boxes_to_original(np.array([[cx, cy, w, h]]), scale, pad_x, pad_y, 1360, 765)
    assert np.allclose(back, original)
