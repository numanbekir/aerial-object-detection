# aerial-object-detection

Small object detection in drone images: YOLO11s trained on
[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) (10 classes, many objects under
32 pixels) at two input sizes, then run without Ultralytics through ONNX Runtime with my own
letterbox, decoding and NMS.

Every mAP here is pycocotools, reproduced to the last digit by
[detection-eval](https://github.com/numanbekir/detection-eval), my evaluation code written from
scratch. That repo also explains why Ultralytics reports about 0.03–0.04 more mAP50 for the
same weights.

- **Try it in the browser:** [tanirim.com/projeler/havadan-nesne-tespiti](https://tanirim.com/projeler/havadan-nesne-tespiti)
  (the ONNX models run on your device, 640 and 1280 side by side)
- **Models:** [huggingface.co/numanbekir/visdrone-yolo11s](https://huggingface.co/numanbekir/visdrone-yolo11s)

## Results

`best.pt` of each run, selected on val. Val: 548 images, 38,759 objects. Test-dev: 1,610 images,
75,102 objects, not used for training or model selection.

| split | mAP | imgsz 640 | imgsz 1280 |
|---|---|---|---|
| val | mAP50 | 0.347 | 0.503 |
| val | mAP50-95 | 0.207 | 0.315 |
| test-dev | mAP50 | 0.288 | 0.411 |
| test-dev | mAP50-95 | 0.169 | 0.246 |

**1280 beats 640 on both splits:** +0.156 mAP50 on val, +0.123 on test-dev. Test-dev is lower
than val because its scenes are harder for this model (people and pedestrians most of all), not
because it is more crowded (46.6 objects per image vs 70.7 on val).

### AP50 per Class

| class | val 640 | val 1280 | test-dev 640 | test-dev 1280 |
|---|---|---|---|---|
| pedestrian | 0.405 | 0.620 | 0.259 | 0.418 |
| people | 0.279 | 0.476 | 0.109 | 0.252 |
| bicycle | 0.115 | 0.319 | 0.075 | 0.195 |
| car | 0.775 | 0.862 | 0.695 | 0.797 |
| van | 0.340 | 0.472 | 0.256 | 0.370 |
| truck | 0.315 | 0.470 | 0.366 | 0.506 |
| tricycle | 0.242 | 0.403 | 0.185 | 0.280 |
| awning-tricycle | 0.100 | 0.179 | 0.142 | 0.233 |
| bus | 0.489 | 0.609 | 0.518 | 0.604 |
| motor | 0.411 | 0.621 | 0.278 | 0.452 |

### AP by Object Size

AP at IoU 0.50:0.95, object area in original image pixels.

| size | val 640 | val 1280 | test-dev 640 | test-dev 1280 |
|---|---|---|---|---|
| small (< 32²) | 0.111 | 0.223 | 0.078 | 0.141 |
| medium | 0.320 | 0.436 | 0.262 | 0.361 |
| large (> 96²) | 0.443 | 0.529 | 0.408 | 0.483 |

At 640 a 1360 × 765 image is scaled down to 640 × 360, so a 20-pixel pedestrian becomes about
9 pixels. Small objects gain the most from 1280 (AP almost doubles on both splits). The cost is
speed: 3.4 ms vs 10.3 ms per image on the GPU (Ultralytics val, RTX 3060 Laptop).

### One Image

![True positives, false positives and missed objects at 640 and 1280](docs/matches_640_vs_1280.jpg)

Part of val image `0000295_02400_d_0000033` (317 objects), detections with score ≥ 0.3,
IoU ≥ 0.5. Green: true positive, red: false positive, orange: missed. Counts are for the whole
image. At 1280 the model finds 63 more objects with 10 more false positives.

## Without Ultralytics: ONNX Runtime

The same weights exported to ONNX and run with ONNX Runtime. Everything around the model is this
repo's code: letterbox, decoding the raw `(1, 14, 8400)` output (`decode.py`), class-aware NMS
(`nms.py` from detection-eval). Settings as in Ultralytics: score ≥ 0.001, NMS IoU 0.7, at most
300 detections per image. mAP50:

| | Ultralytics predict | ONNX Runtime + this repo | difference |
|---|---|---|---|
| 640, val | 0.347 | 0.355 | +0.008 |
| 640, test-dev | 0.288 | 0.294 | +0.006 |
| 1280, val | 0.503 | 0.505 | +0.002 |

The whole difference comes from how the image is shrunk to the input size. Ultralytics uses
OpenCV `INTER_LINEAR` (in training too), which reads only a few source pixels for every new
pixel. This repo uses Pillow bilinear, which averages over all the source pixels it replaces.
With OpenCV `INTER_LINEAR` in the same ONNX pipeline, 640 val drops back to 0.347
(`INTER_AREA`: 0.354), so NMS and decoding match Ultralytics. At 1280 the image is only scaled
to 0.94 and the gain almost disappears. At 640 the gain is spread over all object sizes, not
only small ones.

CPU time per image (i7-11800H, ONNX Runtime): 49 ms model / 73 ms in total at 640,
264 ms / 303 ms at 1280.

## How It Works

```
src/aerial_object_detection/
  decode.py        letterbox, raw YOLO output -> boxes, back to original pixels
scripts/
  train.py         train YOLO11 on VisDrone (Ultralytics)
  predict_coco.py  Ultralytics predict on a split, detections as COCO results JSON
  inspect_onnx.py  input / output shapes and raw candidates of an ONNX model
  predict_onnx.py  ONNX Runtime + decode.py + NMS, COCO results JSON without Ultralytics
```

Scoring: `scripts/eval_coco.py` in detection-eval (pycocotools and detection-eval side by side).

## Usage

Requires [uv](https://docs.astral.sh/uv/). The dataset is not included: download and convert
VisDrone with the Ultralytics [VisDrone.yaml](https://docs.ultralytics.com/datasets/detect/visdrone/),
then put the YAML at `data/VisDrone.yaml` with `path` pointing to the dataset.

```bash
uv sync                  # detection-eval, numpy, pillow, onnxruntime + pycocotools, pytest
uv run pytest -q         # 14 tests
```

Training and Ultralytics prediction (`uv sync --group train`):

```bash
uv run python scripts/train.py --imgsz 640
uv run python scripts/train.py --imgsz 1280 --batch 4 --name visdrone_yolo11s_1280
uv run python scripts/predict_coco.py --weights runs/visdrone_yolo11s_640_baseline/weights/best.pt \
    --imgsz 640 --out runs/coco_eval/yolo11s_640_val/detections.json
```

ONNX Runtime (the export itself needs Ultralytics):

```bash
yolo export model=runs/visdrone_yolo11s_640_baseline/weights/best.pt format=onnx imgsz=640
uv run python scripts/predict_onnx.py --model runs/visdrone_yolo11s_640_baseline/weights/best.onnx \
    --out runs/coco_eval/yolo11s_640_onnx_val/detections.json
```

For test-dev, add `--images data/VisDrone/images/test`. Scoring, from a detection-eval checkout:

```bash
uv run python scripts/eval_coco.py --data <this repo>/data/VisDrone.yaml \
    --detections <this repo>/runs/coco_eval/yolo11s_640_val/detections.json
```

Training settings: YOLO11s from COCO-pretrained weights, 50 epochs, seed 0, batch 16 at 640 and
4 at 1280 (`nbs=64` keeps the update size the same), RTX 3060 Laptop 6 GB: about 1 h 50 min and
7 h 20 min.

## Limitations

- **Ignored regions.** VisDrone marks some areas as "ignored". The Ultralytics converter drops
  them, so detections there count as false positives in every table above.
- **One seed per run.** Differences below about 0.005 mAP may be noise.

## Next

- Sliced inference (SAHI-style): tiles at full resolution, merged with NMS, measured on val.
- Video and FPS.
- Speed and mAP table: PyTorch, ONNX Runtime, TensorRT FP16, browser.

## License

Code: MIT. VisDrone is licensed for academic use only and is not part of this repository.
