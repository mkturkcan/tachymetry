# Detection

Detection turns frames into per-frame road-user boxes, scores, classes and, for cuboid detectors, the projected corners of a 3D box. It is the only stage that needs a GPU for practical speed. Its output is a `DetectionSet` (`detections.py`) cached as `detections.npz` in the run directory ([Pipeline](pipeline.md#caching-and-re-runs)).

## Detectors

`DetectorConfig.name` selects the back-end (`detect/backends.py`, `build_detector`):

| Name | Model | Cuboids |
|---|---|---|
| `uod2-n`, `uod2-s`, `uod2-m`, `uod2-l`, `uod2-x` (default `uod2-x`) | [UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2): a YOLO26 detect-and-pose network that also regresses the eight projected corners of a 3D cuboid per road user, without camera calibration. The checkpoint `checkpoints/urbanomnidetect_v2_<size>_640.pt` is downloaded from Hugging Face on first use (`uod2_checkpoint`). | yes |
| `yolo:<weights>`, or a path ending in `.pt`, `.engine` or `.onnx` | Any Ultralytics detection checkpoint (e.g. `yolo:yolo11x.pt`, a fine-tuned `.pt`, an exported TensorRT `.engine`). | no |

`DetectorConfig.weights` points to a local checkpoint and overrides the download. `tachymetry download --detector uod2-l` fetches UrbanOmniDetect-2 weights ahead of time. Detectors that are not Ultralytics models are plugged in as Python objects; see [Custom detectors](../examples/custom-detector.md).

!!! note "`precomputed:` detectors"
    The comment on `DetectorConfig.name` lists `precomputed:<dir>`, but `build_detector` in version 0.1.0 does not implement it and raises `ValueError`. To reuse detections, keep `detections.npz` in the run directory (it is reused automatically) or wrap an external source in a `Detector` object.

## Class taxonomy

All back-ends map their labels into `ROAD_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")` (`config.py`). For COCO-trained Ultralytics checkpoints the mapping is `COCO_TO_ROAD = {0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck}`; a label whose name already is a road class maps to itself. `DetectorConfig.classes` restricts the classes that are kept.

Association and verdicts use three super-classes, `SUPER_CLASS`: `vehicle` (car, bus, truck), `two_wheeler` (bicycle, motorcycle) and `person`. Label flicker inside a super-class (car and truck, typically) therefore never splits a track; the final class of a track is decided offline by a weighted vote ([Association and track hygiene](tracking.md#class-vote)).

## Cuboid corners

For UrbanOmniDetect-2, `DetectionSet.cuboid` holds an `(N, 8, 3)` array of `x, y, confidence` per corner in raw image pixels (`detections.FrameDetections`). Corners 0 to 3 touch the ground and corners 4 to 7 are the roof above them; corners 0 and 3 form the front face, and the edges 0-1 and 3-2 are the long sides. In `track/measure.py` the ground corners are back-projected to the road plane; the front is the midpoint of corners 0 and 3, the rear the midpoint of corners 1 and 2, the length is the mean of the 0-1 and 3-2 edges and the width the mean of the 0-3 and 1-2 edges.

## Per-frame post-processing

`UltralyticsDetector.__call__` applies these steps to every image, in this order. Defaults are from `DetectorConfig`.

1. **Inference** at `imgsz = 960` with the model threshold `conf = 0.10`, FP16 on CUDA when `half = True`, `batch = 8` frames per call (`detect_video`).
2. **Class mapping and selection** into `ROAD_CLASSES`; labels outside `classes` are dropped.
3. **Class floors** `class_conf = {person: 0.40, bicycle: 0.50, motorcycle: 0.50}`: detections of these classes below their floor are dropped. Other classes keep everything above `conf`.
4. **Width limit**: boxes wider than `max_box_width_frac` of the image width are dropped (default 1.01, i.e. disabled).
5. **Near-duplicate suppression** (`nms_by_group`): greedy NMS at IoU above 0.8 (fixed) within each super-class, which removes double reports of one object with two labels.
6. **Cuboid trust** (`cuboid_trust`): a cuboid is used only if all eight corners are finite, the weakest corner confidence is at least `cuboid_conf = 0.5`, every corner lies inside the box expanded by 0.75 box widths and heights on each side, and no corner collapsed to the image origin. Untrusted cuboids are replaced by NaN and never reach the geometry.
7. **Nested-vehicle suppression** (`suppress_nested`): a vehicle box that lies at least `nested_containment = 0.85` inside a larger vehicle box (intersection over its own area) is dropped when both stand on the same ground: their trusted ground quads (corners 0 to 3, in the image) overlap by at least `nested_ground_overlap = 0.30` of the smaller quad, or, without trusted cuboids, their bottom edges are within 15 % of the larger box height. This removes truck cabs and trailers reported next to the whole truck, and keeps a car standing in front of a bus, which is nested in the image but not on the ground.

Detections with scores between `conf` and `TrackerConfig.high_conf = 0.50` are kept on purpose: they never start a track but extend confirmed tracks in the second association stage ([Association](tracking.md#online-association)).

## Frame rate and decoding

`detect_video` analyses every `round(fps / rate_hz)`-th frame (`rate_hz = 15` by default; `0` analyses every frame). Frames are decoded sequentially (skipped frames are grabbed, not decoded) in a background thread with a queue of `2 × batch` frames, which overlaps decoding with inference. Processed frame indices are stored in `DetectionSet.processed_frames`, so frames without any detection are still known to the tracker.

Measured on the public sample clip (1920×1080, 929 frames, 465 analysed) with `uod2-x`, `imgsz` 960, FP16, batch 8 on one NVIDIA RTX PRO 6000 Blackwell Max-Q: 6.95 s for the detection stage of a full run, including model loading.

## Detection without the pipeline

```python
from tachymetry.config import DetectorConfig
from tachymetry.data import sample_video
from tachymetry.detect import build_detector
from tachymetry.detect.run import detect_video
from tachymetry.io import VideoSource

detector = build_detector(DetectorConfig(name="uod2-x"), device="cuda:0")
with VideoSource(sample_video("intersection-day")) as src:
    dets = detect_video(src, detector, rate_hz=5, max_frames=20)
print(len(dets), dets.cuboid.shape, dets.meta["stride"], dets.meta["rate_hz"])
print(dets.class_names)
```

```text
833 (833, 8, 3) 6 5.0
('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck')
```

`DetectionSet` stores detections in columnar form (`frame`, `time`, `xyxy`, `score`, `cls`, `cuboid`), iterates per processed frame with `frames()`, and round-trips through `save()` and `load()`.

## Box-only detectors

Without cuboids, three parts of the pipeline lose evidence:

* **Footprint centres** fall back to the box-contact model with class priors ([Association and track hygiene](tracking.md#centre-estimation)).
* **Automatic calibration** loses the `cuboids` and `fused` candidates and the vehicle-size metric scale; it falls back to road markings and returns a camera of relative scale (camera height 1, with a warning) unless a metric cue is given. Pass a calibrated camera (`camera="camera.json"`) or a scale cue such as `calibration.scale = "height=8.5"`.
* **The parked-vehicle ghost rule** uses the absence of trusted footprints as evidence only when the detector provides cuboids ([Verdicts](verdicts.md#ghosts)).
