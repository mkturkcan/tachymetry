# Detection

Detection turns frames into per-frame road-user boxes, scores and classes, plus instance masks for segmenting detectors and the projected corners of a 3D box for cuboid detectors. It is the only stage that needs a GPU for practical speed. Its output is a `DetectionSet` (`detections.py`) cached as `detections.npz` in the run directory ([Pipeline](pipeline.md#caching-and-re-runs)).

## Detectors

`DetectorConfig.name` selects the back-end (`detect/backends.py`, `build_detector`). SAM 3 is the default; the others are alternatives:

| Name | Model | Masks | Cuboids |
|---|---|---|---|
| `sam3` (default) | [SAM 3](https://huggingface.co/facebook/sam3) (Meta) through Hugging Face `transformers` (`Sam3Model`, `Sam3Processor`), prompted with text (`detect/sam3.py`, prompts below). The checkpoint `facebook/sam3` is gated. | yes | no |
| `rfdetr-seg-l`, `rfdetr-seg-m`, `rfdetr-seg-s`, `rfdetr-seg-n` | [RF-DETR-Seg](https://github.com/roboflow/rf-detr) (Roboflow): a real-time DETR with a DINOv2 backbone and an instance-segmentation head, trained on COCO. Apache-2.0 weights, downloaded to `~/.roboflow/models` (or `$RF_HOME`) on first use (`detect/rfdetr.py`). | yes | no |
| `rfdetr-l`, `rfdetr-m`, `rfdetr-s`, `rfdetr-n` | RF-DETR detection models (boxes only), same source and licence. | no | no |
| `uod2-x`, `uod2-l`, `uod2-m`, `uod2-s`, `uod2-n` | [UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2): a YOLO26 detect-and-pose network that also regresses the eight projected corners of a 3D cuboid per road user. Runs on Ultralytics (AGPL-3.0), which is installed only with the optional extra: `pip install "tachymetry[uod2]"`. The checkpoint `checkpoints/urbanomnidetect_v2_<size>_640.pt` is downloaded from Hugging Face on first use (`uod2_checkpoint`). | no | yes |

The vehicle geometry of the default pipeline needs instance masks ([Vehicle footprints from masks](footprints.md)), and so does the vehicle-silhouette scale of automatic calibration ([Automatic calibration](auto-calibration.md)). Detectors without masks are supported with reduced evidence ([below](#detectors-without-masks)). The `yolo:<weights>` back-end of earlier versions is removed.

The RF-DETR XLarge and 2XLarge models are not offered. `DetectorConfig.weights` overrides the download: a fine-tuned RF-DETR checkpoint of the selected size (its labels are mapped to the road classes by name), a local UrbanOmniDetect-2 `.pt`, or, for `sam3`, another Hugging Face repository or a local folder with the `transformers` checkpoint. `tachymetry download --detector <name>` fetches weights ahead of time. Other detectors are plugged in as Python objects; see [Custom detectors](../examples/custom-detector.md).

!!! note "SAM 3 access"
    `facebook/sam3` is a gated repository. Request access on its [model page](https://huggingface.co/facebook/sam3), wait for the approval, then authenticate the machine with `hf auth login` (or set `HF_TOKEN`). Without access, `build_detector` raises a `PermissionError` with these instructions.

!!! note "Precomputed detections"
    `detector.name = "precomputed:<path>"` makes the pipeline load detections instead of running a detector (`pipeline._load_precomputed`). The path is either an `.npz` file in the format of `detections.npz` (`DetectionSet.save`), or a folder that holds `<video stem>.npz` or `<video stem>/detections.npz`, for example the output root of an earlier batch. Example: `tachymetry run clip.mp4 --detector precomputed:runs/clip/detections.npz --out runs2`. Detections in other formats are passed through a `Detector` object ([Custom detectors](../examples/custom-detector.md)).

## SAM 3 prompts

SAM 3 is prompted on two levels (`detect/sam3.py`):

* **Detection prompts**: `vehicle` and `person` (`SAM3_PROMPTS`). Their results are the detections that are tracked, with boxes and instance masks, kept above `conf` = 0.40 (`SAM3_CONF`). The score is the class probability times the presence probability.
* **Class-evidence prompts**: `car`, `van`, `suv`, `bus`, `truck`, `bike` and `motorcycle` (`SAM3_EVIDENCE`) run in the same forward pass. Their results vote for the class of the detections they overlap; they are not tracked themselves, except for unclaimed two-wheelers (below).
* **Unclaimed two-wheelers**: `bike` or `motorcycle` evidence that no vehicle detection covers (intersection over the smaller box below 0.6), scored at least `conf`, is kept as a detection of its own, so cyclists that the `vehicle` prompt misses are counted.

**Evidence vectors.** Every kept detection carries a vector (`EVIDENCE_COLUMNS`, stored as `evidence` in `detections.npz`):

| Column | Value |
|---|---|
| `car`, `van`, `suv`, `bus`, `truck`, `bike`, `motorcycle` | highest score among that prompt's detections whose box has an IoU above 0.5 with the detection's box (`SAM3_EVIDENCE_IOU`), counting only scores of at least 0.35 for `car`, `van` and `suv` and at least 0.40 for the others (`SAM3_EVIDENCE_MIN`); 0 without a match |
| `bike_cover` | highest score of a `bike` detection scoring at least 0.5 that lies within the detection's box (intersection over the bike box above 0.6) |
| `bike_ride` | highest score of a `bike` detection scoring at least 0.5 whose box, enlarged by 15 % on each side, covers more than 0.30 of the detection's box (for persons: a bicycle or motorcycle under them) |

The class of a track is decided from these vectors over all its frames, weighted toward large, central, near-field views ([class vote](tracking.md#class-vote)): the argmax over the two-wheeler (the larger of `bike` and `motorcycle`), `bus`, `truck`, `car`, `van` and `suv` shares, with `car`, `van` and `suv` folded into `car`; `motorcycle` against `bike` splits two-wheelers into motorcycles and bicycles. The `bike_cover` and `bike_ride` columns separate two-wheelers and their riders from cars and pedestrians ([Verdicts](verdicts.md#riders-inferred-cyclists-and-passengers)). The class stored with each detection is the same rule applied to a single frame, and a `vehicle` detection without evidence is a car.

The image encoder runs once per frame and the prompt-conditioned decoder once per (frame, prompt) pair, eight pairs per pass (`PAIRS_PER_PASS`). SAM 3 is a set predictor without non-maximum suppression; duplicates are removed by mask overlap (step 6 below).

## Score profiles

Score scales differ between back-ends, and some tracking thresholds refer to the score. `DETECTOR_SCORE_PROFILES` (`config.py`) adjusts them for a back-end when they are left at their defaults (`PipelineConfig.with_detector_profile`, applied when a `Pipeline` is created). The only profile is `rfdetr`: `tracker.high_conf` 0.30, `filters.ghost_min_score` 0.20 and `filters.ghost_parked_score` 0.20, since RF-DETR scores parked and distant vehicles mostly between 0.15 and 0.5. SAM 3 and UrbanOmniDetect-2 use the defaults.

## Input size and score threshold

`DetectorConfig.imgsz` and `DetectorConfig.conf` default to `None`, which selects each back-end's own values:

| Back-end | Network input (`imgsz = None`) | `conf = None` |
|---|---|---|
| RF-DETR, RF-DETR-Seg | square, twice the training resolution and at most 1008 pixels, rounded up to the patch grid: `rfdetr-seg-l` 1008, `rfdetr-seg-m` 864, `rfdetr-seg-s` 768, `rfdetr-seg-n` 624, `rfdetr-l`, `-m`, `-s` 1024, `rfdetr-n` 768 | 0.15 |
| SAM 3 | 1008 × 1008, fixed (`imgsz` is not used) | 0.40 for the detection prompts; the evidence prompts use their own floors, 0.35 or 0.40 (class probability times presence probability) |
| UrbanOmniDetect-2 | letterboxed, long side 960 | 0.10 |

RF-DETR resizes each frame to a square without preserving the aspect ratio, as in its training; an explicit `imgsz` sets the side of that square and is rounded up to a multiple of the patch size times the number of attention windows (24 for `rfdetr-seg-s/m/l`, 12 for `rfdetr-seg-n`, 32 for the detection models). The released RF-DETR-Seg L runs at 504 pixels. On 24 frames each of the sample clips `intersection-day` and `motorway-overpass`, the share of confident UrbanOmniDetect-2 X detections (score at least 0.5) that RF-DETR-Seg L also found (box IoU at least 0.5, same super-class, score at least 0.3) was:

| RF-DETR-Seg L input | `intersection-day` | `motorway-overpass` | ms per frame (FP16, batch 8) |
|---|---|---|---|
| 504 (training resolution) | 0.48 | 0.64 | 3 to 5 |
| 768 | 0.60 | 0.68 | 6 to 7 |
| 1008 (default) | 0.60 | 0.72 | 8 to 10 |
| 1296 | 0.61 | 0.76 | 15 |

The gain comes from small, distant and densely parked vehicles. In an earlier run of the same comparison, an input of 576 × 1008 that keeps the 16:9 aspect ratio found fewer than the 1008 square.

Score scales differ between back-ends. On `intersection-day`, the median score of the vehicle detections kept at the default thresholds was 0.82 for SAM 3 (two-level prompts; 20,742 vehicle detections of the default run), and in an earlier comparison 0.30 for RF-DETR-Seg L, 0.35 for RF-DETR L and 0.67 for UrbanOmniDetect-2 X. The share scoring at least 0.5 was 0.95, 0.19, 0.25 and 0.66. RF-DETR scores parked and distant vehicles mostly between 0.15 and 0.5, hence its [score profile](#score-profiles). The tracking thresholds (`TrackerConfig.high_conf`, `FilterConfig.ghost_min_score`, `FilterConfig.ghost_parked_score`) refer to these scores.

## Class taxonomy

All back-ends map their labels into `ROAD_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")` (`config.py`) by name; labels outside it (traffic lights, aircraft, animals) are dropped, and `DetectorConfig.classes` restricts the classes that are kept.

* **RF-DETR** checkpoints use sparse COCO category ids; `COCO_CATEGORY_TO_ROAD = {1: person, 2: bicycle, 3: car, 4: motorcycle, 6: bus, 8: truck}`. Each of the model's queries yields at most one detection, its highest-scoring kept class, so one object never appears as both `car` and `truck` from the same query.
* **SAM 3** detects with the `vehicle` and `person` prompts and takes vehicle classes from the class-evidence prompts ([SAM 3 prompts](#sam-3-prompts)); `DetectorConfig.classes` removes evidence prompts of classes that are not kept (their evidence columns stay 0).
* **UrbanOmniDetect-2** checkpoints name their labels like the 80 COCO classes; `COCO_TO_ROAD = {0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck}` covers COCO-indexed heads without names.

Association and verdicts use three super-classes, `SUPER_CLASS`: `vehicle` (car, bus, truck), `two_wheeler` (bicycle, motorcycle) and `person`. Label flicker inside a super-class (car and truck, typically) therefore never splits a track. With SAM 3, vehicles and two-wheelers also associate with each other, since their per-frame classes are single-frame votes; the final class of a track is decided offline by a weighted vote ([Association and track hygiene](tracking.md#class-vote)).

## Instance masks

For RF-DETR-Seg and SAM 3, `DetectionSet.masks` (a `MaskStore`) holds one instance mask per detection, cropped to its box plus 2 pixels (`detections.crop_mask`) and bit-packed in `detections.npz`; `masks.crop(i)` returns `(x0, y0, mask)` and `masks.full(i, image_size)` the full-frame mask. Both detectors predict masks at a fraction of their input resolution (RF-DETR-Seg a quarter, SAM 3 288 × 288); `detect.masks.mask_crops` samples the bilinearly upsampled mask only inside each crop window, which gives the same pixels as upsampling to the full frame and cropping afterwards. RF-DETR-Seg masks are thresholded at logit 0, SAM 3 masks at probability 0.5.

## Cuboid corners

For UrbanOmniDetect-2, `DetectionSet.cuboid` holds an `(N, 8, 3)` array of `x, y, confidence` per corner in raw image pixels (`detections.FrameDetections`). Corners 0 to 3 touch the ground and corners 4 to 7 are the roof above them; corners 0 and 3 form the front face, and the edges 0-1 and 3-2 are the long sides. In `tracking/measure.py` the ground corners are back-projected to the road plane; the front is the midpoint of corners 0 and 3, the rear the midpoint of corners 1 and 2, the length is the mean of the 0-1 and 3-2 edges and the width the mean of the 0-3 and 1-2 edges. The other back-ends provide no cuboids, and the steps that use them are skipped.

## Per-frame post-processing

The built-in back-ends apply these steps to each image, in this order (`detect/postprocess.py`, `prefilter` and `refine`). Defaults are from `DetectorConfig`.

1. **Inference** at the input size above with the model threshold `conf`, in reduced precision on CUDA when `half = True` (FP16; SAM 3 in BF16 where the GPU supports it), `batch = 8` frames per call (`detect_video`).
2. **Class mapping and selection** into `ROAD_CLASSES`; labels outside `classes` are dropped. For SAM 3, this is the single-frame class from the evidence vectors (`assign_classes`).
3. **Class floors** `class_conf = {person: 0.40, bicycle: 0.50, motorcycle: 0.50}`: detections of these classes below their floor are dropped. Other classes keep everything above `conf`. For SAM 3 the floor follows the prompt that made the detection: a `vehicle` detection is kept at the vehicle threshold whatever class its evidence suggests, and only unclaimed two-wheelers are held to the two-wheeler floors.
4. **Width limit**: boxes wider than `max_box_width_frac` of the image width are dropped (default 1.01, i.e. disabled).
5. **Near-duplicate suppression** (`nms_by_group`): greedy NMS at box IoU above 0.8 (`DUPLICATE_IOU`) within each super-class, which removes double reports of one object with two labels.
6. **Mask-duplicate suppression** (`suppress_mask_duplicates`, RF-DETR-Seg and SAM 3): of two detections of one super-class whose masks overlap with IoU above 0.5 (`MASK_DUPLICATE_IOU`), the lower-scored one is dropped. This removes the second, lower-scored mask with a looser box that set-prediction detectors (SAM 3, RF-DETR-Seg) report for some vehicles.
7. **Cuboid trust** (`cuboid_trust`, UrbanOmniDetect-2): a cuboid is used only if all eight corners are finite, the weakest corner confidence is at least `cuboid_conf = 0.5`, all corners lie inside the box expanded by 0.75 box widths and heights on each side, and no corner collapsed to the image origin. Untrusted cuboids are replaced by NaN and never reach the geometry.
8. **Nested-vehicle suppression** (`suppress_nested`): a vehicle box that lies at least `nested_containment = 0.85` inside a larger vehicle box (intersection over its own area) is dropped when both stand on the same ground: their trusted ground quads (corners 0 to 3, in the image) overlap by at least `nested_ground_overlap = 0.30` of the smaller quad, or, without trusted cuboids, their bottom edges are within 15 % of the larger box height. This removes truck cabs and trailers reported next to the whole truck, and keeps a car standing in front of a bus, which is nested in the image but not on the ground. For RF-DETR and SAM 3 two conditions are added: the larger box must score at least as high as the nested one, because these detectors report low-scored boxes around rows of parked cars that would otherwise remove the cars; and, with masks, at least half of the nested detection's mask must lie on the larger detection's mask, so a vehicle that occludes the larger one is kept.

Detections with scores between `conf` and `TrackerConfig.high_conf = 0.50` are kept on purpose: they never start a track but extend confirmed tracks in the second association stage ([Association](tracking.md#online-association)).

## Frame rate and decoding

`detect_video` analyses one frame in `round(fps / rate_hz)` (`rate_hz = 15` by default; `0` analyses all frames). Frames are decoded sequentially (skipped frames are grabbed, not decoded) in a background thread with a queue of `2 × batch` frames, which overlaps decoding with inference. Processed frame indices are stored in `DetectionSet.processed_frames`, so frames without detections are still known to the tracker.

Detection speed on the sample clips at the default 15 Hz analysis rate (`detect_video`, batch 8, decoding included, model loading excluded) on one NVIDIA RTX PRO 6000 Blackwell Max-Q:

| Detector | `intersection-day` (1920 × 1080, 465 frames) | `motorway-overpass` (1280 × 720, 530 frames) | Model loading |
|---|---|---|---|
| `rfdetr-seg-l` (1008, FP16) | 99 frames/s | 110 frames/s | 3 s |
| `rfdetr-l` (1024, FP16) | 182 frames/s | 216 frames/s | 2 s |
| `sam3` (1008, BF16, 9 prompts) | 9.1 frames/s | 9.2 frames/s | 4 s |
| `uod2-x` (960, FP16) | 143 frames/s | 150 frames/s | 0.2 s |

The `sam3` row was measured with the two-level prompt scheme of this release (two detection and seven evidence prompts): 51.0 s for the 465 frames of `intersection-day` and 57.5 s for the 530 frames of `motorway-overpass`, with a peak of 7.3 GB of allocated GPU memory at batch 8. The other rows were measured with the same procedure before SAM 3 became the default.

## Detection without the pipeline

```python
from tachymetry.config import DetectorConfig
from tachymetry.data import sample_video
from tachymetry.detect import build_detector
from tachymetry.detect.run import detect_video
from tachymetry.io import VideoSource

detector = build_detector(DetectorConfig(), device="cuda:0")  # the default, sam3
with VideoSource(sample_video("intersection-day")) as src:
    dets = detect_video(src, detector, rate_hz=5, max_frames=20)
print(len(dets), dets.cuboid, dets.meta["stride"], dets.meta["rate_hz"])
print(dets.meta["detector"]["prompts"], dets.meta["detector"]["evidence_prompts"])
print(dets.meta["detector"]["imgsz"], dets.meta["detector"]["conf"])
x0, y0, mask = dets.masks.crop(0)
print(dets.class_names[dets.cls[0]], dets.xyxy[0].round(1), (x0, y0), mask.shape, mask.sum())
```

```text
932 None 6 5.0
['vehicle', 'person'] ['car', 'van', 'suv', 'bus', 'truck', 'bike', 'motorcycle']
[1008, 1008] 0.4
car [1784.7  241.1 1843.3  270.9] (1782, 239) (35, 65) 1300
```

`DetectionSet` stores detections in columnar form (`frame`, `time`, `xyxy`, `score`, `cls`, `cuboid`, `masks`), iterates per processed frame with `frames()`, and round-trips through `save()` and `load()`.

## Detectors without masks

Box-only detectors (`rfdetr-*`, `uod2-*` without masks, or a custom detector that returns no masks) lose two pieces of evidence:

* **Vehicle footprints** come from the box-contact model with class priors ([Association and track hygiene](tracking.md#centre-estimation)), or from trusted cuboid footprints with `uod2-*`, instead of the mask geometry.
* **The metric scale of automatic calibration.** Without masks there is no vehicle-silhouette scale. With `uod2-*` the cuboid footprints provide a vehicle-size scale; otherwise `Pipeline` stops with a `CalibrationError` when the result has no metric scale. Pass a calibrated camera (`camera="camera.json"`) or a scale cue such as `calibration.scale = "height=8.5"`.

The parked-vehicle ghost rule uses the absence of trusted footprints as evidence only when the detector provides cuboids ([Verdicts](verdicts.md#ghosts)).
