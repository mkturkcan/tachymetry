# Custom detectors

The pipeline accepts object detectors that map frames to boxes in the road-user taxonomy. The built-in back-ends (SAM 3, the default, RF-DETR and UrbanOmniDetect-2) are selected by name ([Detection](../guide/detection.md#detectors)); other models are wrapped in a subclass of `tachymetry.detect.Detector` and passed to `Pipeline(detector=...)`.

## The detector contract

A `Detector` is called with a list of BGR `uint8` images (`H × W × 3`, as decoded by OpenCV) and returns one tuple per image, with four elements, or five when it segments:

| Element | Shape and type | Meaning |
|---|---|---|
| `xyxy` | `(N, 4)` `float32` | boxes in pixels of the input image: `x1, y1, x2, y2` |
| `score` | `(N,)` `float32` | confidence in `[0, 1]` |
| `cls` | `(N,)` `int16` | index into `class_names` |
| `cuboid` | `(N, 8, 3)` `float32` or `None` | projected cuboid corners `x, y, confidence`: corners 0 to 3 on the ground with 0 and 3 forming the front face and 0-1 a long side, corners 4 to 7 above them; NaN for untrusted cuboids |
| `masks` (optional) | list of `N` entries, or `None` | one instance mask per detection as `(x0, y0, mask)` with a boolean `mask` cropped to the box (`tachymetry.detections.crop_mask` crops a full-frame mask), or `None` for a detection without one |

Set the class attributes `class_names` (use `tachymetry.config.ROAD_CLASSES` so that class priors, super-classes, colours and calibration apply), `provides_cuboids`, `provides_masks` and `name`. Boxes with scores down to `TrackerConfig.low_conf = 0.10` are useful: they extend existing tracks in the second association stage but never start one ([Association](../guide/tracking.md#online-association)).

The built-in post-processing of the back-ends (class floors, near-duplicate and mask-duplicate suppression, cuboid trust, nested-vehicle suppression) is not applied to custom detectors. The helpers are public in `tachymetry.detect.postprocess` (`prefilter` and `refine`, which run these steps with a `DetectorConfig`, or `nms_by_group`, `suppress_mask_duplicates`, `cuboid_trust` and `suppress_nested` individually) and can be called inside `__call__`.

## Example: torchvision Faster R-CNN

The wrapper below runs torchvision's COCO-trained Faster R-CNN (ResNet-50 FPN v2) and maps its labels to the road classes by name.

```python
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

import tachymetry as tm
from tachymetry.config import ROAD_CLASSES
from tachymetry.data import sample_video
from tachymetry.detect import Detector


class TorchvisionDetector(Detector):
    """COCO Faster R-CNN from torchvision, mapped to the road-user classes (boxes only)."""

    class_names = ROAD_CLASSES
    provides_cuboids = False
    name = "fasterrcnn_resnet50_fpn_v2"

    def __init__(self, device: str = "cuda:0", min_score: float = 0.10):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=min_score).eval().to(device)
        self.device = device
        # torchvision label id -> index into ROAD_CLASSES
        self.lut = {i: ROAD_CLASSES.index(c) for i, c in enumerate(weights.meta["categories"]) if c in ROAD_CLASSES}

    @torch.inference_mode()
    def __call__(self, images):
        batch = [torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1).to(self.device) / 255.0
                 for img in images]  # BGR uint8 -> RGB float in [0, 1]
        out = []
        for r in self.model(batch):
            labels = r["labels"].cpu().numpy()
            keep = np.isin(labels, list(self.lut))
            xyxy = r["boxes"].cpu().numpy()[keep].astype(np.float32)
            score = r["scores"].cpu().numpy()[keep].astype(np.float32)
            cls = np.array([self.lut[int(k)] for k in labels[keep]], dtype=np.int16)
            out.append((xyxy, score, cls, None))
        return out


cfg = tm.PipelineConfig()
cfg.detector.name = "custom:fasterrcnn_resnet50_fpn_v2"  # distinct detection-cache key
pipe = tm.Pipeline(cfg, camera="runs/intersection-day/camera.json", detector=TorchvisionDetector("cuda:0"))
run = pipe.run(sample_video("intersection-day"), "runs/intersection-day-frcnn")
print(run.summary())
print(run.report["timings"])
```

```text
{'tracks': 58, 'by_class': {'car': 37, 'person': 14, 'motorcycle': 1, 'truck': 3, 'bus': 2, 'bicycle': 1}, 'moving_by_class': {'car': 23, 'motorcycle': 1, 'person': 10, 'bus': 2, 'bicycle': 1}, 'parked_by_class': {'car': 14, 'truck': 3}, 'rejected': {'rider': 2, 'duplicate': 22, 'invalid': 1, 'ghost': 5}, 'vehicle_speed_kmh': {'median': 23.89, 'p85': 34.09, 'max': 50.17}}
{'detect_s': 9.14, 'calibrate_s': 6.24, 'track_s': 5.54, 'total_s': 21.2}
```

Three details of this example matter in general:

* **Camera.** A box-only detector provides no instance masks, so automatic calibration has no vehicle-silhouette scale ([Detectors without masks](../guide/detection.md#detectors-without-masks)). The example reuses the camera that the default pipeline calibrated in the [end-to-end example](intersection.md); `calibrate_s` is then only the background estimation. Alternatively give a metric cue, e.g. `cfg.calibration.scale = "height=27.2"`, or calibrate manually.
* **Cache key.** The detection cache key is computed from `cfg.detector` and `rate_hz`, not from the detector object. A distinct `detector.name`, or a fresh output directory, prevents the pipeline from reusing detections of another detector cached in the same directory.
* **Duplicates.** torchvision applies non-maximum suppression per label, so one vehicle often yields overlapping `car` and `truck` boxes. The tracker gates by super-class, so both boxes enter one association problem, and the duplicate verdict removed 22 tracks in this run ([Verdicts](../guide/verdicts.md#duplicates)). Cross-label suppression inside the detector (for example `nms_by_group`) removes them earlier.

Without masks, vehicle centres come from the box-contact model with class priors instead of the mask geometry; [Vehicle footprints from masks](../guide/footprints.md#synthetic-occlusion) compares the two paths on a synthetic occlusion. A detector that segments returns the optional fifth element, and its vehicles then take the mask path.

## Detection only

`detect_video` runs a detector over a video and returns a `DetectionSet`, which can be saved and inspected independently of tracking:

```python
from tachymetry.detect.run import detect_video
from tachymetry.io import VideoSource

with VideoSource(sample_video("intersection-day")) as src:
    dets = detect_video(src, TorchvisionDetector("cuda:0"), rate_hz=5, max_frames=20)
dets.save("frcnn_detections.npz")
print(len(dets), dets.class_names[int(dets.cls[0])], dets.xyxy[0])
```

Run after the class definition above, this printed

```text
1715 person [263.7007  633.77844 280.534   689.0219 ]
```

that is, 1,715 detections in the first 20 analysed frames (at the model's `box_score_thresh = 0.10`), the first of them a pedestrian.
