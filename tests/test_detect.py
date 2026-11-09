"""Detector back-ends with stub model outputs: no network, no GPU."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from tachymetry.config import ROAD_CLASSES, DetectorConfig  # noqa: E402
from tachymetry.detect import build_detector, detector_names  # noqa: E402
from tachymetry.detect.base import COCO_CATEGORY_TO_ROAD, road_class_lut  # noqa: E402
from tachymetry.detect.masks import mask_crops  # noqa: E402
from tachymetry.detect.postprocess import suppress_mask_duplicates, suppress_nested  # noqa: E402
from tachymetry.detections import crop_mask  # noqa: E402

CAR, TRUCK, PERSON, MOTO = (ROAD_CLASSES.index(c) for c in ("car", "truck", "person", "motorcycle"))

#: A slice of the sparse COCO category layout of the released RF-DETR checkpoints.
COCO_LABELS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 8: "truck"}
COCO_LABELS[10] = "traffic light"


def _blob(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_class_mapping_keeps_road_users_only():
    lut = road_class_lut(COCO_LABELS, ROAD_CLASSES)
    assert lut == {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5}  # traffic lights and aircraft are dropped
    assert {k: ROAD_CLASSES[v] for k, v in lut.items()} == COCO_CATEGORY_TO_ROAD
    assert road_class_lut(COCO_LABELS, ("car", "bus")) == {3: 2, 6: 4}


def test_mask_crops_match_full_upsampling():
    """Sampling only the crop windows gives the crops of the full-frame upsampled masks."""
    rng = np.random.default_rng(1)
    h, w, n = 90, 160, 12
    maps = torch.from_numpy(rng.normal(size=(n, 23, 41)).astype(np.float32))
    maps[:, 5:15, 10:30] += 3.0
    x0, y0 = rng.uniform(-10, w - 5, n), rng.uniform(-10, h - 5, n)
    xyxy = np.stack([x0, y0, x0 + rng.uniform(0.5, 70, n), y0 + rng.uniform(0.5, 40, n)], 1).astype(np.float32)
    xyxy[0] = [w + 5, 10, w + 20, 20]  # outside the image
    full = F.interpolate(maps[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0] > 0
    for max_pixels in (1 << 23, 300):  # one chunk, then many
        crops = mask_crops(maps, xyxy, (w, h), threshold=0.0, max_pixels=max_pixels)
        for k in range(n):
            ref = crop_mask(full[k].numpy(), xyxy[k])
            assert (ref is None) == (crops[k] is None)
            if ref is not None:
                assert ref[:2] == crops[k][:2] and np.array_equal(ref[2], crops[k][2])
    assert crops[0] is None and sum(c is not None for c in crops) > n // 2


def test_mask_duplicates_keep_the_higher_score():
    car = (10, 10, _blob(30, 40, 0, 0, 40, 30))
    truck = (11, 10, _blob(31, 40, 0, 1, 40, 31))  # the same vehicle from another prompt
    rider = (30, 5, _blob(30, 12, 0, 0, 12, 30))
    xyxy = np.array([[10, 10, 50, 40], [11, 10, 51, 41], [30, 5, 42, 35]], np.float32)
    keep = suppress_mask_duplicates(xyxy, np.array([0.6, 0.9, 0.8]), [car, truck, rider], iou_thresh=0.7)
    assert keep.tolist() == [False, True, True]
    # with groups, only detections of one group are compared
    moto = (30, 5, _blob(30, 12, 0, 0, 12, 30))
    xy2 = np.array([[30, 5, 42, 35], [30, 5, 42, 35]], np.float32)
    assert suppress_mask_duplicates(xy2, np.array([0.9, 0.5]), [rider, moto], np.array([0, 1])).all()
    assert not suppress_mask_duplicates(xy2, np.array([0.9, 0.5]), [rider, moto], None).all()


def test_nested_suppression_with_scores_and_masks():
    veh = {CAR, TRUCK}
    # two parked cars inside a low-scored box around both: the cars stay when scores are used
    xyxy = np.array([[0, 0, 200, 100], [5, 40, 95, 100], [105, 42, 195, 100]], np.float32)
    cls = np.array([CAR, CAR, CAR])
    score = np.array([0.15, 0.6, 0.5])
    assert suppress_nested(xyxy, cls, veh, None, None).tolist() == [True, False, False]
    assert suppress_nested(xyxy, cls, veh, None, None, score=score).tolist() == [True, True, True]
    # truck cab on the truck mask is dropped; a car in front of the truck (not on its mask) stays
    truck = (0, 0, _blob(100, 200, 0, 0, 200, 100) & ~_blob(100, 200, 120, 50, 190, 100))
    cab = (10, 30, _blob(70, 60, 0, 0, 60, 70))
    car = (120, 50, _blob(50, 70, 0, 0, 70, 50))
    xyxy = np.array([[0, 0, 200, 100], [10, 30, 70, 100], [120, 50, 190, 100]], np.float32)
    cls = np.array([TRUCK, TRUCK, CAR])
    score = np.array([0.9, 0.7, 0.8])
    assert suppress_nested(xyxy, cls, veh, None, None, score=score).tolist() == [True, False, False]
    kept = suppress_nested(xyxy, cls, veh, None, None, score=score, masks=[truck, cab, car])
    assert kept.tolist() == [True, False, True]


def _rfdetr_stub(cfg, logits, boxes, masks):
    from tachymetry.detect.rfdetr import RFDETRDetector

    det = RFDETRDetector.__new__(RFDETRDetector)
    lut = road_class_lut(COCO_LABELS, cfg.classes)
    det.cfg, det.conf, det.provides_masks = cfg, 0.15, masks is not None
    det._cols = sorted(lut)
    det._road = np.array([lut[c] for c in det._cols], np.int16)
    det.class_names = ROAD_CLASSES
    out = (torch.tensor(boxes, dtype=torch.float32), torch.tensor(logits, dtype=torch.float32))
    if masks is not None:
        out = (*out, torch.tensor(masks, dtype=torch.float32))
    det._forward = lambda images: out
    return det


def test_rfdetr_selection_mapping_and_masks():
    """One detection per query (best road class); non-road queries and low scores are dropped."""
    q, slots = 5, 11
    logits = np.full((1, q, slots), -9.0)
    logits[0, 0, 3], logits[0, 0, 8] = 2.0, 0.5  # car (and a weaker truck score on the same query)
    logits[0, 1, 1] = 1.5  # person
    logits[0, 2, 10] = 4.0  # traffic light only
    logits[0, 3, 6] = -1.0  # bus at 0.27
    logits[0, 4, 4] = -3.0  # motorcycle below the threshold
    boxes = np.array([[[0.25, 0.5, 0.2, 0.4], [0.7, 0.5, 0.05, 0.3], [0.5, 0.1, 0.02, 0.05], [0.6, 0.6, 0.3, 0.3]]])
    boxes = np.concatenate([boxes, [[[0.1, 0.1, 0.1, 0.1]]]], axis=1)
    masks = np.full((1, q, 20, 32), -5.0)
    masks[0, 0, 6:14, 4:12] = 5.0
    masks[0, 3, 9:15, 15:24] = 5.0
    det = _rfdetr_stub(DetectorConfig(class_conf={}), logits, boxes, masks)
    img = np.zeros((80, 128, 3), np.uint8)
    xyxy, score, cls, cub, crops = det([img])[0]
    assert cub is None and len(crops) == len(score) == 3
    order = np.argsort(-score)
    assert [ROAD_CLASSES[c] for c in cls[order]] == ["car", "person", "bus"]
    car = order[0]
    np.testing.assert_allclose(xyxy[car], [19.2, 24.0, 44.8, 56.0], atol=1e-4)
    assert crops[car] is not None and crops[car][2].any() and crops[cls.tolist().index(PERSON)] is None
    # restricting the classes removes the others before scoring
    det = _rfdetr_stub(DetectorConfig(classes=("bus",), class_conf={}), logits, boxes, masks)
    assert [ROAD_CLASSES[c] for c in det([img])[0][2]] == ["bus"]


class _Sam3Stub:
    """Stand-in for ``Sam3Model``: prompt k (``text_embeds`` value k) finds the objects listed for it.

    Each object is ``(score, box)`` or ``(score, box, mask_box)``, in normalised image coordinates.
    """

    def __init__(self, found, q=4, hw=(16, 16)):
        self.found, self.q, self.hw = found, q, hw

    def get_vision_features(self, pixel_values):
        b = pixel_values.shape[0]
        return SimpleNamespace(
            fpn_hidden_states=(torch.zeros(b, 1, 2, 2),), fpn_position_encoding=(torch.zeros(b, 1, 2, 2),)
        )

    def __call__(self, vision_embeds, text_embeds, attention_mask):
        n = text_embeds.shape[0]
        logits = torch.full((n, self.q), -9.0)
        boxes = torch.zeros(n, self.q, 4)
        masks = torch.full((n, self.q, *self.hw), -9.0)
        for k in range(n):
            for j, (score, box, *mask_box) in enumerate(self.found.get(int(text_embeds[k, 0, 0]), [])):
                logits[k, j] = float(np.log(score / (1 - score)))
                boxes[k, j] = torch.tensor(box)
                mb = np.array(mask_box[0] if mask_box else box) * [self.hw[1], self.hw[0], self.hw[1], self.hw[0]]
                x0, y0, x1, y1 = mb.round().astype(int)
                masks[k, j, y0:y1, x0:x1] = 9.0
        return SimpleNamespace(
            pred_logits=logits, presence_logits=torch.full((n, 1), 9.0), pred_boxes=boxes, pred_masks=masks
        )


def _sam3_stub(found, classes=ROAD_CLASSES):
    from tachymetry.detect.sam3 import SAM3_PROMPTS, Sam3Detector

    det = Sam3Detector.__new__(Sam3Detector)
    det.cfg = DetectorConfig(name="sam3", class_conf={})
    det.conf, det.half, det.dtype, det.device = 0.3, False, None, torch.device("cpu")
    det.imgsz = (32, 32)
    det._mean = torch.full((1, 3, 1, 1), 0.5)
    det._std = torch.full((1, 3, 1, 1), 0.5)
    det.classes = [c for c in ROAD_CLASSES if c in classes]
    det.prompts = [SAM3_PROMPTS[c] for c in det.classes]
    det._cls = np.array([ROAD_CLASSES.index(c) for c in det.classes], np.int16)
    det._text = torch.arange(len(det.prompts), dtype=torch.float32).view(-1, 1, 1)
    det._text_mask = torch.ones(len(det.prompts), 1, dtype=torch.long)
    det.class_names = ROAD_CLASSES
    det.model = _Sam3Stub({det.classes.index(c): v for c, v in found.items()})
    return det


def test_sam3_cross_prompt_duplicates_are_merged():
    same = [0.25, 0.25, 0.75, 0.75]
    det = _sam3_stub(
        {
            "car": [(0.9, same)],
            # the car again (a looser box, box IoU 0.69, but the same mask), and an object below the threshold
            "truck": [(0.6, [0.2, 0.2, 0.8, 0.8], same), (0.2, [0.0, 0.0, 0.1, 0.1])],
            "person": [(0.8, [0.0, 0.5, 0.125, 1.0])],
        }
    )
    xyxy, score, cls, cub, crops = det([np.zeros((64, 96, 3), np.uint8)] * 2)[1]
    assert cub is None and sorted(ROAD_CLASSES[c] for c in cls) == ["car", "person"]
    car = cls.tolist().index(CAR)
    np.testing.assert_allclose(score[car], 0.9, atol=1e-3)  # class score times presence score
    np.testing.assert_allclose(xyxy[car], [24, 16, 72, 48], atol=1e-3)
    x0, y0, m = crops[car]
    assert m.sum() == pytest.approx(48 * 32, rel=0.15) and x0 <= 24 and y0 <= 16


def test_sam3_gated_repository_error(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Sam3Processor"):
        pytest.skip("transformers without SAM 3")
    from tachymetry.detect.sam3 import load_sam3

    def gated(*args, **kwargs):
        raise OSError("You are trying to access a gated repo.\nMake sure to have access to it at https://hf.co/x")

    monkeypatch.setattr(transformers.Sam3Processor, "from_pretrained", gated)
    with pytest.raises(PermissionError, match="hf auth login") as err:
        load_sam3("facebook/sam3")
    assert "https://huggingface.co/facebook/sam3" in str(err.value)


def test_detector_names_and_removed_backends(monkeypatch):
    names = detector_names()
    assert names[0] == "rfdetr-seg-l" and "sam3" in names and "uod2-x" in names
    assert not any("xlarge" in n or n.endswith("-xl") for n in names)
    assert DetectorConfig().name == "rfdetr-seg-l"
    for bad in ("yolo:yolo11x.pt", "yolo11x.pt", "rfdetr-seg-xl"):
        with pytest.raises(ValueError, match="unknown detector"):
            build_detector(DetectorConfig(name=bad))
    import sys

    from tachymetry.detect import uod2

    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ImportError, match=r"tachymetry\[uod2\]"):
        uod2._ultralytics()


def test_detect_video_stores_masks(tmp_path):
    import cv2

    from tachymetry.detect import Detector
    from tachymetry.detect.run import detect_video
    from tachymetry.io import VideoSource

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.zeros((40, 60, 3), np.uint8))

    class Boxes(Detector):
        def __call__(self, images):
            return [
                (np.array([[1, 1, 9, 9]], np.float32), np.array([0.9], np.float32), np.array([2], np.int16), None)
                for _ in images
            ]

    class Masks(Boxes):
        provides_masks = True

        def __call__(self, images):
            crop = (1, 1, _blob(9, 9, 1, 1, 8, 8))
            return [(*r, [crop]) for r in super().__call__(images)]

    with VideoSource(tmp_path) as src:
        assert detect_video(src, Boxes(), rate_hz=0).masks is None
    with VideoSource(tmp_path) as src:
        ds = detect_video(src, Masks(), rate_hz=0)
    assert ds.masks is not None and len(ds.masks) == 3 and ds.masks.has(2)
    assert ds.masks.full(0, (60, 40))[2:8, 2:8].all()


def test_prefetch_joins_its_worker_when_the_consumer_stops():
    """A consumer that stops early (max_frames) must not leave the decoder running while the source closes."""
    import threading
    import time

    from tachymetry.io.video import prefetch

    workers = []

    def frames():
        workers.append(threading.current_thread())
        for i in range(10_000):
            time.sleep(0.001)
            yield i

    got = []
    for x in prefetch(frames(), depth=4):
        got.append(x)
        if len(got) == 3:
            break
    assert got == [0, 1, 2] and workers and not workers[0].is_alive()
