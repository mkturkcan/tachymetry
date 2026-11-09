from pathlib import Path

import pytest

from tachymetry.batch import output_dir_for, plan_workers
from tachymetry.batch.devices import Device
from tachymetry.cli import main
from tachymetry.config import PipelineConfig


def test_config_round_trip(tmp_path):
    cfg = PipelineConfig()
    cfg.override({"detector.imgsz": "1280", "rate_hz": "10", "hygiene.hold_still": "false"})
    p = cfg.save(tmp_path / "cfg.yaml")
    c2 = PipelineConfig.load(p)
    assert c2.detector.imgsz == 1280 and c2.rate_hz == 10.0 and c2.hygiene.hold_still is False


def test_partial_config_overrides_defaults(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("detector:\n  name: uod2-s\n")
    cfg = PipelineConfig.load(p)
    assert cfg.detector.name == "uod2-s" and cfg.detector.imgsz is None and cfg.detector.batch == 8


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        PipelineConfig().override({"detector.nope": 1})


def test_worker_plan():
    gpus = [Device(f"cuda:{i}", "cuda", "gpu", 80, 80) for i in range(4)]
    slots = plan_workers(gpus, "auto", n_jobs=100)
    assert len(slots) >= 4 and set(slots) == {f"cuda:{i}" for i in range(4)}
    assert plan_workers(gpus, 1, n_jobs=2) == ["cuda:0", "cuda:1"]
    # fewer jobs than slots: the jobs still go to different GPUs
    assert plan_workers(gpus[:2], "auto", n_jobs=2) == ["cuda:0", "cuda:1"]
    assert plan_workers([Device("cpu", "cpu", "cpu")], "auto", n_jobs=3)


def test_output_dirs_are_collision_free(tmp_path):
    a, b = Path(tmp_path / "site1" / "cam.mp4"), Path(tmp_path / "site2" / "cam.mp4")
    assert output_dir_for(a, Path("runs"), tmp_path) != output_dir_for(b, Path("runs"), tmp_path)


def test_cli_help_and_config(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert main(["config"]) == 0
    assert "detector" in capsys.readouterr().out


def test_detector_score_profiles():
    """RF-DETR reports lower confidences: its profile lowers the tracking thresholds left at their defaults."""
    cfg = PipelineConfig()
    cfg.detector.name = "rfdetr-seg-l"
    cfg.filters.ghost_min_score = 0.33  # set by the user: kept
    cfg.with_detector_profile()
    assert (
        cfg.tracker.high_conf == 0.30 and cfg.filters.ghost_parked_score == 0.20 and cfg.filters.ghost_min_score == 0.33
    )
    assert PipelineConfig().with_detector_profile().tracker.high_conf == 0.50  # SAM 3 keeps the defaults
