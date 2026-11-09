# Configuration reference

A run is configured by one `PipelineConfig` (`src/tachymetry/config.py`) made of nested dataclasses. Thresholds are expressed in seconds, metres, metres per second or fractions of the image, never in samples or pixels, so a configuration transfers between frame rates and resolutions.

## Setting values

```bash
tachymetry config                              # print the defaults as YAML
tachymetry config --dump defaults.yaml         # write them to a file
tachymetry run clip.mp4 --config my.yaml       # a partial YAML/JSON file overrides only the keys it contains
tachymetry run clip.mp4 --set tracker.max_lost_s=2.0 --set hygiene.stop_speed_kmh=2
```

```python
import tachymetry as tm
from tachymetry.config import PipelineConfig

cfg = PipelineConfig.load("my.yaml")           # partial files override defaults
cfg.override({"detector.imgsz": 1280, "render.units": "mph"})
cfg.save("runs/clip/config.yaml")
run = tm.track("clip.mp4", config=cfg, hygiene__max_gsd_m=0.25)   # keyword overrides use "__" for "."
```

Values given as strings are converted to the type of the default (`"true"`/`"false"` for booleans, comma-separated lists for tuples). Dictionaries such as `tracker.max_speed_mps` are merged key by key when loaded from a file. Unknown keys raise `KeyError`. Every run writes the configuration it used to `config.yaml` in its run directory.

The command-line flags `--detector`, `--rate`, `--imgsz`, `--layout`, `--units` and `--formats` are shortcuts for `detector.name`, `rate_hz`, `detector.imgsz`, `render.layout`, `render.units` and `exports`.

!!! note "Detection cache"
    Detections are cached in `detections.npz` under a key computed from `detector.{name, weights, imgsz, conf, class_conf, classes, cuboid_conf, nested_containment, nested_ground_overlap}` and `rate_hz` (`pipeline._detector_key`). Changing any other key re-runs only calibration (if automatic), tracking and export.

## All keys

The tables below are generated from the dataclasses in `src/tachymetry/config.py` when this documentation is built; descriptions are the comments next to each field. The rules that use each key are described in [Detection](detection.md), [Association and track hygiene](tracking.md), [Cross-track verdicts](verdicts.md), [Automatic calibration](auto-calibration.md) and [Rendering](rendering.md).

<!-- config-reference -->
