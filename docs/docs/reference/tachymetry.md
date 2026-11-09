# `tachymetry`

The top-level package exposes the three entry points used in most scripts, plus the run loader. `Camera` is documented in [`tachymetry.geometry`](geometry.md) and `PipelineConfig` in [`tachymetry.config`](config.md).

::: tachymetry
    options:
      members: [track, track_many, calibrate, load_run]
      show_root_heading: false

::: tachymetry.pipeline.Pipeline

::: tachymetry.pipeline.resolve_camera

::: tachymetry.data
    options:
      members: [sample_video, SAMPLES, SAMPLE_ATTRIBUTION]
