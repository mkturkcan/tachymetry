# Pipeline overview

`tachymetry.track()` and `tachymetry.Pipeline.run()` (`pipeline.py`) process one video through a fixed sequence of stages. The stages after detection are deterministic given their inputs. Detection, which needs the GPU, and automatic calibration are cached in the run directory, so tracking, filtering and rendering settings can be changed and re-run on the CPU alone.

<figure class="tm-diagram" style="width:100%;max-width:100%;margin:1.5em 0">
<svg viewBox="0 0 980 340" role="img" aria-labelledby="tm-pipe-title" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:inherit">
<title id="tm-pipe-title">Tachymetry pipeline: detection, background, calibration, ground measurement, association, track hygiene, stitching, verdicts, export and rendering</title>
<defs>
<marker id="tm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#9b9ca3"/></marker>
</defs>
<g fill="none" stroke="#9b9ca3" stroke-width="1.4">
<path d="M120 67H146" marker-end="url(#tm-arrow)"/>
<path d="M340 67H366" marker-end="url(#tm-arrow)"/>
<path d="M540 67H566" marker-end="url(#tm-arrow)"/>
<path d="M665 104V150H75V196" marker-end="url(#tm-arrow)"/>
<path d="M150 233H166" marker-end="url(#tm-arrow)"/>
<path d="M310 233H326" marker-end="url(#tm-arrow)"/>
<path d="M480 233H496" marker-end="url(#tm-arrow)"/>
<path d="M610 233H626" marker-end="url(#tm-arrow)"/>
<path d="M740 233H756" marker-end="url(#tm-arrow)"/>
<path d="M860 233H876" marker-end="url(#tm-arrow)"/>
</g>
<rect x="-4" y="178" width="752" height="112" rx="16" fill="none" stroke="#6fe3ff" stroke-opacity="0.55" stroke-dasharray="5 5"/>
<text x="6" y="308" font-size="12" fill="#6fe3ff">tracking.build_tracks: CPU only; with masks, about 1 s per vehicle track</text>
<g stroke="#ffb547" stroke-width="1.4" fill="#ffb547" fill-opacity="0.07">
<rect x="0" y="30" width="120" height="74" rx="10"/>
<rect x="150" y="30" width="190" height="74" rx="10"/>
<rect x="370" y="30" width="170" height="74" rx="10"/>
<rect x="570" y="30" width="190" height="74" rx="10"/>
</g>
<g stroke="#9b9ca3" stroke-width="1.2" fill="#9b9ca3" fill-opacity="0.06">
<rect x="0" y="196" width="150" height="74" rx="10"/>
<rect x="170" y="196" width="140" height="74" rx="10"/>
<rect x="330" y="196" width="150" height="74" rx="10"/>
<rect x="500" y="196" width="110" height="74" rx="10"/>
<rect x="630" y="196" width="110" height="74" rx="10"/>
<rect x="760" y="196" width="100" height="74" rx="10"/>
<rect x="880" y="196" width="100" height="74" rx="10" stroke-dasharray="4 4"/>
</g>
<g font-size="15" font-weight="600" fill="currentColor" text-anchor="middle">
<text x="60" y="62">Video</text>
<text x="245" y="62">Detection</text>
<text x="455" y="62">Background</text>
<text x="665" y="62">Calibration</text>
<text x="75" y="228">Ground</text>
<text x="240" y="228">Association</text>
<text x="405" y="228">Track hygiene</text>
<text x="555" y="228">Stitching</text>
<text x="685" y="228">Verdicts</text>
<text x="810" y="228">Export</text>
<text x="930" y="228">Render</text>
</g>
<g font-size="11.5" fill="#9b9ca3" text-anchor="middle">
<text x="60" y="82">file or images</text>
<text x="245" y="82">boxes and instance masks</text>
<text x="455" y="82">median of 41 frames</text>
<text x="665" y="82">auto, manual or given</text>
<text x="75" y="248">mask contact, GSD</text>
<text x="240" y="248">BYTE, metric gate</text>
<text x="405" y="248">mask boxes, smooth, referees</text>
<text x="555" y="248">fragments</text>
<text x="685" y="248">ghosts, riders</text>
<text x="810" y="248">CSV, JSON, ...</text>
<text x="930" y="248">overlay.mp4</text>
</g>
<g font-size="11" fill="#ffb547" text-anchor="middle" font-family="Geist Mono, ui-monospace, monospace">
<text x="245" y="122">detections.npz</text>
<text x="455" y="122">background.jpg</text>
<text x="665" y="122">camera.json</text>
</g>
</svg>
<figcaption>Stages of <code>Pipeline.run</code>. Amber: stages whose results are cached in the run directory (file names in amber). Grey: stages recomputed on each run. Rendering is optional.</figcaption>
</figure>

## Stages

| Stage | Implementation | Result | Cache file |
|---|---|---|---|
| Detection | `detect.backends.build_detector`, `detect.run.detect_video` | `DetectionSet`: boxes, scores, classes and, for segmenting detectors (`sam3`, the default, and `rfdetr-seg-*`), instance masks; cuboid corners for `uod2-*` | `detections.npz` (masks bit-packed), `detections.json` |
| Background | `calib.lines.background_image` (temporal median of `calibration.background_frames = 41` frames spread evenly over the video) | traffic-free image | `background.jpg` |
| Calibration | `pipeline.resolve_camera`, `calib.auto.calibrate_auto` (orientation from vanishing points or a known field of view, metric scale from vehicle silhouettes) | `Camera` (intrinsics, pose, lens) | `camera.json`, `camera.report.json`, `calibration_preview.jpg` |
| Ground measurement | `tracking.measure.measure_ground` | contact point (the mask bottom-centre when a mask exists), cuboid footprint, ground sampling distance per detection | none |
| Association | `tracking.builder.associate`, `tracking.associate.ByteTracker` | track id per detection | none |
| Track hygiene | `tracking.hygiene.process_raw_track`; for vehicles with masks `tracking.maskgeom.vehicle_geometry` ([Vehicle footprints from masks](footprints.md)) | `Track` objects in metres | none |
| Stitching | `tracking.verdicts.stitch_candidates`, then hygiene again on the joined rows | merged fragments | none |
| Verdicts | `tracking.verdicts.mark_duplicates`, `mark_passengers_and_riders`, `mark_shadows`, `mark_ghosts` | `verdict` and `reason` per track | none |
| Export | `results.TrackSet.export` | `tracks.csv`, `tracks_summary.csv`, `tracks.json` (default `exports = ("csv", "json")`) | |
| Render | `render.video.render_video` | `overlay.mp4` | |

The stages are documented in detail in [Detection](detection.md), [Calibration](calibration.md), [Association and track hygiene](tracking.md), [Vehicle footprints from masks](footprints.md), [Verdicts](verdicts.md), [Bird's-eye view and speed](bev-speed.md), [Outputs](outputs.md) and [Rendering](rendering.md).

## Video input

`io.VideoSource` reads a video file or a folder of images, sorted in natural order. Image folders are read at 10 fps unless a rate is given: `VideoSource(path, fps=...)`, or `PipelineConfig.fps` (`--set fps=12.5`), which overrides the input frame rate of videos as well.

**Backends.** Video files are opened with OpenCV first. A file is passed to the FFmpeg binary that `imageio-ffmpeg` installs (`backend = "ffmpeg"`) in two cases:

- OpenCV opens it but cannot decode a frame, for example AV1 in OpenCV builds without that decoder;
- OpenCV reports a frame rate outside 1 to 1000 fps.

`VideoSource(path).backend` reports the reader in use (`opencv`, `ffmpeg` or `images`), and `backend="opencv"` or `backend="ffmpeg"` forces one.

**File discovery.** Batch inputs are found by extension; `.ogv` and `.ogg` are recognised alongside `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` and the other common containers (`io.video.VIDEO_EXTENSIONS`).

```python
from tachymetry.data import sample_video
from tachymetry.io import VideoSource

with VideoSource(sample_video("motorway-overpass")) as src:
    print(src.backend, src.size, round(src.fps, 3), src.frame_count)
```

```text
opencv (1280, 720) 25.0 1060
```

**Frame times.** Frame times are `index / fps`, with `fps` taken from the container; the timestamps that OpenCV reports are not used. The FFmpeg reader seeks by time and estimates the frame count from the container duration.

## Caching and re-runs

The run directory doubles as a cache. `Pipeline.run` applies the following rules (`pipeline.py`):

* **Detections** are reused when `detections.npz` and `detections.json` exist and the key stored in `detections.json` equals `_detector_key(config)`: the first 12 hexadecimal digits of the SHA-1 of the detector name, weights, `imgsz`, `conf`, `class_conf`, `classes`, `cuboid_conf`, the two nested-suppression thresholds and `rate_hz`. Changing one of these re-runs detection; changing tracker, hygiene, filter, export or render settings does not. Batch size, half precision and device are not part of the key. `keep_detections = False` disables writing the cache.
* **Background** is reused when `background.jpg` exists and has the video's resolution.
* **Camera**: an explicit camera (a `Camera`, a camera JSON, a correspondence file, or a directory of per-video cameras) is used as given, resized with `Camera.resized` if the video has a different resolution, and written to `camera.json`. With `camera="auto"`, an existing `camera.json` with the video's resolution whose `meta.calibration_method` starts with `auto` is reused; otherwise automatic calibration runs and writes `camera.json`, `camera.report.json` and `calibration_preview.jpg`.

`Pipeline.run(..., use_cache=False)`, which `tachymetry run --overwrite` uses, ignores all three caches and rewrites them.

A detector passed as an object (`Pipeline(detector=...)`, see [Custom detectors](../examples/custom-detector.md)) is not part of the cache key. Give such runs their own output directory or a distinct `detector.name`.

Re-running a finished directory with different tracking settings only repeats the CPU stages:

```python
import tachymetry as tm
from tachymetry.data import sample_video

pipe = tm.Pipeline(device="cuda:0")
pipe.config.hygiene.smooth_window_s = 0.8
run = pipe.run(sample_video("intersection-day"), "runs/intersection-day")
print(run.report["timings"])
```

On the public sample clip, after a first full run into the same directory, this printed

```text
{'detect_s': 0.03, 'calibrate_s': 0.01, 'track_s': 41.26, 'total_s': 41.51}
```

With instance masks, the tracking time is dominated by the mask geometry, a median of 0.78 s of CPU per vehicle track on this clip ([Runtime](footprints.md#runtime)).

## Measured stage times

One run of `tm.track(sample_video("intersection-day"), out=..., render=True, calibration__hfov_deg=47.5)` with the default configuration otherwise, on one NVIDIA RTX PRO 6000 Blackwell Max-Q and the CPU of that machine (clip: 1920×1080, 30 fps, 30.97 s; 465 frames analysed at `rate_hz = 15`; the [end-to-end example](../examples/intersection.md)):

| Stage | Seconds |
|---|---|
| Detection (SAM 3, 1008 × 1008, BF16, batch 8, model loading included) | 55.88 |
| Background and automatic calibration | 23.29 |
| Ground measurement, association, hygiene (incl. the mask geometry of about 40 vehicle tracks), stitching, verdicts, export | 41.34 |
| Rendering (`side` layout, 929 frames, H.264) | 24.50 |
| Total | 145.22 |

The values are the `timings` block of `summary.json`. Detection time depends on the GPU and the back-end ([Detection](detection.md#frame-rate-and-decoding)). Calibration time is dominated by line-segment detection and vanishing-point estimation on the background image and by the silhouette-scale search over car masks. Tracking time grows with the number of vehicle tracks that take the mask path ([Runtime](footprints.md#runtime)); with `hygiene.mask_geometry = False`, `build_tracks` took 3.8 s on a run of the same clip.

## Units and rates

All thresholds are in seconds, metres, metres per second or fractions of the image (`config.py`), so a configuration behaves the same across frame rates and resolutions. `rate_hz` (default 15) selects the analysed frames: the frame stride is `round(fps / rate_hz)` (`VideoSource.stride_for`), so a 30 fps video is analysed at one frame in two and a 25 fps video at `round(25 / 15) = 2`, i.e. 12.5 Hz. `rate_hz = 0` analyses all frames. Frame times are `index / fps` from the container frame rate (`io/video.py`), which assumes constant-frame-rate footage.

## Reloading a run

`tm.load_run(run_dir)` rebuilds a `RunResult` from `config.yaml`, `summary.json` and `camera.json`. The tracks are recomputed from the cached detections, so this takes seconds and needs no GPU; nothing in the run directory is modified:

```python
import tachymetry as tm

run = tm.load_run("runs/intersection-day")
print(len(run.tracks), run.camera.describe())
```

```text
50 f=2181.8px (hfov 47.5 deg), height=27.21 m, pitch=17.88 deg, roll=-0.01 deg, yaw=134.0 deg, lens=identity
```
