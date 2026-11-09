"""Command-line interface: ``tachymetry <command>``.

Commands
--------
run         process videos (files, folders, globs) in parallel on the available GPUs
calibrate   auto | lines | manual | pick | preview | depth | geocalib
render      annotated video of a finished run
devices     list compute devices and the worker plan
demo        download a public sample clip and run the full pipeline
download    fetch detector weights ahead of time
config      print or save the default configuration
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__

__all__ = ["main"]


def _console() -> Any:
    try:
        from rich.console import Console

        return Console(highlight=False)
    except Exception:  # pragma: no cover
        return None


def _print(msg: str) -> None:
    c = _console()
    if c is not None:
        c.print(msg)
    else:
        print(msg)


def _parse_sets(items: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def _config(args: argparse.Namespace) -> Any:
    from .config import PipelineConfig

    cfg = PipelineConfig.load(args.config) if getattr(args, "config", None) else PipelineConfig()
    sets = _parse_sets(getattr(args, "set", None))
    for flag, key in (
        ("detector", "detector.name"),
        ("rate", "rate_hz"),
        ("imgsz", "detector.imgsz"),
        ("layout", "render.layout"),
        ("units", "render.units"),
    ):
        val = getattr(args, flag, None)
        if val is not None:
            sets[key] = val
    if getattr(args, "formats", None):
        sets["exports"] = tuple(args.formats.split(","))
    return cfg.override(sets) if sets else cfg


def _summary_table(report: Any) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:  # pragma: no cover
        for j in report.jobs:
            print(j["status"], j["video"])
        return
    t = Table(
        title=f"Tachymetry: {len(report.ok)} of {len(report.jobs)} videos processed in {report.seconds:.0f} s",
        title_justify="left",
        header_style="bold",
    )
    for col in ("video", "status", "tracks", "moving vehicles", "parked", "pedestrians", "median km/h", "seconds"):
        t.add_column(col, justify="right" if col not in ("video", "status") else "left")
    for j in report.jobs:
        s = j.get("summary") or {}
        mv = s.get("moving_by_class", {})
        pk = s.get("parked_by_class", {})
        t.add_row(
            Path(j["video"]).name,
            j["status"],
            str(s.get("tracks", "")),
            str(sum(v for k, v in mv.items() if k in ("car", "bus", "truck"))),
            str(sum(pk.values())) if pk else "0",
            str(mv.get("person", 0)),
            str((s.get("vehicle_speed_kmh") or {}).get("median", "")),
            str(j.get("seconds", "")),
        )
    Console().print(t)
    for j in report.failed:
        _print(f"[red]failed[/red] {j['video']}\n{j.get('error', '').strip().splitlines()[-1]}")


# ---------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace) -> int:
    from .batch import run_batch

    cfg = _config(args)
    report = run_batch(
        args.inputs,
        args.out,
        camera=args.camera,
        config=cfg,
        devices=args.devices,
        workers_per_device=args.workers_per_device,
        render=args.render,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    if not args.quiet:
        _summary_table(report)
        _print(f"Outputs: [bold]{Path(args.out).resolve()}[/bold]")
    return 1 if report.failed else 0


def _write_preview(img: Any, cam: Any, path: str | None, title: str) -> None:
    if not path:
        return
    import cv2

    from .calib.preview import preview

    cv2.imwrite(path, preview(img, cam, title=title))
    _print(f"Preview: {path}")


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calib import _load_image_or_background, calibrate_source

    if args.method == "pick":
        from .calib.picker import launch_picker

        img, _ = _load_image_or_background(args.source)
        out = launch_picker(img, reference=args.reference, gsd=args.gsd, out=args.out or "points.yaml")
        _print(f"Correspondences saved to {out}. Next: tachymetry calibrate manual {args.source} --points {out}")
        return 0
    if args.method == "preview":
        from .geometry import Camera

        img, _ = _load_image_or_background(args.source)
        cam = Camera.load(args.camera)
        _write_preview(
            img,
            cam if cam.image_size == (img.shape[1], img.shape[0]) else cam.resized((img.shape[1], img.shape[0])),
            args.out or "calibration_preview.jpg",
            f"Calibration: {args.camera}",
        )
        return 0
    kwargs: dict[str, Any] = {}
    if args.method in ("auto", "cuboids"):
        kwargs.update(detector=args.detector, device=args.device)
    res = calibrate_source(
        args.source, method=args.method, points=args.points, scale=args.scale, lens=args.lens, **kwargs
    )
    out = args.out or "camera.json"
    res.details.pop("candidate_cameras", None)
    res.save(out)
    _print(f"[bold]{res.summary()}[/bold]\nCamera: {out}")
    img, _ = _load_image_or_background(args.source)
    _write_preview(
        img, res.camera, args.preview or str(Path(out).with_suffix("")) + "_preview.jpg", f"Calibration: {res.method}"
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from . import load_run
    from .config import RenderConfig

    run = load_run(args.run_dir)
    cfg = RenderConfig(layout=args.layout or "side", units=args.units or "km/h", show_ids=args.ids)
    path = run.render(
        args.out or Path(args.run_dir) / f"overlay_{cfg.layout}.mp4",
        layout=cfg.layout,
        cfg=cfg,
        start_s=args.start,
        end_s=args.end,
    )
    _print(f"Video: {path}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    from .batch import discover_devices, plan_workers

    devs = discover_devices()
    for d in devs:
        _print(f"  {d}")
    _print(f"Worker plan ({args.workers_per_device} per device): {plan_workers(devs, args.workers_per_device)}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from . import track
    from .data import SAMPLE_ATTRIBUTION, sample_video

    path = sample_video(args.sample)
    _print(f"Sample clip: {path}\n{SAMPLE_ATTRIBUTION}")
    run = track(path, out=args.out, render=True, device=args.device)
    s = run.summary()
    _print(f"[bold]{s['tracks']} tracks[/bold] {s['by_class']}; outputs in {Path(args.out).resolve()}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .detect.backends import uod2_checkpoint

    name = args.detector.lower()
    if not name.startswith("uod2"):
        _print("Only UrbanOmniDetect-2 weights are downloaded by Tachymetry; Ultralytics fetches its own.")
        return 0
    size = name.split("-")[-1] if "-" in name else "x"
    _print(f"{uod2_checkpoint(size)}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    from .config import PipelineConfig

    cfg = PipelineConfig()
    if args.dump:
        cfg.save(args.dump)
        _print(f"Default configuration written to {args.dump}")
    else:
        import yaml

        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return 0


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tachymetry", description="Metric trajectories, speeds and counts from traffic-camera video."
    )
    p.add_argument("--version", action="version", version=f"tachymetry {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process videos in parallel on the available GPUs")
    r.add_argument("inputs", nargs="+", help="video files, folders or glob patterns")
    r.add_argument("-o", "--out", default="runs", help="output root (default: runs)")
    r.add_argument("--camera", default="auto", help="auto | camera.json | correspondences.yaml | folder of cameras")
    r.add_argument("--config", help="YAML/JSON configuration file")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any setting, e.g. detector.imgsz=1280")
    r.add_argument("--detector", help="uod2-{n,s,m,l,x} or yolo:<weights>")
    r.add_argument("--rate", type=float, help="frames analysed per second (default 15)")
    r.add_argument("--imgsz", type=int, help="detector input size (default 960)")
    r.add_argument("--devices", default="auto", help="auto | cpu | cuda | 0,1 | cuda:0,cuda:2")
    r.add_argument("--workers-per-device", default="auto", help="auto or an integer")
    r.add_argument("--render", action="store_true", help="also write an annotated overlay.mp4 per video")
    r.add_argument("--layout", choices=("side", "dashboard", "camera", "bev"), help="render layout")
    r.add_argument("--units", choices=("km/h", "mph", "m/s"), help="speed units in renders")
    r.add_argument("--formats", help="exports, comma separated: csv,json,geojson,mot,parquet")
    r.add_argument("--overwrite", action="store_true", help="re-process videos that already have results")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", help="calibrate a camera")
    c.add_argument("method", choices=("auto", "lines", "cuboids", "manual", "pick", "preview", "depth", "geocalib"))
    c.add_argument("source", help="video or image")
    c.add_argument("-o", "--out", help="output camera JSON (or points YAML for pick)")
    c.add_argument("--points", help="correspondence file (manual)")
    c.add_argument("--camera", help="camera JSON (preview)")
    c.add_argument("--scale", help="metric cue: vehicles | height=<m> | lane_width=<m>")
    c.add_argument("--lens", help="auto | identity | division | equidistant | brown")
    c.add_argument("--reference", help="orthophoto or site plan (pick)")
    c.add_argument("--gsd", type=float, help="metres per pixel of the reference image (pick)")
    c.add_argument("--preview", help="preview image path")
    c.add_argument("--detector", default="uod2-x")
    c.add_argument("--device", default="auto")
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("render", help="annotated video of a finished run")
    v.add_argument("run_dir")
    v.add_argument("-o", "--out")
    v.add_argument("--layout", choices=("side", "dashboard", "camera", "bev"))
    v.add_argument("--units", choices=("km/h", "mph", "m/s"))
    v.add_argument("--ids", action="store_true", help="show track ids")
    v.add_argument("--start", type=float, default=0.0)
    v.add_argument("--end", type=float)
    v.set_defaults(func=cmd_render)

    d = sub.add_parser("devices", help="list devices and the worker plan")
    d.add_argument("--workers-per-device", default="auto")
    d.set_defaults(func=cmd_devices)

    m = sub.add_parser("demo", help="run the pipeline on a public sample clip")
    m.add_argument("-o", "--out", default="demo_run")
    m.add_argument("--sample", default="intersection-day")
    m.add_argument("--device", default=None)
    m.set_defaults(func=cmd_demo)

    w = sub.add_parser("download", help="download detector weights")
    w.add_argument("--detector", default="uod2-x")
    w.set_defaults(func=cmd_download)

    g = sub.add_parser("config", help="print or save the default configuration")
    g.add_argument("--dump", help="write the defaults to this YAML file")
    g.set_defaults(func=cmd_config)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
