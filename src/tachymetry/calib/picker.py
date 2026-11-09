"""Browser-based picker for image-to-world point correspondences.

:func:`launch_picker` serves the self-contained page
``tachymetry/assets/picker.html`` from a small HTTP server that listens on
``127.0.0.1`` only, opens it in the default browser and blocks until the points
are saved::

    from tachymetry.calib.manual import calibrate_manual
    from tachymetry.calib.picker import launch_picker

    path = launch_picker("frame.png", reference="ortho.png", gsd=0.05, out="points.yaml")
    result = calibrate_manual(path)

The left pane shows the camera frame. With a reference image (orthophoto,
satellite tile or site plan) the right pane shows it and the world side of each
pair is picked on it; the file then stores reference pixels
(``world_units: reference_px`` with ``gsd``). Without a reference, the world
position of each image point is typed in metres (``world_units: metres``). The
page fits a homography as points are added, reports per-point residuals and
previews the bird's-eye view. *Save* posts the points back and the server writes
the file in the layout read by
:class:`~tachymetry.calib.correspondences.Correspondences`.

The page also works without the server: open ``picker.html`` from the package
directory in a browser, load the images with its file buttons and use
*Download YAML*.

Serving uses the standard library only (:mod:`http.server`, :mod:`threading`,
:mod:`webbrowser`); images are decoded and encoded with OpenCV.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import socketserver
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .correspondences import Correspondences

__all__ = ["launch_picker"]

_ASSETS = "tachymetry.assets"
_PAGE = "picker.html"
_FONTS = ("Geist-Regular.ttf", "Geist-Medium.ttf", "Geist-SemiBold.ttf", "GeistMono-Regular.ttf")
_MIN_POINTS = 4  # the manual calibration needs four pairs
_MAX_BODY = 16 * 1024 * 1024
_LOSSLESS_MAX_PIXELS = 24_000_000  # decoded rasters above this are served as JPEG instead of PNG
_UNITS = {
    "metres": "metres",
    "meters": "metres",
    "m": "metres",
    "reference_px": "reference_px",
    "pixels": "reference_px",
    "px": "reference_px",
    "latlon": "latlon",
    "wgs84": "latlon",
}
# The page loads nothing from other origins; this policy has the browser enforce that.
_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data: blob:; "
    "font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def _asset_bytes(name: str) -> bytes:
    """Contents of a file shipped in ``tachymetry/assets``."""
    try:
        return resources.files(_ASSETS).joinpath(name).read_bytes()
    except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError, TypeError, AttributeError):
        # source trees on interpreters whose importlib.resources lacks namespace-package support
        return (Path(__file__).resolve().parents[1] / "assets" / name).read_bytes()


# --------------------------------------------------------------------------- images
@dataclass
class _Image:
    """An image as served to the page."""

    data: bytes
    content_type: str
    width: int
    height: int
    name: str
    source: str | None = None  # absolute path when read from disk


def _sniff(data: bytes) -> str | None:
    """Content type of PNG and JPEG data, which browsers display without conversion."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return None


def _to_bgr8(array: np.ndarray) -> np.ndarray:
    """Convert a grey, BGR or BGRA array of a common dtype to contiguous 8-bit BGR or grey."""
    a = np.asarray(array)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)) or 0 in a.shape:
        raise ValueError(f"expected an HxW, HxWx3 (BGR) or HxWx4 (BGRA) image, got shape {a.shape}")
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    elif np.issubdtype(a.dtype, np.floating):
        f = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if f.max(initial=0.0) <= 1.0:  # treat [0, 1] floats as normalised intensities
            f = f * 255.0
        a = np.clip(np.rint(f), 0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    return np.ascontiguousarray(a)


def _encode(image: np.ndarray, name: str) -> _Image:
    import cv2

    h, w = image.shape[:2]
    if w * h <= _LOSSLESS_MAX_PIXELS:
        ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        content_type = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        content_type = "image/jpeg"
    if not ok:
        raise ValueError(f"could not encode {name}")
    return _Image(buf.tobytes(), content_type, int(w), int(h), name)


def _prepare_image(image: str | os.PathLike[str] | np.ndarray, what: str, default_name: str) -> _Image:
    """Read or encode an image for serving; ``what`` names it in error messages."""
    if isinstance(image, np.ndarray):
        return _encode(_to_bgr8(image), default_name)
    if isinstance(image, (str, os.PathLike)):
        import cv2

        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{what} not found: {path}")
        raw = path.read_bytes()
        # imdecode on the bytes also handles non-ASCII paths on Windows
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"could not decode the {what} {path} (unsupported image format?)")
        content_type = _sniff(raw)
        if content_type is not None:
            served = _Image(raw, content_type, int(decoded.shape[1]), int(decoded.shape[0]), path.name)
        else:  # TIFF, BMP, WebP, ...: re-encode for the browser
            served = _encode(decoded, path.name)
        served.source = str(path.resolve())
        return served
    raise TypeError(f"{what} must be an image path or a numpy array, not {type(image).__name__}")


# --------------------------------------------------------------------------- existing points
def _read_existing(existing: Any) -> tuple[Correspondences, dict[str, Any] | None]:
    """Load ``existing`` as ``(correspondences, raw document or None)``."""
    if isinstance(existing, Correspondences):
        return existing, None
    if isinstance(existing, dict):
        data = existing
    else:
        path = Path(existing).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"existing correspondences not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return Correspondences.load(path), None
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # written in the locale encoding, as Correspondences.load reads it
            text = path.read_text(encoding="utf-8")
        if suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a correspondence file")
    units = _UNITS.get(str(data.get("world_units", "metres")).lower())
    if units is None:
        raise ValueError(f"unknown world_units {data.get('world_units')!r} in the existing correspondences")
    if units == "reference_px" and data.get("gsd") is None:
        raise ValueError("existing correspondences in reference_px have no gsd")
    return Correspondences.from_dict(data), data


def _prepare_existing(
    existing: Any, mode: str, size: tuple[int, int]
) -> tuple[dict[str, Any] | None, float | None, str | None]:
    """Convert ``existing`` to the session's world units.

    Returns ``(document, gsd from the file, note)`` where ``document`` holds
    ``points`` and ``lines`` as the page expects them.
    """
    if existing is None:
        return None, None, None
    corr, data = _read_existing(existing)
    if data is not None:
        units = _UNITS[str(data.get("world_units", "metres")).lower()]
    else:
        units = {"reference_px": "reference_px", "latlon": "latlon"}.get(str(corr.georef.get("type")), "metres")
    file_gsd: float | None = None
    note = None
    if mode == "reference":
        if units != "reference_px":
            raise ValueError(
                f"the existing correspondences are in {units}; a session with a reference image "
                "edits reference_px correspondences picked on that image"
            )
        if data is not None:
            file_gsd = float(data["gsd"])
            world = np.asarray([p["world"] for p in data.get("points", [])], dtype=np.float64).reshape(-1, 2)
        else:
            file_gsd = float(corr.georef["gsd"])
            world = np.column_stack([corr.world[:, 0] / file_gsd, -corr.world[:, 1] / file_gsd])
    else:
        world = corr.world
        if units != "metres":
            note = f"the existing {units} coordinates were converted to local metres"
    if corr.image_size is not None and tuple(int(v) for v in corr.image_size) != tuple(size):
        w0, h0 = (int(v) for v in corr.image_size)
        raise ValueError(
            f"the existing correspondences were picked on a {w0}x{h0} image, "
            f"but the camera frame is {size[0]}x{size[1]}"
        )
    names = corr.names or [str(i + 1) for i in range(len(corr))]
    points = [
        {"name": str(n), "image": [float(u), float(v)], "world": [float(x), float(y)]}
        for n, (u, v), (x, y) in zip(names, corr.image, world)
    ]
    lines = [np.asarray(line, dtype=np.float64).reshape(-1, 2).tolist() for line in corr.lines]
    return {"points": points, "lines": lines}, file_gsd, note


# --------------------------------------------------------------------------- output
def _xy(value: Any, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{what} must be a pair of numbers")
    try:
        out = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a pair of numbers") from None
    if not (math.isfinite(out[0]) and math.isfinite(out[1])):
        raise ValueError(f"{what} is not finite")
    return out


class _FlowList(list):
    """A list written in YAML flow style (``[x, y]``)."""


class _FlowDict(dict):
    """A mapping written in YAML flow style (``{name: a, image: [u, v], world: [x, y]}``)."""


def _yaml_text(doc: dict[str, Any], header: list[str]) -> str:
    import yaml

    class Dumper(yaml.SafeDumper):
        pass

    Dumper.add_representer(_FlowList, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
    Dumper.add_representer(_FlowDict, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=True))
    body = dict(doc)
    body["image_size"] = _FlowList(doc["image_size"])
    body["points"] = [
        _FlowDict(name=p["name"], image=_FlowList(p["image"]), world=_FlowList(p["world"])) for p in doc["points"]
    ]
    body["lines"] = [_FlowList(_FlowList(q) for q in line) for line in doc["lines"]]
    # ASCII output with escapes: Correspondences.load reads the file in the locale encoding (cp1252 on
    # many Windows systems), so non-ASCII names and paths would not survive a UTF-8 file there.
    text = yaml.dump(body, Dumper=Dumper, sort_keys=False, default_flow_style=False, allow_unicode=False, width=1 << 16)
    comments = (line.encode("ascii", "backslashreplace").decode("ascii") for line in header)
    return "".join(f"# {line}\n" for line in comments) + text


def _write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file so an interrupted save never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# --------------------------------------------------------------------------- session
class _AlreadySaved(RuntimeError):
    """A second save arrived after the file was written."""


class _Session:
    """What the page needs, and the outcome of the save."""

    def __init__(
        self, camera: _Image, reference: _Image | None, gsd: float | None, out: Path, existing: dict[str, Any] | None
    ) -> None:
        self.camera = camera
        self.reference = reference
        self.gsd = gsd
        self.out = out
        self.existing = existing
        self.token = secrets.token_urlsafe(24)
        self.page = _asset_bytes(_PAGE)
        self.saved = threading.Event()
        self.saved_path: Path | None = None
        self.summary = ""
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return "metric" if self.reference is None else "reference"

    def describe(self) -> dict[str, Any]:
        """Session description served at ``/api/session``."""

        def image(img: _Image, url: str) -> dict[str, Any]:
            return {"url": url, "width": img.width, "height": img.height, "name": img.name}

        return {
            "app": "tachymetry-picker",
            "mode": self.mode,
            "token": self.token,
            "image": image(self.camera, "image"),
            "reference": None if self.reference is None else image(self.reference, "reference"),
            "gsd": self.gsd,
            "out": str(self.out),
            "out_name": self.out.name,
            "min_points": _MIN_POINTS,
            "existing": self.existing,
        }

    def document(self, payload: Any) -> dict[str, Any]:
        """Validate a save request and build the correspondence document."""
        if not isinstance(payload, dict):
            raise ValueError("the request body must be a JSON object")
        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("'points' must be a list")
        world_digits = 3 if self.mode == "reference" else 4
        points = []
        for i, p in enumerate(raw_points, 1):
            if not isinstance(p, dict):
                raise ValueError(f"point {i} must be an object")
            raw_name = p.get("name")
            name = str(raw_name).strip() if raw_name not in (None, "") else ""
            name = name or str(i)
            image = _xy(p.get("image"), f"point {name}: image")
            world = _xy(p.get("world"), f"point {name}: world")
            points.append(
                {"name": name, "image": [round(v, 3) for v in image], "world": [round(v, world_digits) for v in world]}
            )
        if len(points) < _MIN_POINTS:
            raise ValueError(f"at least {_MIN_POINTS} point pairs are needed for calibration, got {len(points)}")
        raw_lines = payload.get("lines") or []
        if not isinstance(raw_lines, list):
            raise ValueError("'lines' must be a list of polylines")
        lines = []
        for j, line in enumerate(raw_lines, 1):
            if not isinstance(line, list) or len(line) < 2:
                raise ValueError(f"line {j} needs at least two vertices")
            lines.append([[round(v, 3) for v in _xy(q, f"line {j}: vertex")] for q in line])
        doc: dict[str, Any] = {"image_size": [self.camera.width, self.camera.height]}
        if self.mode == "reference":
            gsd = payload.get("gsd")
            gsd = self.gsd if gsd is None else gsd
            try:
                gsd = float(gsd)
            except (TypeError, ValueError):
                gsd = float("nan")
            if not (math.isfinite(gsd) and gsd > 0):
                raise ValueError(
                    "the ground sampling distance of the reference image (gsd, metres per pixel) "
                    "must be a positive number"
                )
            doc["world_units"] = "reference_px"
            doc["gsd"] = gsd
        else:
            doc["world_units"] = "metres"
        doc["points"] = points
        doc["lines"] = lines
        Correspondences.from_dict(doc)  # the same parser the calibration uses
        return doc

    def header(self) -> list[str]:
        """Comment lines that record where the points were picked."""

        def describe(img: _Image) -> str:
            return f"{img.source or 'in-memory image'} ({img.width}x{img.height})"

        lines = [
            "Image-to-world correspondences from tachymetry.calib.picker",
            f"camera frame: {describe(self.camera)}",
        ]
        if self.reference is not None:
            lines.append(f"reference image: {describe(self.reference)}")
        return [" ".join(line.splitlines()) for line in lines]

    def save(self, payload: Any) -> Path:
        with self._lock:
            if self.saved_path is not None:
                raise _AlreadySaved(str(self.saved_path))
            doc = self.document(payload)
            if self.out.suffix.lower() == ".json":
                text = json.dumps(doc, indent=2) + "\n"
            else:
                text = _yaml_text(doc, self.header())
            _write_atomic(self.out, text)
            self.saved_path = self.out
            self.summary = f"{_plural(len(doc['points']), 'point pair')} and {_plural(len(doc['lines']), 'line')}"
            return self.out


# --------------------------------------------------------------------------- HTTP
class _Handler(BaseHTTPRequestHandler):
    server: _PickerServer
    server_version = "tachymetry-picker"
    sys_version = ""
    timeout = 60

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base-class signature
        """Keep the terminal quiet; the picker prints its own status lines."""

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def _host_allowed(self) -> bool:
        """Only answer requests addressed to the loopback server (defeats DNS rebinding)."""
        host = (self.headers.get("Host") or "").strip().lower()
        port = self.server.server_port
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
            return
        session = self.server.session
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, session.page, "text/html; charset=utf-8", {"Content-Security-Policy": _CSP})
        elif path == "/api/session":
            self._json(HTTPStatus.OK, session.describe())
        elif path == "/image":
            self._send(HTTPStatus.OK, session.camera.data, session.camera.content_type)
        elif path == "/reference" and session.reference is not None:
            self._send(HTTPStatus.OK, session.reference.data, session.reference.content_type)
        elif path.lstrip("/") in _FONTS:
            self._send(HTTPStatus.OK, _asset_bytes(path.lstrip("/")), "font/ttf")
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
            return
        if urlsplit(self.path).path != "/save":
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        session = self.server.session
        if self.headers.get_content_type() != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected an application/json body")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(HTTPStatus.LENGTH_REQUIRED, "missing Content-Length")
            return
        if not 0 < length <= _MAX_BODY:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is empty or too large")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}")
            return
        token = self.headers.get("X-Picker-Token")
        if token is None and isinstance(payload, dict):
            token = payload.get("token")
        if not isinstance(token, str) or not secrets.compare_digest(token.encode(), session.token.encode()):
            self._error(HTTPStatus.FORBIDDEN, "invalid session token")
            return
        try:
            path = session.save(payload)
        except _AlreadySaved as exc:
            self._error(HTTPStatus.CONFLICT, f"this session has already been saved to {exc}")
            return
        except (ValueError, KeyError, TypeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except OSError as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"could not write {session.out}: {exc}")
            return
        self._json(HTTPStatus.OK, {"ok": True, "path": str(path), "summary": session.summary})
        session.saved.set()  # after the reply is on the wire, so the page learns where the file went


class _PickerServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that is in use.
    allow_reuse_address = os.name != "nt"

    def __init__(self, port: int, session: _Session) -> None:
        self.session = session
        super().__init__(("127.0.0.1", port), _Handler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind resolves the host name with getfqdn(), which can stall for seconds.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return  # the browser dropped a connection, e.g. on reload
        super().handle_error(request, client_address)


def _say(message: str) -> None:
    """Print a status line; paths the console encoding cannot show are escaped instead of failing."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _open_browser(url: str) -> None:
    try:
        opened = webbrowser.open(url, new=2)
    except Exception:  # noqa: BLE001 - no usable browser, e.g. in a headless session
        opened = False
    if not opened:
        _say(f"  Could not open a browser; open {url} manually.")


# --------------------------------------------------------------------------- public API
def launch_picker(
    image: str | os.PathLike[str] | np.ndarray,
    reference: str | os.PathLike[str] | np.ndarray | None = None,
    gsd: float | None = None,
    out: str | os.PathLike[str] = "points.yaml",
    port: int = 0,
    open_browser: bool = True,
    existing: Any = None,
) -> Path:
    """Pick image-to-world correspondences in the browser; block until they are saved.

    Parameters
    ----------
    image:
        Camera frame: an image path or a BGR ``numpy`` array, such as a
        background frame of the video. Point coordinates refer to its pixels.
    reference:
        Optional north-up reference image (orthophoto, satellite tile or site
        plan): a path or a BGR array. World points are then picked on it and
        saved as ``world_units: reference_px``. Without a reference, world
        positions are typed in metres and saved as ``world_units: metres``.
    gsd:
        Ground sampling distance of the reference image in metres per pixel.
        It can also be entered or corrected in the page; saving a reference
        session requires it.
    out:
        Output file: ``.yaml``/``.yml`` (the default) or ``.json``.
    port:
        Local port; ``0`` picks a free one. The server listens on ``127.0.0.1`` only.
    open_browser:
        Open the page in the default browser. The URL is printed either way.
    existing:
        Correspondences to pre-load for editing: a file path, a dict in the
        file layout or a :class:`~tachymetry.calib.correspondences.Correspondences`.
        A reference session needs ``reference_px`` correspondences picked on
        the same reference image; a metric session converts other units to metres.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    KeyboardInterrupt
        On Ctrl+C before a save; the server is shut down first.
    """
    out_path = Path(out).expanduser().resolve()
    if out_path.suffix.lower() == ".csv":
        raise ValueError("the picker writes YAML (.yaml, .yml) or JSON (.json), not CSV")
    if gsd is not None:
        gsd = float(gsd)
        if not (math.isfinite(gsd) and gsd > 0):
            raise ValueError(f"gsd must be a positive number of metres per pixel, got {gsd}")
        if reference is None:
            raise ValueError("gsd describes a reference image; pass reference=... or leave gsd out")
    port = int(port)
    if not 0 <= port <= 65535:
        raise ValueError(f"port must be in [0, 65535], got {port}")

    camera = _prepare_image(image, "camera frame", "frame.png")
    ref = None if reference is None else _prepare_image(reference, "reference image", "reference.png")
    mode = "metric" if ref is None else "reference"
    doc, file_gsd, note = _prepare_existing(existing, mode, (camera.width, camera.height))
    session = _Session(camera, ref, gsd if gsd is not None else file_gsd, out_path, doc)

    server = _PickerServer(port, session)
    url = f"http://127.0.0.1:{server.server_port}/"
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, name="tachymetry-picker", daemon=True
    )
    thread.start()
    if ref is None:
        world = "world positions typed in metres"
    else:
        scale = f"{session.gsd:g} m/px" if session.gsd else "GSD to be entered in the page"
        world = f"reference {ref.width}x{ref.height} ({ref.name}, {scale})"
    _say(f"Correspondence picker running at {url}")
    _say(f"  camera frame {camera.width}x{camera.height} ({camera.name}); {world}")
    if doc is not None:
        _say(f"  loaded {_plural(len(doc['points']), 'existing point pair')} and {_plural(len(doc['lines']), 'line')}")
    if note:
        _say(f"  note: {note}")
    _say(f"  output: {out_path}")
    _say("  Press Save in the page when done (Ctrl+C here cancels).")
    if open_browser:
        threading.Thread(target=_open_browser, args=(url,), name="tachymetry-picker-browser", daemon=True).start()

    try:
        # Wait with a timeout: on Windows a blocking Event.wait() does not return for Ctrl+C.
        while not session.saved.wait(0.25):
            if not thread.is_alive():
                raise RuntimeError("the picker server stopped unexpectedly")
    except KeyboardInterrupt:
        _say("Picker cancelled; nothing was saved.")
        raise
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
    if session.saved_path is None:  # the loop only ends normally after a save
        raise RuntimeError("the picker stopped without saving")
    _say(f"Saved {session.summary} to {session.saved_path}")
    return session.saved_path
