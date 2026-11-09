"""Tests for the browser-based correspondence picker server (:mod:`tachymetry.calib.picker`).

The session tests run :func:`launch_picker` in a background thread on an
ephemeral port, talk to it over HTTP the way the page does, and check that the
saved file loads with :class:`Correspondences` and calibrates with
:func:`calibrate_manual` to the camera that generated the points. The last test
covers argument errors that must be raised before the server starts.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from tachymetry.calib.correspondences import Correspondences
from tachymetry.calib.manual import calibrate_manual
from tachymetry.calib.picker import launch_picker
from tachymetry.geometry import Camera

SIZE = (1280, 720)
GSD = 0.05  # metres per reference pixel
REF_SIZE = (1200, 1200)  # north-up reference covering x in [0, 60] m and y in [-60, 0] m

# The server listens on the loopback interface only, so bypass proxy settings.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _camera() -> Camera:
    """A pole camera 9 m up, south of the picked area, looking roughly north."""
    return Camera.from_pose(SIZE, focal=1000.0, height=9.0, pitch=28.0, yaw=95.0, roll=1.2, position=(24.0, -62.0))


def _world_points() -> np.ndarray:
    """Sixteen ground points (metres) spread over the visible road surface."""
    rows = {
        -52.0: [19.5, 22.5, 25.5, 28.5],
        -45.0: [18.0, 22.0, 26.0, 30.0],
        -38.0: [16.0, 21.0, 26.0, 31.0],
        -31.0: [15.0, 21.0, 27.0, 33.0],
    }
    return np.array([[x + 0.25 * (i % 2), y + 0.4 * (i % 3)] for y, xs in rows.items() for i, x in enumerate(xs)])


def _line(cam: Camera) -> np.ndarray:
    """Image polyline of a kerb that is straight on the ground (x = 21 m)."""
    world = np.column_stack([np.full(6, 21.0), np.linspace(-52.0, -30.0, 6)])
    return cam.ground_to_image(world)


def _frame() -> np.ndarray:
    yy, xx = np.mgrid[0 : SIZE[1], 0 : SIZE[0]]
    return np.stack([xx % 256, yy % 256, (xx // 3 + yy) % 256], axis=-1).astype(np.uint8)


def _orthophoto() -> np.ndarray:
    ortho = np.full((REF_SIZE[1], REF_SIZE[0], 3), 70, dtype=np.uint8)
    ortho[::100] = 200
    ortho[:, ::100] = 200
    return ortho


def _request(req: urllib.request.Request) -> tuple[int, bytes, Any]:
    try:
        with _OPENER.open(req, timeout=10) as resp:
            return resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read(), exc.headers
        finally:
            exc.close()


def _get(url: str) -> tuple[int, bytes, Any]:
    return _request(urllib.request.Request(url))


def _post(
    url: str,
    payload: Any,
    token: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    h = {"Content-Type": "application/json"}
    if token is not None:
        h["X-Picker-Token"] = token
    h.update(headers or {})
    status, body, _ = _request(urllib.request.Request(url, data=json.dumps(payload).encode(), headers=h, method="POST"))
    return status, json.loads(body)


class _Picker:
    """:func:`launch_picker` running in a background thread, as a user would run it in a terminal."""

    def __init__(self, capsys: pytest.CaptureFixture[str], **kwargs: Any) -> None:
        self.result: dict[str, Any] = {}
        self.thread = threading.Thread(target=self._run, kwargs=kwargs, daemon=True)
        self.thread.start()
        self.url = self._wait_for_url(capsys)

    def _run(self, **kwargs: Any) -> None:
        try:
            self.result["path"] = launch_picker(port=0, open_browser=False, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised by finish()
            self.result["error"] = exc

    def _wait_for_url(self, capsys: pytest.CaptureFixture[str]) -> str:
        printed = ""
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            printed += capsys.readouterr().out
            match = re.search(r"http://127\.0\.0\.1:(\d+)/", printed)
            if match:
                assert int(match.group(1)) > 0
                return match.group(0)
            if not self.thread.is_alive():
                self.finish()
            time.sleep(0.02)
        raise AssertionError(f"the picker did not print its URL; output so far: {printed!r}")

    def finish(self, timeout: float = 30.0) -> Path:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "launch_picker did not return after the save"
        if "error" in self.result:
            raise self.result["error"]
        return self.result["path"]


def _assert_recovers(corr: Correspondences, truth: Camera) -> None:
    result = calibrate_manual(corr, lens="identity", cv_folds=None)
    cam = result.camera
    assert result.rmse_m < 2e-3
    assert cam.focal == pytest.approx(truth.focal, rel=2e-3)
    assert cam.height == pytest.approx(truth.height, abs=0.02)
    for got, want in zip(cam.angles, truth.angles):
        assert got == pytest.approx(want, abs=0.05)
    np.testing.assert_allclose(cam.center[:2], truth.center[:2], atol=0.05)


def test_reference_session_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cam = _camera()
    world = _world_points()
    image = cam.ground_to_image(world)
    assert np.all((image > 0) & (image < np.array(SIZE)))
    ref_px = np.column_stack([world[:, 0] / GSD, -world[:, 1] / GSD])  # north-up, y down
    assert np.all((ref_px > 0) & (ref_px < np.array(REF_SIZE)))
    frame = _frame()
    ortho_path = tmp_path / "ortho.png"
    assert cv2.imwrite(str(ortho_path), _orthophoto())
    out = tmp_path / "picked" / "points.yaml"

    picker = _Picker(capsys, image=frame, reference=ortho_path, gsd=GSD, out=out)

    status, html, headers = _get(picker.url)
    assert status == 200 and headers.get_content_type() == "text/html"
    page = html.decode("utf-8")
    assert "<canvas" in page and "tachymetryPicker" in page
    assert not re.search(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", page), "the page must not load remote assets"
    assert "@import" not in page
    assert "default-src 'none'" in headers["Content-Security-Policy"]

    status, body, _ = _get(picker.url + "api/session")
    assert status == 200
    session = json.loads(body)
    assert session["mode"] == "reference"
    assert (session["image"]["width"], session["image"]["height"]) == SIZE
    assert (session["reference"]["width"], session["reference"]["height"]) == REF_SIZE
    assert session["gsd"] == GSD
    assert session["existing"] is None
    assert Path(session["out"]) == out.resolve()

    status, png, headers = _get(picker.url + session["image"]["url"])
    assert status == 200 and headers.get_content_type() == "image/png"
    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), frame)
    status, ref_bytes, headers = _get(picker.url + session["reference"]["url"])
    assert status == 200 and headers.get_content_type() == "image/png"
    assert ref_bytes == ortho_path.read_bytes()  # PNG files are served unchanged
    assert _get(picker.url + "Geist-Regular.ttf")[0] == 200
    assert _get(picker.url + "../picker.py")[0] == 404

    points = [
        {"name": f"p{i}", "image": u.tolist(), "world": r.tolist()} for i, (u, r) in enumerate(zip(image, ref_px), 1)
    ]
    payload = {
        "image_size": list(SIZE),
        "world_units": "reference_px",
        "gsd": GSD,
        "points": points,
        "lines": [_line(cam).tolist()],
    }
    status, reply = _post(picker.url + "save", payload, token=session["token"])
    assert status == 200, reply
    assert reply["ok"] and Path(reply["path"]) == out.resolve()

    path = picker.finish()
    assert path == out.resolve() and path.is_file()
    with pytest.raises(OSError):  # the server has shut down
        _get(picker.url)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("# ") and str(ortho_path.resolve()) in text
    doc = yaml.safe_load(text)
    assert doc["image_size"] == list(SIZE)
    assert doc["world_units"] == "reference_px" and doc["gsd"] == GSD
    corr = Correspondences.load(path)
    assert corr.image_size == SIZE
    assert corr.names == [f"p{i}" for i in range(1, len(world) + 1)]
    np.testing.assert_allclose(corr.image, image, atol=1e-3)
    np.testing.assert_allclose(corr.world, world, atol=1e-4)  # reference pixels back to metres
    assert len(corr.lines) == 1
    np.testing.assert_allclose(corr.lines[0], _line(cam), atol=1e-3)
    _assert_recovers(corr, cam)

    # Re-open the saved file for editing: the reference pixels come back unchanged.
    out2 = tmp_path / "edited.yaml"
    picker = _Picker(capsys, image=frame, reference=ortho_path, out=out2, existing=path)
    session = json.loads(_get(picker.url + "api/session")[1])
    assert session["gsd"] == GSD  # taken from the file
    loaded = session["existing"]
    np.testing.assert_allclose([p["world"] for p in loaded["points"]], ref_px, atol=1e-3)
    assert len(loaded["lines"]) == 1
    loaded["points"][0]["name"] = "Stra\u00dfe kerb: #1"
    status, reply = _post(picker.url + "save", {"gsd": GSD, **loaded}, token=session["token"])
    assert status == 200, reply
    path2 = picker.finish()
    assert path2.read_bytes().isascii()  # loads the same under the locale encodings of Windows
    edited = Correspondences.load(path2)
    assert edited.names[0] == "Stra\u00dfe kerb: #1"
    np.testing.assert_allclose(edited.world, corr.world, atol=1e-6)


def test_metric_session_with_existing_points(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cam = _camera()
    world = _world_points()
    image = cam.ground_to_image(world)
    frame_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame_path), _frame())
    existing = {
        "image_size": list(SIZE),
        "world_units": "metres",
        "points": [
            {"name": f"e{i}", "image": u.tolist(), "world": w.tolist()}
            for i, (u, w) in enumerate(zip(image[:5], world[:5]), 1)
        ],
    }
    out = tmp_path / "points.yaml"

    picker = _Picker(capsys, image=frame_path, out=out, existing=existing)

    session = json.loads(_get(picker.url + "api/session")[1])
    assert session["mode"] == "metric" and session["reference"] is None and session["gsd"] is None
    pre = session["existing"]["points"]
    assert [p["name"] for p in pre] == [f"e{i}" for i in range(1, 6)]
    np.testing.assert_allclose([p["world"] for p in pre], world[:5])
    assert _get(picker.url + "reference")[0] == 404
    status, png, _ = _get(picker.url + "image")
    assert status == 200 and png == frame_path.read_bytes()

    points = [
        {"name": f"e{i}", "image": u.tolist(), "world": w.tolist()} for i, (u, w) in enumerate(zip(image, world), 1)
    ]
    good = {"image_size": list(SIZE), "world_units": "metres", "points": points, "lines": []}
    token = session["token"]

    # Rejected requests leave the session running and write nothing.
    assert _post(picker.url + "save", good)[0] == 403  # no token
    assert _post(picker.url + "save", good, token="not-the-token")[0] == 403
    assert _post(picker.url + "save", good, token=token, headers={"Host": "attacker.example:80"})[0] == 403
    status, reply = _post(picker.url + "save", {"points": points[:3]}, token=token)
    assert status == 400 and "at least 4" in reply["error"]
    nan_point = dict(points[0], world=[float("nan"), 1.0])
    status, reply = _post(picker.url + "save", {"points": [nan_point] + points[1:]}, token=token)
    assert status == 400 and "not finite" in reply["error"]
    status, _, _ = _request(
        urllib.request.Request(
            picker.url + "save",
            data=b"points=1",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "X-Picker-Token": token},
        )
    )
    assert status == 415
    assert _post(picker.url + "elsewhere", good, token=token)[0] == 404
    assert picker.thread.is_alive() and not out.exists()

    status, reply = _post(picker.url + "save", good, token=token)
    assert status == 200, reply
    path = picker.finish()
    assert path == out.resolve()

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["world_units"] == "metres" and "gsd" not in doc and doc["lines"] == []
    corr = Correspondences.load(path)
    assert corr.image_size == SIZE and len(corr) == len(world)
    np.testing.assert_allclose(corr.world, world, atol=1e-4)
    _assert_recovers(corr, cam)


def test_invalid_arguments_fail_before_serving(tmp_path: Path) -> None:
    frame = _frame()
    out = tmp_path / "points.yaml"
    metres = {"world_units": "metres", "points": [{"name": "a", "image": [1.0, 2.0], "world": [3.0, 4.0]}]}
    with pytest.raises(ValueError, match="reference_px"):
        launch_picker(frame, reference=_orthophoto(), gsd=GSD, out=out, open_browser=False, existing=metres)
    with pytest.raises(ValueError, match="640x480"):
        launch_picker(frame, out=out, open_browser=False, existing=dict(metres, image_size=[640, 480]))
    with pytest.raises(ValueError, match="CSV"):
        launch_picker(frame, out=tmp_path / "points.csv", open_browser=False)
    with pytest.raises(ValueError, match="reference"):
        launch_picker(frame, gsd=0.1, out=out, open_browser=False)
    with pytest.raises(FileNotFoundError):
        launch_picker(tmp_path / "missing.png", out=out, open_browser=False)
    with pytest.raises(FileNotFoundError):
        launch_picker(frame, out=out, open_browser=False, existing=tmp_path / "missing.yaml")
    assert not out.exists()
