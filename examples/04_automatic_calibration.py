"""Compare the automatic calibration methods on one video and preview them side by side."""

import sys

import cv2

import tachymetry as tm
from tachymetry.calib import compare
from tachymetry.calib.lines import background_image
from tachymetry.data import sample_video
from tachymetry.io import VideoSource

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else sample_video("intersection-day")
    result = tm.calibrate(video, method="auto")
    print(result.summary())
    for c in result.details["candidates"]:
        print(f"{c['method']:8s} score={c['score']:.3f}  {c['camera']}")
    with VideoSource(video) as src:
        bg = background_image(src.sample_frames(41))
    cams = [(name, cam) for name, cam in result.details["candidate_cameras"].items()]
    cv2.imwrite("calibration_candidates.jpg", compare(bg, cams, cols=1))
