"""Manual calibration from correspondences, with cross-validated lens selection.

Points can be given in metres, in pixels of a north-up reference image (with its
ground sampling distance), or as latitude/longitude. Create the file with
``tachymetry calibrate pick clip.mp4 --reference ortho.png --gsd 0.1``.
"""

import sys

import cv2

from tachymetry.calib import Correspondences, background_image, calibrate_manual, preview
from tachymetry.io import VideoSource

if __name__ == "__main__":
    video, points = sys.argv[1], sys.argv[2]
    corr = Correspondences.load(points)
    with VideoSource(video) as src:
        bg = background_image(src.sample_frames(41))
        result = calibrate_manual(corr, src.size, lens="auto")
    print(result.summary())
    print({k: v["cv_rmse_m"] for k, v in result.details["lens_candidates"].items()})
    result.save("camera.json")
    cv2.imwrite("calibration_preview.jpg", preview(bg, result.camera))
