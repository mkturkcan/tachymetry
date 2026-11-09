import numpy as np
import pytest

from tachymetry.geometry import Camera


@pytest.fixture
def pole_camera() -> Camera:
    return Camera.from_pose(
        (1280, 720), focal=914.0, height=8.0, pitch=32.0, yaw=45.0, roll=1.0, position=(-14.0, -14.0)
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
