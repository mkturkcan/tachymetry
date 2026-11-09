"""Synthetic scenes and traffic with ground truth."""

from .scene import (
    Box3D,
    GroundTexture,
    RoadLayout,
    SyntheticScene,
    default_camera,
    make_ground_texture,
    make_scene,
    render_view,
)

__all__ = [
    "Box3D",
    "GroundTexture",
    "RoadLayout",
    "SyntheticScene",
    "default_camera",
    "make_ground_texture",
    "make_scene",
    "render_view",
]
