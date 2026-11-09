"""Parallel processing of many videos across the available GPUs (or CPU cores)."""

from .devices import Device, discover_devices, parse_devices, plan_workers
from .scheduler import BatchReport, output_dir_for, run_batch

__all__ = ["run_batch", "BatchReport", "Device", "discover_devices", "parse_devices", "plan_workers", "output_dir_for"]
