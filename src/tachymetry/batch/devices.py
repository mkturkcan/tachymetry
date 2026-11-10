"""Compute-device discovery and worker planning."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["worker_gb", "Device", "discover_devices", "plan_workers"]


@dataclass
class Device:
    """A compute device available to this process."""

    name: str  # "cuda:0", "mps", "cpu"
    kind: str  # "cuda" | "mps" | "cpu"
    label: str
    total_gb: float = 0.0
    free_gb: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        mem = f", {self.free_gb:.1f}/{self.total_gb:.1f} GB free" if self.total_gb else ""
        return f"{self.name} ({self.label}{mem})"


def discover_devices() -> list[Device]:
    """All CUDA GPUs (or Apple MPS), falling back to the CPU. Never raises."""
    devices: list[Device] = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                try:
                    free, total = torch.cuda.mem_get_info(i)
                except Exception:
                    props = torch.cuda.get_device_properties(i)
                    free, total = props.total_memory, props.total_memory
                devices.append(Device(f"cuda:{i}", "cuda", name, total / 2**30, free / 2**30))
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices.append(Device("mps", "mps", "Apple GPU (Metal)"))
    except Exception:
        pass
    if not devices:
        devices.append(Device("cpu", "cpu", f"{os.cpu_count() or 1} CPU threads"))
    return devices


def parse_devices(spec: str | Sequence[str] | None) -> list[Device]:
    """``"auto"``, ``"cpu"``, ``"cuda"``, ``"0,2"``, ``"cuda:1,cuda:3"`` or a list of names."""
    found = discover_devices()
    if spec is None or spec == "auto":
        return found
    names = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    if names == ["cpu"]:
        return [Device("cpu", "cpu", f"{os.cpu_count() or 1} CPU threads")]
    if names == ["cuda"] or names == ["gpu"]:
        gpus = [d for d in found if d.kind == "cuda"]
        if not gpus:
            raise RuntimeError("no CUDA device is available")
        return gpus
    out = []
    by_name = {d.name: d for d in found}
    for n in names:
        key = f"cuda:{n}" if n.isdigit() else n
        if key in by_name:
            out.append(by_name[key])
        elif key == "mps" or key == "cpu":
            out.append(Device(key, key, key))
        else:
            raise RuntimeError(f"device {n!r} not available; found {[d.name for d in found]}")
    return out


def worker_gb(cfg) -> float:
    """GPU memory one worker process needs, by detector (measured peaks plus headroom).

    SAM 3 peaks at about 7.3 GB with batch 8 in BF16; RF-DETR-Seg L and
    UrbanOmniDetect-2 X stay below 4 GB.
    """
    name = str(getattr(getattr(cfg, "detector", None), "name", "")).lower()
    return 9.0 if name.startswith("sam3") else 6.0


def plan_workers(
    devices: Sequence[Device],
    workers_per_device: int | str = "auto",
    n_jobs: int | None = None,
    gb_per_worker: float = 6.0,
) -> list[str]:
    """One device name per worker process.

    ``auto`` runs two workers per GPU when memory allows (one decodes, tracks
    and renders while the other occupies the GPU), one on MPS, and on CPU a
    worker per four threads. Never more workers than jobs.
    """
    cpu_threads = os.cpu_count() or 1
    per_device: list[int] = []
    for d in devices:
        if isinstance(workers_per_device, int) or (
            isinstance(workers_per_device, str) and workers_per_device.isdigit()
        ):
            k = int(workers_per_device)
        elif d.kind == "cuda":
            k = max(1, min(3, int(d.free_gb // gb_per_worker))) if d.free_gb else 2
            k = min(k, 2)
        elif d.kind == "mps":
            k = 1
        else:
            k = max(1, cpu_threads // 4)
        per_device.append(k)
    # Round-robin order, so that truncating the list below still spreads workers over all devices.
    slots = [d.name for i in range(max(per_device, default=0)) for d, k in zip(devices, per_device) if i < k]
    gpu_slots = sum(1 for s in slots if s.startswith("cuda"))
    if gpu_slots:  # keep enough CPU threads per worker for decoding and tracking
        cap = max(1, cpu_threads // 2)
        slots = slots[: max(len(devices), min(len(slots), cap))]
    if n_jobs is not None:
        slots = slots[: max(1, n_jobs)] if len(slots) > n_jobs else slots
    return slots
