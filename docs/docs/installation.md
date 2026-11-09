# Installation

Tachymetry requires Python 3.10 to 3.13 and installs from GitHub:

```bash
pip install git+https://github.com/mkturkcan/tachymetry
```

The runtime dependencies (`pyproject.toml`) are NumPy, SciPy, OpenCV, Pillow, PyYAML, Rich, `huggingface_hub`, `imageio-ffmpeg` (a bundled FFmpeg for H.264 output), pandas, PyTorch 2.1 or newer and Ultralytics 8.4.61 or newer. Detector weights and the sample clips are downloaded into the Hugging Face cache on first use.

Check the installation and the compute devices:

```bash
tachymetry --version
tachymetry devices
```

`tachymetry devices` lists the CUDA GPUs (or the Apple GPU, or the CPU) with their free memory and the worker plan that `tachymetry run` would use.

## Linux

The PyTorch wheels on PyPI for Linux include CUDA, so the command above is sufficient on a machine with an NVIDIA driver. To select a specific CUDA build, install PyTorch first from the [PyTorch index](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/mkturkcan/tachymetry
```

## Windows

The default PyTorch wheel on PyPI for Windows is CPU-only. With an NVIDIA GPU, install the CUDA build of PyTorch first, then Tachymetry:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/mkturkcan/tachymetry
python -c "import torch; print(torch.cuda.is_available())"
```

Batch processing uses the `spawn` start method on every platform. Scripts that call `tm.track_many` must therefore protect their entry point with `if __name__ == "__main__":`, as described in [Batch processing](guide/batch.md). Paths may use either separator.

## macOS

```bash
pip install git+https://github.com/mkturkcan/tachymetry
```

On Apple silicon the detector runs on the Metal (MPS) device, which `device="auto"` selects when CUDA is unavailable; the batch scheduler runs one worker on it.

## CPU only

Install the CPU build of PyTorch first to avoid downloading CUDA libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/mkturkcan/tachymetry
```

Detection dominates the run time on a CPU. A smaller detector (`--detector uod2-s`), a lower analysis rate (`--rate 5`) and a smaller input size (`--imgsz 640`) reduce it; see the [configuration reference](guide/configuration.md).

## Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `depth` | `transformers>=4.48`, `accelerate>=0.26` | Calibration from monocular metric depth and road segmentation (`tachymetry calibrate depth`, `method="depth"`) with Depth Pro, Depth Anything V2 and SegFormer |
| `parquet` | `pyarrow>=12` | Parquet export (`--formats parquet`) |
| `docs` | `mkdocs-material`, `mkdocstrings[python]` | Building this documentation |
| `dev` | `pytest`, `ruff`, `pyarrow`, `pandas` | Tests and linting |
| `all` | `depth` and `parquet` | Both optional run-time features |

```bash
pip install "tachymetry[depth] @ git+https://github.com/mkturkcan/tachymetry"
```

Two back-ends are installed from their repositories: MoGe-2 for depth calibration (`pip install git+https://github.com/microsoft/MoGe.git`) and GeoCalib for single-image calibration (`pip install git+https://github.com/cvg/GeoCalib.git`).

## Development install

```bash
git clone https://github.com/mkturkcan/tachymetry
cd tachymetry
pip install -e ".[dev,docs]"
pytest
mkdocs serve
```

Detector weights can be fetched ahead of time, for example on a machine that will later run without network access:

```bash
tachymetry download --detector uod2-x
```

The command prints the path of the checkpoint in the Hugging Face cache.
