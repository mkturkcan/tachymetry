# Contributing

Bug reports and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/mkturkcan/tachymetry
cd tachymetry
pip install -e ".[dev,docs]"
pytest                      # CPU-only test suite (about 10 s)
ruff check src tests tools
mkdocs serve                # documentation at http://127.0.0.1:8000
```

## Guidelines

- Express new thresholds in physical units (seconds, metres, metres per second) so behaviour does not depend on frame rate or resolution, and add them to `tachymetry/config.py` with a comment.
- New filtering rules should come with a test on simulated traffic (`tachymetry.synthetic.traffic`) showing the failure they remove and that they do not drop valid tracks.
- Keep the public API small: `tachymetry.track`, `tachymetry.track_many`, `tachymetry.calibrate`, `Camera`, `PipelineConfig`.
- Documentation lives in `docs/docs/`; regenerate figures with `python tools/make_docs_media.py`.
