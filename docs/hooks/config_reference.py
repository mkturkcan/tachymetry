"""MkDocs hook: generate the configuration reference from the dataclasses in ``src/tachymetry/config.py``.

The page ``docs/guide/configuration.md`` contains the marker ``<!-- config-reference -->``. At build
time the marker is replaced by one table per dataclass (field, type, default, description). The
source is parsed with :mod:`ast` rather than imported, so building the documentation does not need
the package's runtime dependencies. Descriptions come from the ``#:`` comments next to each field.
"""

from __future__ import annotations

import ast
from pathlib import Path

MARKER = "<!-- config-reference -->"
CONFIG = Path(__file__).resolve().parents[2] / "src" / "tachymetry" / "config.py"
# Order of the sections and the attribute path of each dataclass inside PipelineConfig.
SECTIONS = [
    ("PipelineConfig", ""),
    ("DetectorConfig", "detector."),
    ("TrackerConfig", "tracker."),
    ("HygieneConfig", "hygiene."),
    ("FilterConfig", "filters."),
    ("CalibrationConfig", "calibration."),
    ("RenderConfig", "render."),
]


def _default(node: ast.AST | None, src: str) -> str:
    if node is None:
        return "required"
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "field":
        for kw in node.keywords:
            if kw.arg == "default_factory":
                v = kw.value
                if isinstance(v, ast.Lambda):
                    return ast.get_source_segment(src, v.body) or ""
                if isinstance(v, ast.Name):
                    return f"{v.id}()"
        return ""
    return ast.get_source_segment(src, node) or ""


def _comment(lines: list[str], start: int, end: int) -> str:
    for ln in range(end, start - 1, -1):
        text = lines[ln - 1]
        if "#:" in text:
            parts = [text.split("#:", 1)[1].strip()]
            # a long description continues on the following lines that hold only a "#:" comment
            nxt = end
            while nxt < len(lines) and lines[nxt].strip().startswith("#:"):
                parts.append(lines[nxt].strip()[2:].strip())
                nxt += 1
            return " ".join(p for p in parts if p)
    return ""


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def build_reference() -> str:
    src = CONFIG.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    out: list[str] = []
    for name, prefix in SECTIONS:
        cls = classes[name]
        doc = ast.get_docstring(cls) or ""
        out.append(f"### `{name}`" + (f" (`{prefix[:-1]}`)" if prefix else "") + f" {{#{name.lower()}}}")
        out.append("")
        if doc:
            out.append(doc.strip().splitlines()[0])
            out.append("")
        out.append("| Key | Type | Default | Description |")
        out.append("|---|---|---|---|")
        for node in cls.body:
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            key = node.target.id
            ann = ast.get_source_segment(src, node.annotation) or ""
            if ann.endswith("Config"):
                desc = f"see [`{ann}`](#{ann.lower()})"
                default = f"`{ann}()`"
            else:
                desc = _comment(lines, node.lineno, node.end_lineno or node.lineno)
                default = f"`{_cell(_default(node.value, src))}`"
            out.append(f"| `{prefix}{key}` | `{_cell(ann)}` | {default} | {_cell(desc)} |")
        out.append("")
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - MkDocs hook signature
    if MARKER in markdown:
        markdown = markdown.replace(MARKER, build_reference())
    return markdown
