"""Rendering evaluation output for terminals and browsers."""

from .html import (
    render_eval_html,
    render_sweep_html,
    write_eval_report,
    write_report,
    write_sweep_report,
)
from .terminal import (
    Painter,
    force_utf8_output,
    render_eval,
    render_hits,
    render_sweep,
    render_table,
    supports_color,
)

__all__ = [
    "Painter",
    "force_utf8_output",
    "render_eval",
    "render_eval_html",
    "render_hits",
    "render_sweep",
    "render_sweep_html",
    "render_table",
    "supports_color",
    "write_eval_report",
    "write_report",
    "write_sweep_report",
]
