"""Terminal rendering: plain-text tables with optional ANSI colour."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from ..eval.harness import EvalResult
from ..eval.sweep import SweepReport

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def supports_color(stream=None) -> bool:
    """Honour NO_COLOR and skip escapes when output is piped to a file."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return hasattr(stream, "isatty") and stream.isatty()


class Painter:
    """Wraps text in ANSI codes, or does nothing when colour is off."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = supports_color() if enabled is None else enabled

    def __call__(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return "".join(codes) + text + RESET

    def score(self, value: float) -> str:
        """Colour a 0–1 score by band, so a leaderboard scans at a glance."""
        text = f"{value:.3f}"
        if not self.enabled:
            return text
        if value >= 0.8:
            return self(text, GREEN)
        if value >= 0.5:
            return self(text, YELLOW)
        return self(text, RED)


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    align_right: Sequence[int] | None = None,
) -> str:
    """Fixed-width table. Right-aligns the columns named in ``align_right``."""
    if not rows:
        return "(no rows)"
    align_right = set(align_right or [])
    columns = len(headers)
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for i in range(columns):
            widths[i] = max(widths[i], len(_visible(str(row[i]))))

    def format_row(values: Sequence[str]) -> str:
        cells = []
        for i, value in enumerate(values):
            text = str(value)
            padding = widths[i] - len(_visible(text))
            cells.append(" " * padding + text if i in align_right else text + " " * padding)
        return "  ".join(cells).rstrip()

    separator = "  ".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def _visible(text: str) -> str:
    """Length of a string ignoring ANSI escapes, so colour does not break alignment."""
    out: list[str] = []
    skipping = False
    for char in text:
        if char == "\033":
            skipping = True
            continue
        if skipping:
            if char.isalpha():
                skipping = False
            continue
        out.append(char)
    return "".join(out)


def render_eval(
    result: EvalResult,
    ks: Sequence[int] = (1, 3, 5, 10),
    color: bool | None = None,
) -> str:
    """Full report for a single evaluation run."""
    paint = Painter(color)
    lines = [
        paint(f"Evaluation — {result.label}", BOLD, CYAN),
        paint(
            f"{result.n_queries} queries · {result.n_chunks} chunks · "
            f"indexed in {result.index_seconds:.2f}s",
            DIM,
        ),
        "",
    ]

    headers = ["k", "recall", "precision", "hit rate", "nDCG"]
    rows = []
    for k in ks:
        if f"recall@{k}" not in result.metrics:
            continue
        get = result.metrics.get
        rows.append(
            [
                str(k),
                paint.score(get(f"recall@{k}", 0.0)),
                paint.score(get(f"precision@{k}", 0.0)),
                paint.score(get(f"hit@{k}", 0.0)),
                paint.score(get(f"ndcg@{k}", 0.0)),
            ]
        )
    lines.append(render_table(headers, rows, align_right=range(1, 5)))
    lines.append("")

    low, high = result.confidence_interval("recall@5")
    lines.append(
        f"MRR {result.metrics.get('mrr', 0):.3f}   "
        f"MAP {result.metrics.get('map', 0):.3f}   "
        f"recall@5 95% CI [{low:.3f}, {high:.3f}]"
    )
    lines.append(
        paint(
            f"latency  p50 {result.latency_ms.get('p50', 0):.1f}ms · "
            f"p95 {result.latency_ms.get('p95', 0):.1f}ms · "
            f"max {result.latency_ms.get('max', 0):.1f}ms",
            DIM,
        )
    )

    misses = result.failures("mrr", 0.0)
    if misses:
        lines.append("")
        lines.append(paint(f"{len(misses)} queries retrieved nothing relevant:", RED))
        for score in misses[:5]:
            lines.append(f"  · {score.query}")
            lines.append(paint(f"      expected {', '.join(score.relevant[:3])}", DIM))
        if len(misses) > 5:
            lines.append(paint(f"  … and {len(misses) - 5} more", DIM))
    return "\n".join(lines)


def render_sweep(report: SweepReport, color: bool | None = None) -> str:
    """Leaderboard across every configuration in a sweep."""
    paint = Painter(color)
    ranked = report.ranked
    lines = [
        paint(f"Sweep leaderboard — ranked by {report.primary}", BOLD, CYAN),
        paint(f"{len(ranked)} configurations · {report.n_queries} queries", DIM),
        "",
    ]

    headers = ["#", "configuration", report.primary, "mrr", "ndcg@10", "p95 ms", "chunks"]
    rows = []
    for position, row in enumerate(ranked, start=1):
        if row.error:
            rows.append([str(position), row.label, paint("error", RED), "-", "-", "-", "-"])
            continue
        rows.append(
            [
                str(position),
                paint(row.label, BOLD) if position == 1 else row.label,
                paint.score(row.get(report.primary)),
                f"{row.get('mrr'):.3f}",
                f"{row.get('ndcg@10'):.3f}",
                f"{row.latency_ms.get('p95', 0):.1f}",
                str(row.n_chunks),
            ]
        )
    lines.append(render_table(headers, rows, align_right=[0, 2, 3, 4, 5, 6]))

    broken = [row for row in ranked if row.error]
    if broken:
        lines.append("")
        lines.append(paint("failed configurations:", RED))
        for row in broken:
            lines.append(f"  · {row.label}: {row.error}")

    best = report.best
    if best:
        lines.append("")
        lines.append(
            paint(f"best: {best.label} — {report.primary} {best.get(report.primary):.3f}", GREEN)
        )
    return "\n".join(lines)


def render_hits(query: str, hits, color: bool | None = None, width: int = 240) -> str:
    """Human-readable retrieval output for ``ragforge query``."""
    paint = Painter(color)
    lines = [paint(f'query: "{query}"', BOLD, CYAN), ""]
    if not hits:
        return "\n".join(lines + [paint("no results", RED)])
    for hit in hits:
        snippet = " ".join(hit.text.split())
        if len(snippet) > width:
            snippet = snippet[:width].rstrip() + "…"
        detail = ""
        if hit.components:
            detail = "  " + " ".join(f"{k}={v:.3f}" for k, v in hit.components.items())
        lines.append(
            f"{paint(f'{hit.rank:>2}.', BOLD)} {paint(f'{hit.score:.4f}', GREEN)} "
            f"{paint(hit.doc_id, CYAN)}{paint(detail, DIM)}"
        )
        lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines)
