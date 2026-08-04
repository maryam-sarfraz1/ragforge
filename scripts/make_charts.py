"""Regenerate the README figures from a real evaluation run.

    python scripts/make_charts.py            # core install only (3 configs)
    python scripts/make_charts.py --encoders # adds MiniLM rows, needs [st]

Every number in docs/*.svg comes from an actual sweep over examples/corpus, so the
figures cannot drift away from the code. Light and dark variants are emitted for
each figure and paired with <picture> in the README.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ragforge import PipelineConfig, load_documents, load_queries  # noqa: E402
from ragforge.config import expand_grid  # noqa: E402
from ragforge.eval import run_sweep  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")

# Theme parameters — the only values that change between the two renders.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2df",
        "series": ["#2a78d6", "#eb6834"],
        "ramp": ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "series": ["#3987e5", "#d95926"],
        "ramp": ["#cde2fb", "#9ec5f4", "#5598e7", "#3987e5", "#1c5cab", "#0d366b"],
    },
}

MINILM = {"model": "sentence-transformers/all-MiniLM-L6-v2"}
BASE = {"chunker": "sentence", "chunker_args": {"size": 180, "overlap": 40}}


def _style(theme: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["text"],
            "axes.labelcolor": theme["muted"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def _save(fig, name: str, mode: str) -> None:
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, f"{name}-{mode}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


# --------------------------------------------------------------------- figures


def figure_recall(rows, mode: str) -> None:
    """Grouped horizontal bars: recall@1 against recall@5 per configuration."""
    theme = THEMES[mode]
    _style(theme)
    labels = [row["label"] for row in rows]
    r1 = [row["recall@1"] for row in rows]
    r5 = [row["recall@5"] for row in rows]

    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(rows) + 1.5))
    positions = range(len(rows))
    height = 0.36
    gap = 0.035  # 2px surface gap between the paired bars

    ax.barh([p + height / 2 + gap for p in positions], r1, height=height,
            color=theme["series"][0], label="recall@1", zorder=3)
    ax.barh([p - height / 2 - gap for p in positions], r5, height=height,
            color=theme["series"][1], label="recall@5", zorder=3)

    for position, (a, b) in enumerate(zip(r1, r5)):
        ax.text(a + 0.008, position + height / 2 + gap, f"{a:.3f}", va="center",
                ha="left", fontsize=8.5, color=theme["muted"])
        ax.text(b + 0.008, position - height / 2 - gap, f"{b:.3f}", va="center",
                ha="left", fontsize=8.5, color=theme["muted"])

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9.5, color=theme["text"])
    ax.set_xlim(0, 1.09)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("share of relevant documents retrieved", fontsize=9)
    ax.set_title("Retrieval quality by configuration", fontsize=12.5, color=theme["text"],
                 loc="left", pad=34, fontweight="bold")
    # Legend sits above the plot area — at lower right it collided with the last row.
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False,
                       fontsize=9, ncols=2, handlelength=1.2, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(theme["muted"])
    _save(fig, "recall", mode)


def figure_tradeoff(rows, mode: str) -> None:
    """Quality against cost. One hue, every point directly labelled."""
    theme = THEMES[mode]
    _style(theme)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for row in rows:
        ax.scatter(row["p95"], row["recall@1"], s=150, color=theme["series"][0],
                   edgecolor=theme["surface"], linewidth=2.0, zorder=3)
        ax.annotate(
            row["label"],
            (row["p95"], row["recall@1"]),
            textcoords="offset points",
            xytext=row.get("offset", (12, 6)),
            ha=row.get("ha", "left"),
            fontsize=9,
            color=theme["text"],
        )

    ax.set_xscale("log")
    # Log-scale headroom on both sides so direct labels have somewhere to sit.
    fastest = min(row["p95"] for row in rows)
    slowest = max(row["p95"] for row in rows)
    ax.set_xlim(fastest / 3.0, slowest * 3.0)
    span = max(row["recall@1"] for row in rows) - min(row["recall@1"] for row in rows)
    ax.set_ylim(
        min(row["recall@1"] for row in rows) - span * 0.22,
        max(row["recall@1"] for row in rows) + span * 0.22,
    )
    ax.set_xlabel("p95 latency per query (ms, log scale) — lower is better", fontsize=9)
    ax.set_ylabel("recall@1 — higher is better", fontsize=9)
    ax.grid(True, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Quality costs latency — how much is the question", fontsize=12.5,
                 color=theme["text"], loc="left", pad=14, fontweight="bold")
    _save(fig, "tradeoff", mode)


def figure_heatmap(grid_rows, chunkers, retrievers, mode: str) -> None:
    """Sequential heatmap: recall@1 for every chunker × retriever pairing."""
    theme = THEMES[mode]
    _style(theme)
    cmap = LinearSegmentedColormap.from_list("seq", theme["ramp"])

    matrix = [[grid_rows[(chunker, retriever)] for retriever in retrievers]
              for chunker in chunkers]
    low = min(min(row) for row in matrix)
    high = max(max(row) for row in matrix)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bottom, top = low - 0.02, high + 0.005
    image = ax.imshow(matrix, cmap=cmap, vmin=bottom, vmax=top, aspect="auto")

    def _luminance(rgb) -> float:
        channels = [
            c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]
        ]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    for i, _ in enumerate(chunkers):
        for j, _ in enumerate(retrievers):
            value = matrix[i][j]
            # Pick the ink by measured contrast against the cell, not by a guessed
            # threshold — otherwise neighbouring cells of similar value disagree.
            cell = cmap((value - bottom) / max(top - bottom, 1e-9))
            luminance = _luminance(cell)
            on_white = 1.05 / (luminance + 0.05)
            on_black = (luminance + 0.05) / 0.05
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10.5,
                    color="#0b0b0b" if on_black >= on_white else "#ffffff")

    # 2px surface gap between cells.
    for edge in range(1, len(retrievers)):
        ax.axvline(edge - 0.5, color=theme["surface"], linewidth=2.5)
    for edge in range(1, len(chunkers)):
        ax.axhline(edge - 0.5, color=theme["surface"], linewidth=2.5)

    ax.set_xticks(range(len(retrievers)), retrievers, fontsize=10, color=theme["text"])
    ax.set_yticks(range(len(chunkers)), chunkers, fontsize=10, color=theme["text"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("recall@1 across the configuration grid", fontsize=12.5,
                 color=theme["text"], loc="left", pad=14, fontweight="bold")
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8.5, colors=theme["muted"])
    _save(fig, "heatmap", mode)


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", action="store_true",
                        help="include sentence-transformers rows (needs the [st] extra)")
    args = parser.parse_args()

    documents = load_documents(os.path.join(ROOT, "examples", "corpus"))
    queries = load_queries(os.path.join(ROOT, "examples", "evalset.jsonl"))
    print(f"corpus {len(documents)} docs · evalset {len(queries)} queries")

    named = [
        ("BM25 only", PipelineConfig(**BASE, retriever="bm25")),
        ("Dense (hashing)",
         PipelineConfig(**BASE, retriever="dense", embedder_args={"dim": 1024})),
        ("Hybrid RRF (hashing)",
         PipelineConfig(**BASE, retriever="hybrid", embedder_args={"dim": 1024})),
    ]
    if args.encoders:
        named += [
            ("Dense (MiniLM-L6)",
             PipelineConfig(**BASE, retriever="dense", embedder="sentence-transformer",
                            embedder_args=MINILM, fit_embedder=False)),
            ("Hybrid RRF (MiniLM-L6)",
             PipelineConfig(**BASE, retriever="hybrid", embedder="sentence-transformer",
                            embedder_args=MINILM, fit_embedder=False)),
            ("Hybrid + MMR (MiniLM-L6)",
             PipelineConfig(**BASE, retriever="hybrid", embedder="sentence-transformer",
                            embedder_args=MINILM, fit_embedder=False, rerank="mmr")),
        ]

    print("\nrunning the headline comparison…")
    report = run_sweep([config for _, config in named], documents, queries,
                       ks=(1, 3, 5, 10), primary="recall@1")
    rows = [
        {
            "label": name,
            "recall@1": row.get("recall@1"),
            "recall@3": row.get("recall@3"),
            "recall@5": row.get("recall@5"),
            "mrr": row.get("mrr"),
            "ndcg@10": row.get("ndcg@10"),
            "p95": row.latency_ms.get("p95", 0.0),
        }
        for (name, _), row in zip(named, report.rows)
    ]
    for row in rows:
        print(f"  {row['label']:26} r@1 {row['recall@1']:.3f}  r@5 {row['recall@5']:.3f}"
              f"  p95 {row['p95']:.1f}ms")

    # Emit the same numbers as markdown so the README table cannot drift from the
    # figures — both come from this one run.
    os.makedirs(DOCS, exist_ok=True)
    columns = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@10"]
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as handle:
        handle.write(f"<!-- generated by scripts/make_charts.py · {len(documents)} docs · "
                     f"{len(queries)} queries -->\n\n")
        handle.write("| Configuration | " + " | ".join(columns) + " | p95 latency |\n")
        handle.write("|---" * (len(columns) + 2) + "|\n")
        for row in rows:
            cells = " | ".join(f"{row[column]:.3f}" for column in columns)
            handle.write(f"| {row['label']} | {cells} | {row['p95']:.1f} ms |\n")
    print(f"  wrote {os.path.join('docs', 'results.md')}")

    # Hand-placed labels: the three MiniLM points cluster tightly, and the two at
    # identical recall@1 would otherwise print on top of one another.
    placements = [
        ((10, -16), "left"),    # BM25 only
        ((10, 10), "left"),     # Dense (hashing)
        ((10, -16), "left"),    # Hybrid RRF (hashing)
        ((-11, 9), "right"),    # Dense (MiniLM)
        ((10, -17), "left"),    # Hybrid RRF (MiniLM)
        ((10, 9), "left"),      # Hybrid + MMR (MiniLM)
    ]
    for row, (offset, align) in zip(rows, placements):
        row["offset"], row["ha"] = offset, align

    print("\nrunning the chunker × retriever grid…")
    chunkers = ["fixed", "sentence", "recursive", "markdown"]
    retrievers = ["bm25", "dense", "hybrid"]
    grid_configs = expand_grid(
        PipelineConfig(embedder_args={"dim": 1024}),
        {"chunker": chunkers, "retriever": retrievers},
    )
    grid_report = run_sweep(grid_configs, documents, queries, ks=(1, 3, 5, 10),
                            primary="recall@1")
    grid_rows = {
        (config.chunker, config.retriever): row.get("recall@1")
        for config, row in zip(grid_configs, grid_report.rows)
    }

    print("\nrendering figures…")
    for mode in ("light", "dark"):
        figure_recall(list(reversed(rows)), mode)
        figure_tradeoff(rows, mode)
        figure_heatmap(grid_rows, chunkers, retrievers, mode)
    print("\ndone")


if __name__ == "__main__":
    main()
