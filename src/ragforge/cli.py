"""``ragforge`` command line interface.

    ragforge index   --corpus examples/corpus
    ragforge query   "how do I rotate an API key?"
    ragforge eval    --queries examples/evalset.jsonl --html report.html
    ragforge sweep   --grid examples/sweep.json --html sweep.html
    ragforge compare baseline.json candidate.json

``eval`` exits non-zero when ``--min-recall`` is not met, which is what makes it
usable as a retrieval regression gate in CI.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

from .config import PipelineConfig, load_grid
from .eval.harness import compare as compare_results
from .eval.harness import evaluate
from .eval.sweep import run_sweep
from .loaders import load_documents, load_queries, validate_eval_set
from .pipeline import RagPipeline
from .report.html import write_eval_report, write_sweep_report
from .report.terminal import render_eval, render_hits, render_sweep, render_table

DEFAULT_INDEX = ".ragforge"

EXIT_OK = 0
EXIT_THRESHOLD = 1
EXIT_USAGE = 2


def _parse_ks(text: str) -> list[int]:
    try:
        return sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--ks must be comma-separated integers: {exc}") from exc


def _parse_json(text: str | None, flag: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} is not valid JSON: {exc}") from exc


def _load_config(path: str | None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    config = PipelineConfig.from_file(path) if path else PipelineConfig()
    if overrides:
        config = config.merged(**{k: v for k, v in overrides.items() if v is not None})
    return config


def _build_indexed_pipeline(corpus: str, config: PipelineConfig) -> tuple:
    documents = load_documents(corpus)
    started = time.perf_counter()
    pipeline = RagPipeline.from_config(config)
    pipeline.index(documents)
    return pipeline, documents, time.perf_counter() - started


# --------------------------------------------------------------------- commands


def cmd_index(args: argparse.Namespace) -> int:
    config = _load_config(
        args.config,
        {"chunker": args.chunker, "embedder": args.embedder, "store": args.store,
         "retriever": args.retriever},
    )
    pipeline, documents, elapsed = _build_indexed_pipeline(args.corpus, config)
    pipeline.save(args.out)

    stats = pipeline.stats()
    print(render_table(
        ["metric", "value"],
        [[key, str(value)] for key, value in stats.items()],
    ))
    print(f"\nIndexed {len(documents)} documents in {elapsed:.2f}s → {args.out}")
    return EXIT_OK


def cmd_query(args: argparse.Namespace) -> int:
    pipeline = RagPipeline.load(args.index)
    where = _parse_json(args.where, "--where")
    result = pipeline.retrieve(args.query, k=args.k, where=where)

    if args.json:
        print(json.dumps(
            {
                "query": result.query,
                "latency_ms": round(result.latency_ms, 3),
                "hits": [
                    {
                        "rank": hit.rank,
                        "score": hit.score,
                        "chunk_id": hit.id,
                        "doc_id": hit.doc_id,
                        "text": hit.text,
                        "metadata": hit.chunk.metadata,
                    }
                    for hit in result.hits
                ],
            },
            indent=2,
            ensure_ascii=False,
        ))
        return EXIT_OK

    print(render_hits(result.query, result.hits))
    print(f"{len(result.hits)} hits in {result.latency_ms:.1f}ms")
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    queries = load_queries(args.queries)

    if args.corpus:
        config = _load_config(args.config, {"chunker": args.chunker, "retriever": args.retriever})
        documents = load_documents(args.corpus)
        problems = validate_eval_set(queries, documents)
        if problems:
            print("Evaluation set references documents that are not in the corpus:",
                  file=sys.stderr)
            for problem in problems[:10]:
                print(f"  · {problem}", file=sys.stderr)
            return EXIT_USAGE
        pipeline, _, index_seconds = _build_indexed_pipeline(args.corpus, config)
    else:
        pipeline = RagPipeline.load(args.index)
        index_seconds = 0.0

    result = evaluate(
        pipeline,
        queries,
        ks=args.ks,
        granularity=args.granularity,
        index_seconds=index_seconds,
    )

    print(render_eval(result, ks=args.ks))
    if args.html:
        write_eval_report(result, args.html, ks=args.ks)
        print(f"\nHTML report → {args.html}")
    if args.json:
        result.save(args.json)
        print(f"JSON results → {args.json}")

    if args.min_recall is not None:
        metric = f"recall@{args.gate_k}"
        actual = result.metrics.get(metric, 0.0)
        if actual + 1e-9 < args.min_recall:
            print(
                f"\nFAIL  {metric} {actual:.3f} is below the required {args.min_recall:.3f}",
                file=sys.stderr,
            )
            return EXIT_THRESHOLD
        print(f"\nPASS  {metric} {actual:.3f} ≥ {args.min_recall:.3f}")
    return EXIT_OK


def cmd_sweep(args: argparse.Namespace) -> int:
    documents = load_documents(args.corpus)
    queries = load_queries(args.queries)

    problems = validate_eval_set(queries, documents)
    if problems:
        print("Evaluation set references unknown documents:", file=sys.stderr)
        for problem in problems[:10]:
            print(f"  · {problem}", file=sys.stderr)
        return EXIT_USAGE

    if args.grid:
        configs = load_grid(args.grid)
    else:
        from .config import expand_grid
        from .eval.sweep import ablation_grid

        configs = expand_grid(PipelineConfig(), ablation_grid())

    def progress(position: int, total: int, label: str) -> None:
        if not args.quiet:
            print(f"[{position}/{total}] {label}", file=sys.stderr)

    report = run_sweep(
        configs,
        documents,
        queries,
        ks=args.ks,
        primary=args.primary,
        granularity=args.granularity,
        on_progress=progress,
    )

    print()
    print(render_sweep(report))
    if args.html:
        write_sweep_report(report, args.html)
        print(f"\nHTML leaderboard → {args.html}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        print(f"JSON leaderboard → {args.json}")
    if args.save_best and report.best:
        PipelineConfig.from_dict(report.best.config).to_json(args.save_best)
        print(f"Best config → {args.save_best}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    from .eval.harness import EvalResult, QueryScore

    def _read(path: str) -> EvalResult:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return EvalResult(
            label=payload.get("label", os.path.basename(path)),
            metrics=payload.get("metrics", {}),
            per_query=[
                QueryScore(
                    query_id=item["query_id"],
                    query=item.get("query", ""),
                    metrics=item.get("metrics", {}),
                    latency_ms=item.get("latency_ms", 0.0),
                    retrieved=item.get("retrieved", []),
                    relevant=item.get("relevant", []),
                )
                for item in payload.get("per_query", [])
            ],
            latency_ms=payload.get("latency_ms", {}),
            n_queries=payload.get("n_queries", 0),
            n_chunks=payload.get("n_chunks", 0),
        )

    baseline, candidate = _read(args.baseline), _read(args.candidate)
    diff = compare_results(baseline, candidate, metric=args.metric)

    print(render_table(
        ["", "run", args.metric],
        [
            ["baseline", baseline.label, f"{diff['baseline']:.4f}"],
            ["candidate", candidate.label, f"{diff['candidate']:.4f}"],
        ],
        align_right=[2],
    ))
    sign = "+" if diff["delta"] >= 0 else ""
    pct = f" ({sign}{diff['delta_pct']:.1f}%)" if diff["delta_pct"] is not None else ""
    print(f"\ndelta {sign}{diff['delta']:.4f}{pct}")
    print(
        f"{len(diff['improved'])} improved · {len(diff['regressed'])} regressed · "
        f"{diff['unchanged']} unchanged"
    )
    if diff["regressed"]:
        print("\nregressed queries:")
        for query_id in diff["regressed"][:10]:
            print(f"  · {query_id}")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(index_path=args.index, host=args.host, port=args.port)
    return EXIT_OK


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="Hybrid retrieval with a built-in evaluation harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version="ragforge 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    index_parser = subparsers.add_parser("index", help="chunk, embed and store a corpus")
    index_parser.add_argument("--corpus", required=True,
                              help="directory of text files, or a .jsonl")
    index_parser.add_argument("--config", help="pipeline config (.json/.yaml)")
    index_parser.add_argument("--out", default=DEFAULT_INDEX, help="where to write the index")
    index_parser.add_argument("--chunker", help="override the config's chunker")
    index_parser.add_argument("--embedder", help="override the config's embedder")
    index_parser.add_argument("--store", help="override the config's vector store")
    index_parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"])
    index_parser.set_defaults(func=cmd_index)

    # query
    query_parser = subparsers.add_parser("query", help="search an existing index")
    query_parser.add_argument("query")
    query_parser.add_argument("--index", default=DEFAULT_INDEX)
    query_parser.add_argument("-k", type=int, default=5, help="how many chunks to return")
    query_parser.add_argument("--where", help='metadata filter, e.g. \'{"source": "faq.md"}\'')
    query_parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    query_parser.set_defaults(func=cmd_query)

    # eval
    eval_parser = subparsers.add_parser("eval", help="score a pipeline against a query set")
    eval_parser.add_argument("--queries", required=True, help="evaluation set (.jsonl)")
    eval_parser.add_argument("--corpus", help="index this corpus fresh instead of loading --index")
    eval_parser.add_argument("--index", default=DEFAULT_INDEX)
    eval_parser.add_argument("--config", help="pipeline config used with --corpus")
    eval_parser.add_argument("--chunker")
    eval_parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"])
    eval_parser.add_argument("--ks", type=_parse_ks, default=[1, 3, 5, 10])
    eval_parser.add_argument("--granularity", choices=["doc", "chunk"], default="doc")
    eval_parser.add_argument("--html", help="write a self-contained HTML report here")
    eval_parser.add_argument("--json", help="write raw results here")
    eval_parser.add_argument(
        "--min-recall",
        type=float,
        help="fail with exit code 1 if recall falls below this (for CI gating)",
    )
    eval_parser.add_argument("--gate-k", type=int, default=5, help="cut-off used by --min-recall")
    eval_parser.set_defaults(func=cmd_eval)

    # sweep
    sweep_parser = subparsers.add_parser("sweep", help="grid-search configurations")
    sweep_parser.add_argument("--corpus", required=True)
    sweep_parser.add_argument("--queries", required=True)
    sweep_parser.add_argument("--grid", help="sweep definition; omit for the default ablation")
    sweep_parser.add_argument("--ks", type=_parse_ks, default=[1, 3, 5, 10])
    sweep_parser.add_argument("--primary", default="recall@5", help="metric to rank by")
    sweep_parser.add_argument("--granularity", choices=["doc", "chunk"], default="doc")
    sweep_parser.add_argument("--html")
    sweep_parser.add_argument("--json")
    sweep_parser.add_argument("--save-best", help="write the winning config to this path")
    sweep_parser.add_argument("--quiet", action="store_true")
    sweep_parser.set_defaults(func=cmd_sweep)

    # compare
    compare_parser = subparsers.add_parser("compare", help="diff two eval JSON files")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--metric", default="recall@5")
    compare_parser.set_defaults(func=cmd_compare)

    # serve
    serve_parser = subparsers.add_parser("serve", help="expose an index over HTTP")
    serve_parser.add_argument("--index", default=DEFAULT_INDEX)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def _force_utf8_output() -> None:
    """Stop a legacy console encoding from turning output into a crash.

    Windows terminals still default to cp1252, which cannot encode the ``≥``, ``·``
    and ``—`` this CLI prints. Worse, ``UnicodeEncodeError`` subclasses ``ValueError``,
    so without this it was caught below and reported as a usage error — turning a
    cosmetic problem into a non-zero exit and a red CI run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # Detached or already-closed streams raise; never fail a run over output setup.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ImportError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
