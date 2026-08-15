"""End-to-end tour of the ragforge API.

    python examples/quickstart.py

Runs on the core install — numpy only, no model downloads, no network.
"""

from __future__ import annotations

import os

from ragforge import PipelineConfig, RagPipeline, evaluate, load_documents, load_queries
from ragforge.config import expand_grid
from ragforge.eval import compare, run_sweep
from ragforge.report.terminal import force_utf8_output, render_eval, render_sweep

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
EVALSET = os.path.join(HERE, "evalset.jsonl")


def main() -> None:
    force_utf8_output()
    documents = load_documents(CORPUS)
    queries = load_queries(EVALSET)
    print(f"corpus: {len(documents)} documents · evalset: {len(queries)} queries\n")

    # 1. Index and search -----------------------------------------------------
    pipeline = RagPipeline.from_config(
        PipelineConfig(chunker="sentence", retriever="hybrid")
    )
    pipeline.index(documents)
    print(pipeline.describe())

    for hit in pipeline.search("how do I rotate an API key?", k=3):
        print(f"  {hit.rank}. {hit.doc_id:<24} {hit.score:.4f}")

    # Dense retrieval alone tends to miss exact identifiers; hybrid does not.
    print("\nexact identifier lookup — ERR_4021:")
    for hit in pipeline.search("ERR_4021", k=2):
        print(f"  {hit.rank}. {hit.doc_id:<24} {hit.score:.4f}")

    # 2. Measure it -----------------------------------------------------------
    print("\n" + "=" * 78)
    hybrid_result = evaluate(pipeline, queries, ks=(1, 3, 5, 10))
    print(render_eval(hybrid_result, ks=(1, 3, 5, 10)))

    # 3. Compare against a baseline ------------------------------------------
    baseline = RagPipeline.from_config(PipelineConfig(chunker="sentence", retriever="dense"))
    baseline.index(documents)
    dense_result = evaluate(baseline, queries, ks=(1, 3, 5, 10), label="dense only")

    print("\n" + "=" * 78)
    diff = compare(dense_result, hybrid_result, metric="recall@1")
    sign = "+" if diff["delta"] >= 0 else ""
    print(
        f"dense-only recall@1 {diff['baseline']:.3f} → hybrid {diff['candidate']:.3f} "
        f"({sign}{diff['delta']:.3f})"
    )
    print(
        f"{len(diff['improved'])} queries improved · {len(diff['regressed'])} regressed · "
        f"{diff['unchanged']} unchanged"
    )

    # 4. Sweep the configuration space ---------------------------------------
    print("\n" + "=" * 78)
    configs = expand_grid(
        PipelineConfig(embedder_args={"dim": 1024}),
        {"chunker": ["fixed", "sentence", "markdown"], "retriever": ["bm25", "dense", "hybrid"]},
    )
    report = run_sweep(configs, documents, queries, ks=(1, 3, 5, 10), primary="recall@1")
    print(render_sweep(report))


if __name__ == "__main__":
    main()
