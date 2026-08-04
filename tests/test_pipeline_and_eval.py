from __future__ import annotations

import json

import pytest

from ragforge.config import PipelineConfig, expand_grid, load_grid
from ragforge.eval import compare, evaluate, run_sweep
from ragforge.loaders import load_documents, load_queries, validate_eval_set
from ragforge.pipeline import RagPipeline
from ragforge.report.html import render_eval_html, render_sweep_html
from ragforge.report.terminal import render_eval, render_sweep, render_table
from ragforge.types import Query

# ---------------------------------------------------------------------- config


def test_config_round_trips_through_json(tmp_path):
    config = PipelineConfig(chunker="fixed", chunker_args={"size": 128}, retriever="dense")
    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    assert PipelineConfig.from_file(str(path)) == config


def test_config_rejects_an_unknown_retriever():
    with pytest.raises(ValueError, match="retriever must be one of"):
        PipelineConfig(retriever="teleport")


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown config keys"):
        PipelineConfig.from_dict({"chunker": "fixed", "chunkr": "typo"})


def test_config_label_captures_the_settings_that_matter():
    label = PipelineConfig(
        chunker="fixed", chunker_args={"size": 256}, retriever="hybrid", rerank="mmr"
    ).label
    assert "fixed@256" in label and "hybrid" in label and "+mmr" in label


def test_expand_grid_produces_the_cartesian_product():
    configs = expand_grid(PipelineConfig(), {"chunker": ["fixed", "sentence"],
                                             "retriever": ["dense", "bm25"]})
    assert len(configs) == 4
    assert len({config.label for config in configs}) == 4


def test_load_grid_reads_a_base_and_grid(tmp_path):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps({"base": {"embedder": "hashing"},
                                "grid": {"retriever": ["dense", "bm25"]}}), encoding="utf-8")
    configs = load_grid(str(path))
    assert {config.retriever for config in configs} == {"dense", "bm25"}


# --------------------------------------------------------------------- loaders


def test_load_documents_reads_front_matter(corpus_dir):
    documents = load_documents(str(corpus_dir))
    assert len(documents) == 5
    auth = next(d for d in documents if d.id == "auth")
    assert auth.metadata["category"] == "test"
    assert not auth.text.startswith("---")


def test_load_documents_uses_stable_ids_from_filenames(corpus_dir):
    ids = {document.id for document in load_documents(str(corpus_dir))}
    assert "auth" in ids and "billing" in ids


def test_load_documents_rejects_an_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No documents found"):
        load_documents(str(tmp_path / "empty"))


def test_load_documents_reads_jsonl(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"id": "a", "text": "first doc", "team": "x"}\n{"id": "b", "text": "second doc"}\n',
        encoding="utf-8",
    )
    documents = load_documents(str(path))
    assert [d.id for d in documents] == ["a", "b"]
    assert documents[0].metadata["team"] == "x"


def test_load_queries_rejects_unjudged_queries(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "q1", "query": "no judgements"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no relevance judgements"):
        load_queries(str(path))


def test_validate_eval_set_flags_unknown_document_ids(documents):
    problems = validate_eval_set([Query(id="q", text="t", grades={"ghost": 1.0})], documents)
    assert problems and "ghost" in problems[0]


def test_validate_eval_set_passes_for_a_good_set(documents, queries):
    assert validate_eval_set(queries, documents) == []


# -------------------------------------------------------------------- pipeline


@pytest.mark.parametrize("retriever", ["dense", "bm25", "hybrid"])
def test_pipeline_indexes_and_retrieves(documents, retriever):
    pipeline = RagPipeline.from_config(PipelineConfig(retriever=retriever))
    pipeline.index(documents)
    hits = pipeline.search("how do I rotate an api key", k=3)
    assert hits and hits[0].doc_id == "auth"


def test_pipeline_reports_latency(documents):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    assert pipeline.retrieve("api key", k=2).latency_ms > 0


def test_pipeline_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="empty document list"):
        RagPipeline.from_config(PipelineConfig()).index([])


def test_retrieving_before_indexing_is_an_error():
    with pytest.raises(RuntimeError, match="not indexed yet"):
        RagPipeline.from_config(PipelineConfig()).retrieve("anything")


def test_pipeline_save_and_load_preserves_results(documents, tmp_path):
    original = RagPipeline.from_config(PipelineConfig())
    original.index(documents)
    before = [hit.doc_id for hit in original.search("rotate api key", k=3)]
    original.save(str(tmp_path / "idx"))

    reloaded = RagPipeline.load(str(tmp_path / "idx"))
    assert [hit.doc_id for hit in reloaded.search("rotate api key", k=3)] == before
    assert reloaded.stats()["chunks"] == original.stats()["chunks"]


def test_loading_a_missing_index_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="ragforge index"):
        RagPipeline.load(str(tmp_path / "nothing"))


def test_indexing_is_idempotent(documents):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    count = pipeline.store.count()
    pipeline.index(documents)
    assert pipeline.store.count() == count


def test_context_helper_joins_the_top_hits(documents):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    context = pipeline.retrieve("api key", k=2).as_context()
    assert "---" in context


def test_metadata_filter_flows_through_the_pipeline(corpus_dir):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(load_documents(str(corpus_dir)))
    hits = pipeline.search("api key", k=5, where={"source": "auth.md"})
    assert hits and all(hit.chunk.metadata["source"] == "auth.md" for hit in hits)


# ------------------------------------------------------------------------ eval


def test_evaluate_produces_all_expected_metrics(documents, queries):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    result = evaluate(pipeline, queries, ks=(1, 3))

    for name in ["recall@1", "recall@3", "precision@1", "hit@3", "ndcg@3", "mrr", "map"]:
        assert name in result.metrics
        assert 0.0 <= result.metrics[name] <= 1.0
    assert result.n_queries == len(queries)
    assert len(result.per_query) == len(queries)


def test_recall_is_monotonic_in_k(documents, queries):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    result = evaluate(pipeline, queries, ks=(1, 3, 5))
    assert result.metrics["recall@1"] <= result.metrics["recall@3"] <= result.metrics["recall@5"]


def test_this_corpus_is_retrievable_at_all(documents, queries):
    """A smoke check that the defaults actually work — if this drops, something
    real broke rather than a metric being mis-specified."""
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    assert evaluate(pipeline, queries, ks=(1, 3)).metrics["recall@3"] >= 0.8


def test_evaluate_rejects_an_empty_query_set(documents):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    with pytest.raises(ValueError, match="empty query set"):
        evaluate(pipeline, [])


def test_eval_result_serialises(documents, queries, tmp_path):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    result = evaluate(pipeline, queries, ks=(1, 3))
    path = tmp_path / "result.json"
    result.save(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["recall@1"] == pytest.approx(result.metrics["recall@1"])


def test_confidence_interval_brackets_the_mean(documents, queries):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    result = evaluate(pipeline, queries, ks=(1, 5))
    low, high = result.confidence_interval("recall@5")
    assert low <= result.metrics["recall@5"] <= high


def test_compare_counts_improvements_and_regressions(documents, queries):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    baseline = evaluate(pipeline, queries, ks=(1, 5), label="base")
    diff = compare(baseline, baseline, metric="recall@5")
    assert diff["delta"] == 0.0
    assert diff["improved"] == [] and diff["regressed"] == []


# ----------------------------------------------------------------------- sweep


def test_sweep_ranks_every_configuration(documents, queries):
    configs = expand_grid(PipelineConfig(), {"retriever": ["dense", "bm25", "hybrid"]})
    report = run_sweep(configs, documents, queries, ks=(1, 3), primary="recall@3")
    assert len(report.rows) == 3
    ranked = report.ranked
    assert all(
        a.get("recall@3") >= b.get("recall@3") for a, b in zip(ranked, ranked[1:])
    )
    assert report.best is not None


def test_a_broken_config_does_not_abort_the_sweep(documents, queries):
    good = PipelineConfig(retriever="bm25")
    broken = PipelineConfig(retriever="dense", embedder_args={"dim": -1})
    report = run_sweep([good, broken], documents, queries, ks=(1, 3), primary="recall@3")
    assert len(report.rows) == 2
    assert sum(row.error is not None for row in report.rows) == 1
    assert report.best is not None  # the healthy config still wins


def test_sweep_rejects_an_empty_config_list(documents, queries):
    with pytest.raises(ValueError, match="configs is empty"):
        run_sweep([], documents, queries)


# --------------------------------------------------------------------- reports


def test_terminal_and_html_reports_render(documents, queries):
    pipeline = RagPipeline.from_config(PipelineConfig())
    pipeline.index(documents)
    result = evaluate(pipeline, queries, ks=(1, 3, 5))

    text = render_eval(result, ks=(1, 3, 5), color=False)
    assert "recall" in text and "\x1b[" not in text

    html = render_eval_html(result, ks=(1, 3, 5))
    assert html.startswith("<!doctype html>")
    assert "prefers-color-scheme" in html
    assert "http://" not in html.replace("http://www.w3.org", "")  # no external assets

    report = run_sweep([PipelineConfig(retriever="bm25")], documents, queries, ks=(1, 3),
                       primary="recall@3")
    assert "leaderboard" in render_sweep(report, color=False).lower()
    assert render_sweep_html(report).startswith("<!doctype html>")


def test_html_escapes_untrusted_document_text(documents):
    from ragforge.eval.harness import EvalResult, QueryScore

    result = EvalResult(
        label="<script>alert(1)</script>",
        metrics={"recall@5": 0.5, "mrr": 0.5, "ndcg@10": 0.5},
        per_query=[QueryScore("q", "<img onerror=x>", {"mrr": 0.0}, 1.0, [], ["a"])],
        latency_ms={"p95": 1.0},
    )
    html = render_eval_html(result)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_table_alignment_survives_ansi_codes():
    table = render_table(["a", "b"], [["\033[32mx\033[0m", "yy"]], align_right=[1])
    assert len(table.splitlines()) == 3
