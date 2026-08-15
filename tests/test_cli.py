"""End-to-end CLI tests, driven through ``main()`` so exit codes are covered."""

from __future__ import annotations

import json
import sys

import pytest

from ragforge.cli import EXIT_OK, EXIT_THRESHOLD, EXIT_USAGE, main


@pytest.fixture
def indexed(tmp_path, corpus_dir):
    index = tmp_path / "idx"
    assert main(["index", "--corpus", str(corpus_dir), "--out", str(index)]) == EXIT_OK
    return index


def test_index_writes_a_loadable_index(indexed):
    assert (indexed / "pipeline.json").exists()
    assert (indexed / "vectors.npz").exists()


def test_index_reports_stats(tmp_path, corpus_dir, capsys):
    main(["index", "--corpus", str(corpus_dir), "--out", str(tmp_path / "i2")])
    out = capsys.readouterr().out
    assert "chunks" in out and "Indexed 5 documents" in out


def test_query_returns_the_right_document(indexed, capsys):
    assert main(["query", "how do I rotate an api key", "--index", str(indexed)]) == EXIT_OK
    assert "auth" in capsys.readouterr().out


def test_query_json_output_is_valid(indexed, capsys):
    assert main(["query", "webhook signature", "--index", str(indexed), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"] and payload["hits"][0]["rank"] == 1
    assert payload["latency_ms"] >= 0


def test_query_honours_a_metadata_filter(indexed, capsys):
    code = main(["query", "api key", "--index", str(indexed), "--json",
                 "--where", '{"source": "billing.md"}'])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert all(hit["metadata"]["source"] == "billing.md" for hit in payload["hits"])


def test_query_rejects_malformed_filter_json(indexed):
    with pytest.raises(SystemExit, match="not valid JSON"):
        main(["query", "x", "--index", str(indexed), "--where", "{oops"])


def test_eval_against_a_saved_index(indexed, evalset_file, capsys):
    assert main(["eval", "--queries", str(evalset_file), "--index", str(indexed)]) == EXIT_OK
    assert "recall" in capsys.readouterr().out


def test_eval_writes_html_and_json(indexed, evalset_file, tmp_path):
    html, results = tmp_path / "r.html", tmp_path / "r.json"
    code = main(["eval", "--queries", str(evalset_file), "--index", str(indexed),
                 "--html", str(html), "--json", str(results)])
    assert code == EXIT_OK
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "metrics" in json.loads(results.read_text(encoding="utf-8"))


def test_eval_gate_passes_when_recall_is_met(indexed, evalset_file):
    code = main(["eval", "--queries", str(evalset_file), "--index", str(indexed),
                 "--min-recall", "0.5", "--gate-k", "3"])
    assert code == EXIT_OK


def test_eval_gate_fails_below_the_threshold(indexed, evalset_file):
    """The CI regression gate: an impossible bar must produce a non-zero exit."""
    code = main(["eval", "--queries", str(evalset_file), "--index", str(indexed),
                 "--min-recall", "1.01", "--gate-k", "1"])
    assert code == EXIT_THRESHOLD


def test_eval_detects_judgements_pointing_at_missing_documents(corpus_dir, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "q", "query": "x", "grades": {"ghost": 1.0}}) + "\n",
                   encoding="utf-8")
    assert main(["eval", "--queries", str(bad), "--corpus", str(corpus_dir)]) == EXIT_USAGE


def test_sweep_ranks_configurations(corpus_dir, evalset_file, tmp_path, capsys):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"grid": {"retriever": ["bm25", "hybrid"]}}), encoding="utf-8")
    best = tmp_path / "best.json"
    code = main(["sweep", "--corpus", str(corpus_dir), "--queries", str(evalset_file),
                 "--grid", str(grid), "--primary", "recall@3", "--ks", "1,3",
                 "--html", str(tmp_path / "s.html"), "--save-best", str(best), "--quiet"])
    assert code == EXIT_OK
    assert "leaderboard" in capsys.readouterr().out.lower()
    assert (tmp_path / "s.html").exists()
    assert "retriever" in json.loads(best.read_text(encoding="utf-8"))


def test_compare_diffs_two_result_files(indexed, evalset_file, tmp_path, capsys):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for path in (first, second):
        main(["eval", "--queries", str(evalset_file), "--index", str(indexed), "--json", str(path)])
    assert main(["compare", str(first), str(second), "--metric", "recall@5"]) == EXIT_OK
    assert "delta" in capsys.readouterr().out


def test_missing_corpus_exits_with_a_usage_code(tmp_path):
    assert main(["index", "--corpus", str(tmp_path / "nope")]) == EXIT_USAGE


def test_bad_ks_is_rejected_by_the_parser(indexed, evalset_file):
    with pytest.raises(SystemExit):
        main(["eval", "--queries", str(evalset_file), "--index", str(indexed), "--ks", "one,two"])


def test_config_file_drives_the_index(corpus_dir, tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"chunker": "markdown", "retriever": "bm25"}), encoding="utf-8")
    code = main(["index", "--corpus", str(corpus_dir), "--out", str(tmp_path / "i"),
                 "--config", str(config)])
    assert code == EXIT_OK
    state = json.loads((tmp_path / "i" / "pipeline.json").read_text(encoding="utf-8"))
    assert state["config"]["chunker"] == "markdown"


def test_cli_flags_override_the_config_file(corpus_dir, tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"chunker": "markdown"}), encoding="utf-8")
    main(["index", "--corpus", str(corpus_dir), "--out", str(tmp_path / "i"),
          "--config", str(config), "--chunker", "fixed"])
    state = json.loads((tmp_path / "i" / "pipeline.json").read_text(encoding="utf-8"))
    assert state["config"]["chunker"] == "fixed"


def test_non_ascii_output_survives_a_legacy_console(tmp_path):
    """Regression: ``≥``/``·``/``—`` in the CLI's output raised UnicodeEncodeError on a
    cp1252 console. Because that subclasses ValueError it was caught as a usage error,
    so a passing gate exited non-zero."""
    import io

    from ragforge.report.terminal import force_utf8_output

    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with pytest.raises(UnicodeEncodeError):
        legacy.write("recall ≥ 0.9")
        legacy.flush()

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    original = sys.stdout
    sys.stdout = stream
    try:
        force_utf8_output()
        sys.stdout.write("recall ≥ 0.9 · p95 — ok")
        sys.stdout.flush()
    finally:
        sys.stdout = original
    assert stream.encoding.lower().replace("-", "") == "utf8"


def test_example_scripts_guard_their_own_output():
    """The guard used to live in ``cli.main`` only, so ``examples/quickstart.py`` —
    which prints the same ``·`` and ``→`` — still died partway through on cp1252."""
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = os.path.join(root, "examples", "quickstart.py")
    if not os.path.exists(source):  # installed-from-wheel checkouts ship no examples
        pytest.skip("examples/ not present")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "force_utf8_output()" in text
