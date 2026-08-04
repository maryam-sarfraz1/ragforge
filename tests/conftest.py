"""Shared fixtures. Everything here runs without network access or model downloads."""

from __future__ import annotations

import json

import pytest

from ragforge.types import Document, Query

DOC_TEXTS = {
    "auth": (
        "# Authentication\n\n"
        "API keys authenticate server to server traffic. Create a key under settings "
        "and copy the secret, which is shown only once.\n\n"
        "## Rotating\n\n"
        "Rotate a key by creating a replacement with the same scopes, deploying it, "
        "then revoking the old key. Revocation applies within thirty seconds."
    ),
    "limits": (
        "# Rate limits\n\n"
        "Requests are limited per API key. The free plan allows sixty requests per "
        "minute with a burst of one hundred.\n\n"
        "## Throttling\n\n"
        "A throttled request returns status 429 with code ERR_4029. Honour the "
        "Retry-After header and back off exponentially with jitter."
    ),
    "webhooks": (
        "# Webhooks\n\n"
        "Webhooks deliver events to your endpoint over HTTPS. Verify the signature "
        "using an HMAC over the raw request body.\n\n"
        "## Retries\n\n"
        "Failed deliveries retry five times with backoff. Delivery is at least once, "
        "so handlers must be idempotent and deduplicate on event id."
    ),
    "billing": (
        "# Billing\n\n"
        "Invoices are generated on the first of each month and charged to the card on "
        "file. Upgrades are prorated immediately and downgrades apply next period."
    ),
    "roles": (
        "# Roles\n\n"
        "A viewer reads data. An editor changes datasets and runs jobs. An admin "
        "invites members and manages keys. Only an owner can change the plan."
    ),
}


@pytest.fixture
def documents():
    return [Document(id=key, text=text) for key, text in DOC_TEXTS.items()]


@pytest.fixture
def queries():
    return [
        Query(id="q1", text="how do I rotate an API key", grades={"auth": 1.0}),
        Query(id="q2", text="ERR_4029", grades={"limits": 1.0}),
        Query(id="q3", text="verify webhook signature", grades={"webhooks": 1.0}),
        Query(id="q4", text="when am I invoiced", grades={"billing": 1.0}),
        Query(id="q5", text="what can a viewer do", grades={"roles": 1.0}),
    ]


@pytest.fixture
def corpus_dir(tmp_path):
    """A small on-disk corpus with front matter, mirroring examples/corpus."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    for key, text in DOC_TEXTS.items():
        (directory / f"{key}.md").write_text(
            f"---\ntitle: {key.title()}\ncategory: test\n---\n\n{text}\n",
            encoding="utf-8",
        )
    return directory


@pytest.fixture
def evalset_file(tmp_path, queries):
    path = tmp_path / "evalset.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.to_dict()) + "\n")
    return path
