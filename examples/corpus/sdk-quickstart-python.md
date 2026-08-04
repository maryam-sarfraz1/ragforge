---
title: Python SDK
category: sdk
---

# Python SDK quickstart

The Python client supports 3.9 and newer.

```bash
pip install meridian-sdk
```

## Authenticating

The constructor reads `MERIDIAN_API_KEY` from the environment when no key is passed
explicitly, which keeps secrets out of source control.

```python
from meridian import Client

client = Client()                      # reads MERIDIAN_API_KEY
client = Client(api_key="mk_live_...") # or pass it directly
```

## Listing datasets

```python
for dataset in client.datasets.list(limit=50):
    print(dataset.id, dataset.name, dataset.row_count)
```

`list()` returns a lazy paginator, so iterating it walks every page transparently.
Call `.page()` instead if you want to handle cursors yourself.

## Running a job and waiting for it

```python
job = client.jobs.create(dataset_id="ds_123", transform="normalise_addresses")
job = job.wait(timeout=600)   # polls with backoff
print(job.status, job.output_uri)
```

## Errors and retries

Every API error raises a subclass of `MeridianError` carrying `.code`,
`.request_id` and `.status`. `RateLimitError` and `ServerError` are retried
automatically — five attempts with exponential backoff and jitter. Turn that off
with `Client(max_retries=0)` when you need to control retries yourself.

```python
from meridian import MeridianError

try:
    client.datasets.get("ds_missing")
except MeridianError as exc:
    print(exc.code, exc.request_id)
```

## Async

`AsyncClient` mirrors the synchronous API and is the better fit inside FastAPI or
any asyncio service.

```python
from meridian import AsyncClient

async with AsyncClient() as client:
    datasets = [d async for d in client.datasets.list()]
```
