---
title: Node SDK
category: sdk
---

# Node SDK quickstart

The Node client targets Node 18 and above, and ships its own TypeScript
declarations — no `@types` package is required.

```bash
npm install @meridian/sdk
```

## Authenticating

```ts
import { Client } from "@meridian/sdk";

const client = new Client();                          // reads MERIDIAN_API_KEY
const explicit = new Client({ apiKey: "mk_live_..." });
```

## Listing datasets

```ts
for await (const dataset of client.datasets.list({ limit: 50 })) {
  console.log(dataset.id, dataset.name, dataset.rowCount);
}
```

Field names are camelCase in the Node SDK even though the wire format is
snake_case; the client translates in both directions.

## Types

Resource types are exported directly, so responses are fully typed:

```ts
import type { Dataset, Job, JobStatus } from "@meridian/sdk";

function summarise(job: Job): string {
  return `${job.id} is ${job.status}`;
}
```

## Errors

Errors are instances of `MeridianError` with `code`, `requestId` and `status`.
`RateLimitError` and `ServerError` extend it and are retried automatically.

```ts
import { MeridianError } from "@meridian/sdk";

try {
  await client.datasets.get("ds_missing");
} catch (err) {
  if (err instanceof MeridianError) console.error(err.code, err.requestId);
}
```

## Bundlers and edge runtimes

The package ships both ESM and CJS builds. It runs on Cloudflare Workers and Vercel
Edge, where it uses the platform `fetch` rather than Node's HTTP stack. Never ship a
live key to a browser bundle — issue a scoped, short-lived token from your own
backend instead.
