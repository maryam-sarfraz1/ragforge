---
title: Webhooks
category: platform
---

# Webhooks

Webhooks push events to your endpoint as they happen, which is cheaper and lower
latency than polling.

## Registering an endpoint

Register under **Settings → Webhooks**, or with `POST /v1/webhooks`. Your URL must
be HTTPS and must answer the validation `GET` within five seconds. Subscribe to
specific event types — `job.completed`, `job.failed`, `dataset.created`,
`dataset.deleted`, `invoice.paid` — rather than to everything.

## Verifying the signature

Every delivery carries an `X-Meridian-Signature` header of the form
`t=<timestamp>,v1=<hex>`. The `v1` value is an HMAC-SHA256 over
`"{timestamp}.{raw_body}"` keyed with your endpoint's signing secret.

Compute the HMAC over the **raw** request body. Parsing the JSON and re-serialising
it changes the byte sequence and the signature will never match — this is the most
common integration bug we see. Compare with a constant-time function, and reject
deliveries whose timestamp is more than five minutes old to blunt replay attacks.

## Retries and duplicates

A delivery is successful when your endpoint returns any `2xx` within ten seconds.
Anything else is retried with exponential backoff at 1m, 5m, 30m, 2h and 6h. After
five failures the endpoint is disabled and the organisation's owners are emailed.

Delivery is **at least once**, so your handler must be idempotent. Every event has a
stable `event_id`; record the ones you have processed and ignore repeats. Duplicates
are expected during retries and after a network timeout where your endpoint
succeeded but the response never reached us.

Events are not guaranteed to arrive in order. Use the `created_at` field to
reconcile rather than assuming sequence.
