---
title: Authentication
category: platform
---

# Authentication

Every request to the Meridian API must carry a bearer token in the `Authorization`
header. Meridian issues two kinds of credential: **API keys** for server-to-server
traffic and **session tokens** for the web console. Session tokens are short lived
and cannot be created programmatically.

## Creating an API key

Open **Settings → Developers → API keys** and choose *Create key*. You must give the
key a name and select its scopes. The secret is shown exactly once, at creation
time; Meridian stores only a hash of it. If you lose a secret you must create a new
key — there is no way to recover the original value.

Keys are prefixed by environment: `mk_live_` for production and `mk_test_` for the
sandbox. Test keys never touch production data and are not billed.

## Scopes

A key carries an explicit list of scopes, and a request fails with `403` if the key
lacks the scope for that endpoint. Available scopes are `datasets:read`,
`datasets:write`, `jobs:read`, `jobs:write`, `webhooks:manage` and `admin:billing`.
Grant the narrowest set that works — a read-only export job has no business holding
a write scope.

## Rotating a key

Keys older than 90 days are flagged in the console, and organisation owners receive
a reminder email at 83 days. Rotation is deliberately a two-step process so it
causes no downtime:

1. Create a new key with identical scopes and deploy it to your environment.
2. Confirm traffic has moved over on the key's usage graph, then revoke the old key.

Revocation takes effect within 30 seconds across all regions. A revoked key cannot
be restored. In the sandbox, rotation is automated nightly and old test keys are
purged after seven days.

If a secret leaks, revoke it immediately rather than waiting for the rotation
window, then audit the key's request log for unexpected source addresses.
