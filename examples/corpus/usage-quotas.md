---
title: Usage quotas
category: account
---

# Usage quotas

Quotas cap what an organisation may consume in a billing period. They are distinct
from rate limits: a rate limit shapes traffic second by second, a quota governs the
month.

## What is metered

| Resource        | Free   | Team    | Enterprise |
|-----------------|--------|---------|------------|
| Storage         | 5 GB   | 250 GB  | Custom     |
| Job minutes     | 500    | 20,000  | Custom     |
| Rows processed  | 1M     | 100M    | Custom     |
| Seats           | 3      | 25      | Unlimited  |

Storage is measured hourly and billed on the period's peak, not its average, so a
large temporary dataset does affect the bill even after deletion.

## Approaching the limit

Owners are emailed at 80% and 95% of any quota, and the console shows a banner from
80%. Usage resets at the start of each billing period; it does not carry over.

## Exceeding the limit

Behaviour past 100% depends on the plan. Free organisations are hard-stopped: new
writes fail with `ERR_4030` until usage falls back under the cap or the plan is
upgraded. Team and Enterprise organisations continue to run and the excess appears
as metered overage on the next invoice, at $0.40 per GB-month of storage and $0.02
per job minute.

Overage is capped at twice the plan fee unless you have raised the ceiling under
**Billing → Spend controls**. Once the cap is hit, execution is suspended for the
rest of the period rather than billing without limit.

Deleting a dataset frees its storage immediately, but rows already processed and job
minutes already consumed are not refunded.
