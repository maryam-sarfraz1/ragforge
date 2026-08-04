---
title: Error codes
category: platform
---

# Error codes

Every error response carries a machine-readable `code`, a human `message`, and a
`request_id`. Always quote the `request_id` when contacting support — it lets us
find the exact trace.

```json
{ "code": "ERR_4021", "message": "Upstream token expired", "request_id": "req_8f2b..." }
```

## 4xx — the request needs changing

| Code       | HTTP | Meaning                                                    |
|------------|------|------------------------------------------------------------|
| `ERR_4001` | 400  | Malformed JSON body                                        |
| `ERR_4003` | 400  | A required field is missing                                |
| `ERR_4010` | 401  | No credential supplied                                     |
| `ERR_4011` | 401  | API key revoked or unknown                                 |
| `ERR_4021` | 401  | Upstream token expired — refresh credentials and retry     |
| `ERR_4030` | 403  | The key lacks the scope this endpoint needs                |
| `ERR_4041` | 404  | Resource does not exist, or the key cannot see it          |
| `ERR_4090` | 409  | Conflict: the resource changed since you last read it      |
| `ERR_4029` | 429  | Rate limited — honour `Retry-After`                        |

## 5xx — the request may be retried

| Code       | HTTP | Meaning                                                    |
|------------|------|------------------------------------------------------------|
| `ERR_5000` | 500  | Unexpected internal error; already alerted on our side     |
| `ERR_5003` | 503  | A dependency is unavailable; retry with backoff            |
| `ERR_5040` | 504  | The upstream query exceeded its deadline                   |

Retry `ERR_5003` and `ERR_5040` with exponential backoff. Do not blindly retry
`ERR_5000` — if it repeats for the same input, the request itself is the trigger and
retrying only multiplies the failure.

`ERR_4041` is deliberately ambiguous between "missing" and "not visible to you", so
that an attacker cannot use the API to probe which resources exist.
