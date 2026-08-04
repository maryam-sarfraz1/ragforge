---
title: Data retention
category: compliance
---

# Data retention and deletion

## Live data

Datasets you upload are kept until you delete them. Meridian does not expire
customer data on its own, and does not use customer content to train models.

## Deleting a dataset

Deleting moves a dataset to a **30-day recovery window**. During that time it is
invisible to the API and does not count toward your row quota, but it still occupies
storage and can be restored with a single click under **Datasets → Recently
deleted**.

After 30 days the recovery window closes and a purge job removes the objects. Purged
data is unrecoverable — support cannot retrieve it. Encrypted backup segments are
overwritten on their own cycle, completing within 35 days of the purge.

## Other retention windows

| Data                 | Retained for |
|----------------------|--------------|
| Job execution logs   | 90 days      |
| API request logs     | 30 days      |
| Audit log            | 400 days     |
| Webhook delivery log | 14 days      |
| Backups              | 35 days      |

Enterprise organisations can extend the audit log to seven years.

## Immediate deletion requests

To skip the recovery window — as you may need to for a GDPR erasure request or after
uploading something by mistake — an organisation owner can raise a deletion request
under **Settings → Privacy → Erasure request**. We acknowledge within one business
day and complete the purge within seven, then send written confirmation naming the
affected objects. This is irreversible from the moment it is accepted.

Closing an account starts the same purge on the whole organisation after a 30-day
grace period. Export anything you need first; exports are not automatic.
