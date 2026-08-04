---
title: Billing and plans
category: account
---

# Billing and plans

Meridian bills monthly in arrears. Your invoice combines a flat plan fee with any
metered overage recorded during the period.

## Plans

- **Free** — one project, 5 GB storage, community support. No card required.
- **Team** — $99/month, ten projects, 250 GB storage, email support.
- **Enterprise** — custom pricing, unlimited projects, SSO, a dedicated support
  channel and a signed uptime commitment.

## Changing plan

Upgrade under **Settings → Billing → Change plan**. Upgrades apply immediately and
are prorated for the remainder of the period. Downgrades take effect at the start of
the next period, so you retain the higher limits for what you have already paid for.

An organisation owner or a member with the `admin:billing` scope may change the
plan; other roles see the page read-only.

## Invoices and payment

Invoices are generated on the first of the month and charged to the card on file the
same day. Every invoice is a downloadable PDF under **Settings → Billing → History**.
Enterprise customers may pay by bank transfer on net-30 terms.

If a charge fails, we retry on days 3, 7 and 14 and email the owners each time.
After the third failure the organisation moves to a restricted state: the API keeps
serving reads, but writes and job execution are suspended until the balance clears.
Nothing is deleted for non-payment.

## Taxes and credits

VAT or sales tax is added where applicable; add your registration number under
**Billing → Tax details** to have it shown on the invoice. Credits from a support
resolution are applied automatically against the next invoice.
