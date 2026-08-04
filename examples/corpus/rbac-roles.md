---
title: Roles and permissions
category: account
---

# Roles and permissions

Access is role-based. Every member of an organisation holds exactly one role, and
roles are strictly nested — each includes everything the one below it can do.

## The roles

**Viewer** — read datasets, jobs and dashboards. Cannot change anything, cannot see
billing, cannot see API key metadata. The right default for analysts and for
just-in-time SSO provisioning.

**Editor** — everything a viewer can do, plus create and modify datasets, run jobs,
and manage webhooks. Cannot invite people or touch billing.

**Admin** — everything an editor can do, plus invite and remove members, assign
roles up to admin, create and revoke API keys, and configure SSO.

**Owner** — everything an admin can do, plus change the plan, manage payment
details, transfer ownership and delete the organisation.

## Owner versus admin

The distinction is deliberately narrow: an admin runs day-to-day access, an owner
can end the account or change what it costs. Billing, plan changes, ownership
transfer and organisation deletion are owner-only. Every organisation must keep at
least one owner — the last one cannot be demoted or removed until another is
appointed.

## Changing someone's role

Under **Settings → Members**, use the row menu and choose *Change role*. Admins may
assign up to admin; only owners may grant owner. The change takes effect on the
member's next request; no re-login is needed.

## Service accounts

Automation should use an API key rather than a human's account, so that access
survives someone leaving. A key's scopes are capped by the role of the member who
created it: an editor cannot mint a key with `admin:billing`.
