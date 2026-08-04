---
title: Incident response
category: engineering
---

# Incident response

## Severity levels

**SEV1** — complete outage, or any confirmed data loss or exposure. Customers are
broadly unable to work. Paged immediately, day or night.

**SEV2** — major degradation: a core feature is unavailable or badly slow, but there
is a workaround. Paged during working hours, escalated to SEV1 if it lasts beyond
two hours.

**SEV3** — a contained bug affecting a minority of customers with a workaround.
Handled in the normal queue, no page.

## Declaring

Anyone may declare an incident, and nobody is ever criticised for over-declaring.
Type `/incident declare` in Slack; the bot opens a dedicated channel, starts a
timeline and pages the on-call engineer for the owning service.

Declare early. Downgrading a SEV1 that turned out to be minor costs nothing;
discovering an hour late that a SEV3 was really a SEV1 costs a great deal.

## Roles

The first responder is **incident commander** until they explicitly hand over. The
commander coordinates and decides — they do not debug. A **communications lead**
handles the status page and customer updates. A **scribe** keeps the timeline. On a
SEV1 these must be three different people.

## During

Update the status page within fifteen minutes of declaring a SEV1, then at least
every thirty minutes until resolution, even when the update is "still
investigating". Mitigate before diagnosing: roll back, fail over or disable the
feature flag first, and understand the cause afterwards.

## Afterwards

Every SEV1 and SEV2 gets a written postmortem within five business days, reviewed in
the weekly engineering forum. Postmortems are blameless and are published internally
in full. Action items get an owner and a date, and are tracked to completion — an
unowned action item is not an action item.
