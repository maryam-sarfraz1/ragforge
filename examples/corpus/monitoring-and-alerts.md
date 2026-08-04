---
title: Monitoring
category: engineering
---

# Monitoring and alerts

## Dashboards

Every service has a standard dashboard at `grafana.internal/d/<service>`, generated
from the service definition so the layout is identical everywhere. It shows the four
golden signals — traffic, error rate, latency percentiles, saturation — plus that
service's own key metrics.

Three dashboards matter most during an incident: **Platform overview** for whether
the problem is broad or local, **Request pipeline** for where in the path latency is
accumulating, and **Dependencies** for whether a downstream is at fault.

## What we alert on

Alert on symptoms customers feel, not on causes. High CPU is not an alert; requests
failing is. Every paging alert must be actionable, urgent and real — if the response
is to acknowledge and go back to sleep, it should not have paged.

Standard SLO alerts are generated for each service: error budget burn rate at 2% per
hour pages, 5% per day opens a ticket.

## Adding an alert

Alerts live in code, in `monitoring/alerts/<service>.yaml`, and ship through the
normal pipeline. Each needs a name, expression, `for` duration, severity, a runbook
URL and an owning team. The runbook link is mandatory and CI rejects an alert
without one — being paged at 03:00 with no idea what to do is the failure mode this
prevents.

Use `severity: page` only for something that needs a human within minutes.
`severity: ticket` covers everything else.

## Tuning

A `for` duration under two minutes on a latency alert will fire on noise. Review
alert volume monthly: any alert that fired more than five times without a real
incident behind it gets tuned or deleted. A noisy alert is worse than no alert,
because it teaches people to ignore the pager.
