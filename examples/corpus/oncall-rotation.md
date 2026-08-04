---
title: On-call
category: engineering
---

# On-call

## The rotation

On-call runs weekly, handing over at 10:00 on Wednesday — mid-week and mid-morning,
so a messy handover happens when the whole team is around rather than on a Friday
evening. Each service has a primary and a secondary. The secondary is paged only if
the primary does not acknowledge within five minutes.

Every engineer joins the rotation after completing onboarding and shadowing two full
shifts. A rotation never has fewer than four people; below that we merge it with a
neighbouring team rather than run people into the ground.

## Expectations

Acknowledge a page within five minutes and be able to reach a laptop and decent
connectivity within thirty. You are not expected to fix everything yourself —
escalating to the service owner is a normal, encouraged part of the job.

While on call, keep planned work light. Half a sprint's capacity is the assumption,
and taking on a large project during your week is how both the project and the
incident response end up going badly.

## Compensation and recovery

On-call weeks are paid at a flat weekly rate plus an hourly rate for time worked
outside business hours. If you are paged between 22:00 and 06:00, take the following
morning off — this is not a favour to ask for, it is expected, and your lead will
ask about it if you do not.

## Swapping a shift

Swap directly with a colleague and update the schedule in PagerDuty yourself; no
approval is needed. For a swap inside 24 hours of the shift, tell your lead as well
so escalation paths stay accurate. If you cannot find a swap, say so in the team
channel — leads are responsible for covering, not you.

## Handover

The outgoing engineer writes a handover note covering open incidents, anything
suppressed or muted, and any alert that fired more than twice. Alerts that page
without being actionable are bugs; file them and fix them.
