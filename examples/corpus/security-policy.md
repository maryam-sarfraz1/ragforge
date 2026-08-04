---
title: Security
category: compliance
---

# Security

## Certifications

Meridian holds SOC 2 Type II, renewed annually, and ISO 27001. The current report is
available under NDA from **Settings → Compliance → Documents**, or from your account
team. We are GDPR compliant and will sign a DPA; standard contractual clauses cover
transfers out of the EU. HIPAA is supported on Enterprise under a signed BAA.

## Encryption

Data is encrypted in transit with TLS 1.3; TLS 1.0 and 1.1 are refused outright. At
rest, everything is encrypted with AES-256 using keys held in a managed KMS and
rotated annually. Enterprise customers may supply their own KMS key, which lets you
revoke our access to your data by disabling the key.

Secrets — API key hashes, webhook signing secrets, SSO certificates — are stored in
a separate vault with its own key hierarchy and a distinct access path.

## Access control

Engineer access to production is just-in-time: it is requested for a stated reason,
approved by a second person, granted for at most four hours, and logged. There is no
standing production access. Every access event lands in the audit log, and
Enterprise customers can stream that log to their own SIEM.

## Testing

An independent firm runs a full penetration test annually and the summary letter is
available to customers. Dependencies are scanned continuously and critical
vulnerabilities are patched within 72 hours of a fix being published.

## Reporting a vulnerability

Email security@meridian.example with reproduction steps. We acknowledge within one
business day and aim to remediate critical findings within seven. Please do not open
a public issue before we have responded. We run a paid bounty and will credit you in
the advisory unless you ask us not to.
