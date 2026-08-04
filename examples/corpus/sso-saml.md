---
title: SSO and SAML
category: account
---

# Single sign-on

SSO is available on the Enterprise plan. Meridian supports SAML 2.0 and OIDC.

## Supported providers

Okta, Microsoft Entra ID (formerly Azure AD), Google Workspace, OneLogin, JumpCloud
and Ping Identity are tested and documented. Any provider that implements SAML 2.0
with SP-initiated flow will work, but only the tested list is supported.

## Setting up SAML

1. Under **Settings → Security → SSO**, copy the ACS URL and Entity ID.
2. Create the application in your identity provider using those values.
3. Map `email`, `firstName` and `lastName` as attributes. `email` must be the
   assertion's `NameID` and must be verified in your directory.
4. Paste the provider's metadata XML back into Meridian and press *Test connection*,
   which runs a full round trip without changing anything.
5. Enable the connection once the test passes.

Keep one break-glass owner account on password login until the connection is
verified. Enabling SSO with a broken mapping otherwise locks the whole organisation
out, and recovering that needs a support ticket and a manual identity check.

## Enforcement

With **Require SSO** on, password and OAuth logins are refused for every member and
existing sessions are terminated within fifteen minutes. API keys are unaffected —
they authenticate services, not people, and continue to work.

## Just-in-time provisioning

JIT creates a Meridian account on first successful assertion, with the default role
set under **Security → SSO → Default role**; `viewer` is the safe choice. SCIM
provisioning is available for Okta and Entra ID, and it deactivates a Meridian
account within minutes of the directory deactivating the user.
