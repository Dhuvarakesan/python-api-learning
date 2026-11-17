# Shared Security and Secret Handling

Custom services must keep source code, checked-in configuration, logs, frontend bundles, and exported SaaS configuration free of secret values.

## Secret Handling

- Store API keys, OAuth client secrets, access tokens, OTP signing secrets, private keys, and production credentials in a managed vault or cloud secret manager.
- Keep only secret references in deployment configuration.
- Resolve secrets at runtime inside backend services only.
- Never expose secret values or secret references to browser code or SaaS-hosted frontend artifacts.
- Rotate vendor and EHR credentials through the secret manager, not through source commits.

`secrets.ts` provides a placeholder `SecretManager` interface and an `UnconfiguredSecretManager` that fails closed until a cloud-specific adapter is added.

## Minimum Necessary PHI

Operational records should store the minimum data needed to recover workflows:

- EHR appointment or hold references instead of full appointment payloads.
- Vendor delivery identifiers instead of message bodies.
- Patient contact channel references instead of full contact details when possible.
- Status, timestamps, correlation IDs, and idempotency keys needed for reconciliation.

Do not log patient names, full contact details, OTP codes, message bodies, access tokens, or raw EHR payloads.

## Retention

- Operational PHI outside the EHR uses `operational-90-days`.
- Audit evidence uses `audit-6-years`.
- Legal hold must be explicit in retention metadata and reviewed separately from normal operational retention.
