# Shared Configuration

Configuration is environment-specific and secret-free. Source-controlled files may define names, endpoints, feature flags, retention settings, and references to managed secrets, but must not contain static API keys, access tokens, OTP secrets, private keys, or production credentials.

## Loading Rules

- `APP_ENV` must be one of `local`, `test`, `staging`, or `production`.
- `SERVICE_NAME` identifies the emitting service in logs, traces, and audit metadata.
- Vendor and EHR endpoints use non-secret environment variables such as `EHR_BASE_URL`.
- Credentials use secret references such as `cloud-secret-manager:ehr-client-credential:current`.
- Frontend artifacts must receive only public SaaS/embed configuration and never secret references or secret values.

## Secret Reference Variables

- `EHR_CLIENT_CREDENTIAL_SECRET_REF`
- `SCHEDULING_SAAS_CREDENTIAL_SECRET_REF`
- `SMS_VENDOR_CREDENTIAL_SECRET_REF`
- `EMAIL_VENDOR_CREDENTIAL_SECRET_REF`
- `OBSERVABILITY_CREDENTIAL_SECRET_REF`

Allowed reference formats:

- `managed-vault:name`
- `managed-vault:name:version`
- `cloud-secret-manager:name`
- `cloud-secret-manager:name:version`
- `local-dev:name`

`local-dev` is only for local developer secret stores or ignored `.env.local` files. It must not be used in staging or production deployment manifests.

## Default Retention Values

- Operational PHI or PHI-adjacent records outside the EHR: 90 days.
- Audit records: 6 years.
