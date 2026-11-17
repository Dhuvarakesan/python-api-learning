# Environment Separation Controls

## Scope

These controls apply to configuration for local development, dev, test, staging, and production environments across SaaS configuration exports, custom integration services, operational data stores, observability, and audit/compliance components.

## Environment Boundaries

| Environment | Purpose | Data allowed | Credential source | Deployment boundary |
|---|---|---|---|---|
| Local development | Individual developer implementation and debugging | Synthetic or de-identified fixtures only | Local developer secret store or ignored `.env.local` values | Developer workstation or disposable local runtime |
| Dev | Shared integration development | Synthetic data and vendor sandbox data only | Non-production managed secrets | Isolated non-production cloud/runtime resources |
| Test | Automated and manual verification | Synthetic test datasets only | Non-production managed secrets scoped to test | Isolated test resources and resettable stores |
| Staging | Release rehearsal and operational validation | Synthetic, de-identified, or approved vendor sandbox data only | Staging managed secrets separate from production | Production-like but isolated runtime and data stores |
| Production | Live clinic scheduling operations | Minimum necessary live PHI and audit evidence | Production managed secrets | Production runtime, stores, observability, and vendor integrations |

## Required Controls

- Production credentials are not used in lower environments, including local development, dev, test, and staging.
- Production PHI is not copied, restored, imported, replayed, or manually entered into lower environments.
- Lower environments must use separate secret names, service accounts, encryption keys, data stores, queues, webhook endpoints, SaaS tenants or sandboxes, and observability destinations from production.
- Configuration files in source control may contain environment names, non-secret endpoints, feature flags, retention settings, and secret references, but must not contain static API keys, access tokens, OTP secrets, private keys, or production credential values.
- Production access requires least-privilege roles, MFA-backed staff or service identity, audit logging, and approved operational need.
- CI/CD deployment permissions must be separated by environment so non-production pipelines cannot read production secrets or deploy to production without the production release path.
- Any production incident investigation that needs replay or reproduction in a lower environment must use synthetic or de-identified records derived without retaining production identifiers, message bodies, OTP codes, or full patient contact details.

## Configuration Expectations

- `APP_ENV` values must map to an isolated environment. Local tooling may use `local`; shared environments must use `dev`, `test`, `staging`, or `production`.
- Secret references must identify the target environment in the managed secret name or path.
- SaaS configuration exports must be stored per environment and reviewed before promotion.
- Retention settings must preserve the architecture constraint that operational PHI outside the EHR is retained for 90 days and audit logs are retained for 6 years.

## Evidence

- Environment manifests or configuration exports should show distinct resource names for each environment.
- Secret manager audit logs should show that lower-environment service identities cannot access production credentials.
- Operational data store retention evidence should show daily enforcement for production PHI-bearing records and audit log retention coverage for 6 years.
