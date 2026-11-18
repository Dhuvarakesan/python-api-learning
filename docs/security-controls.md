# Security Controls

The clinic booking platform may process PHI in later environments. Production deployment is blocked until HIPAA-eligible vendors and signed BAAs are confirmed for every service that stores, processes, transmits, observes, or supports PHI.

## CI/CD Controls

- Install gate: CI runs `npm ci` from the lockfile for repeatable dependency installation.
- Quality gate: CI runs linting, TypeScript type-checking, unit tests, and workspace builds before security and infrastructure placeholder jobs.
- Dependency/CVE gate: CI runs `npm run security:scan`, currently backed by `npm audit --audit-level=high`.
- SBOM gate: CI runs `npm run security:sbom` and uploads `sbom.cdx.json` as a build artifact.
- IaC gate: CI runs `npm run iac:validate` as the future hook for Terraform, policy-as-code, or vendor-native infrastructure validation.
- Promotion gate: CI runs `npm run promotion:check` as the future hook for environment approvals and release evidence.

## Production Controls Required Before PHI Processing

- Use HTTPS/TLS in transit for browser, service, database, queue, scheduler, identity, and vendor integration traffic.
- Use encryption at rest for managed relational data, backups, logs, object stores, queues, and generated artifacts that may contain PHI.
- Enforce RBAC for staff users, service accounts, database roles, CI/CD identities, and operators.
- Keep logs, traces, metrics, alerts, test fixtures, and support exports PHI-redacted by default.
- Keep vendor integrations isolated behind adapters and configuration so a vendor change does not spread PHI handling assumptions through application code.
- Keep confirmed appointment history in the existing scheduling system for the first release; do not bulk-migrate appointment history into this application.

## Implementation Hooks

- Replace `npm run security:scan` with the selected SCA scanner when procurement chooses a vendor.
- Publish the generated CycloneDX SBOM to the selected artifact or compliance evidence store.
- Add IaC modules under `infra/modules` only after the target platform and HIPAA controls are approved.
- Add promotion rules for development, staging, and production after environment ownership, approvers, and rollback procedures are defined.
