# Infrastructure Release Gates

This directory documents the release-readiness controls for the scheduling integration services, operational data store, observability assets, and deployment pipeline. The workflows are intentionally lightweight so they can support a small-business clinic implementation while still retaining evidence for HIPAA-oriented operational review.

## CI Quality Gate

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch. It verifies:

- build and deployable artifact generation with `npm run build`;
- linting with `npm run lint`;
- TypeScript/static analysis with `npm run typecheck`;
- unit tests and coverage with `npm run coverage`;
- contract and integration tests for shared interfaces and custom services;
- Playwright E2E tests;
- accessibility scanning with Axe;
- Lighthouse performance and accessibility assertions;
- audit assertion checks for PHI controls, secret handling, and audit event contracts.

The workflow uploads quality, coverage, E2E, accessibility, performance, and audit assertion evidence with a 90-day retention period.

## Security and SBOM Gate

`.github/workflows/security.yml` blocks release candidates when high or critical findings are detected by dependency, filesystem, IaC, secret, or container scans. The workflow has no inline high/critical allowlist and no risk-acceptance bypass.

Security controls include:

- `npm audit --audit-level=high` for dependency vulnerabilities;
- Gitleaks for secret scanning;
- Trivy filesystem scans for dependency and configuration vulnerabilities;
- Checkov for IaC and infrastructure configuration checks;
- Trivy image scanning when container build definitions are present;
- CycloneDX SBOM generation for production dependencies and full dependency evidence.

SBOM and security evidence are retained for 365 days. Audit logs and operational compliance evidence produced by the application remain subject to the six-year audit-retention requirement defined by the architecture.

## Risk Acceptance Mechanism

High and critical findings must be fixed before release unless clinic leadership and the security owner formally accept the risk outside the workflow. A valid risk acceptance record must be stored in the organization's ticketing, GRC, or risk register system, not as a workflow allowlist. The record must include:

- scanner finding ID, package/resource/image, severity, and affected release;
- PHI exposure analysis and compensating controls;
- named owner and accountable approver;
- expiration date no later than 30 days for critical findings and 90 days for high findings;
- remediation plan and target date;
- link to the pull request or deployment record being considered.

Because the security workflow still fails on high and critical results, releasing with an accepted risk requires an explicit out-of-band exception process, such as a protected-environment approval tied to the risk record and a documented change-management decision.

## Deployment Evidence Gate

`.github/workflows/deploy.yml` is manually triggered per environment. It validates infrastructure and release evidence before deployment, then captures smoke-test and rollback evidence.

The deployment gate:

- captures release metadata for the target environment and commit;
- validates Terraform modules when `.tf` files exist;
- runs Checkov against `infra`;
- builds the release artifact;
- generates a deployment SBOM;
- creates a GitHub deployment record;
- executes the environment-specific deployment command from `DEPLOY_COMMAND`;
- runs smoke tests against `SMOKE_TEST_URL`;
- executes `ROLLBACK_COMMAND` after a failed deployment or smoke gate;
- uploads deployment, smoke-test, rollback, and SBOM evidence for 365 days.

The deploy command, smoke-test URL, and rollback command are externalized as environment secrets so each HIPAA-eligible runtime can provide its own implementation without hard-coding cloud credentials or PHI-adjacent endpoints in the repository.

## Operational Constraints

- The EHR/practice-management system remains the appointment system of record.
- Operational PHI outside the EHR should be retained for 90 days unless a vendor workflow requires longer retention.
- Audit logs should be retained for 6 years.
- Cloud runtimes and vendors that handle PHI must be HIPAA-eligible and covered by BAAs.
- Custom services should stay narrow: integration edge, scheduling orchestration, messaging orchestration, configuration adapter, EHR adapter, and audit-compliance responsibilities only.
