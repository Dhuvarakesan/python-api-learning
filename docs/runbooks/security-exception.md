# Security Reviewer Exception Runbook

## Purpose

High and critical dependency/CVE findings block production deployment. A production deployment may proceed only when the Security Reviewer documents and approves a time-bound exception.

## Exception Requirements

The exception record must include:

- Finding identifier, package, severity, affected component, and CI run.
- Reason the production deployment cannot wait for remediation.
- Patient, staff, privacy, and operational impact assessment.
- Compensating controls.
- Expiration date.
- Remediation owner and target date.
- Security Reviewer approval.

## CI/CD Handling

The CI workflow runs dependency/CVE scanning with `npm audit --audit-level=high` and generates a CycloneDX SBOM with `cyclonedx-npm`.

For production deployment readiness:

- Build, lint, type check, unit test, integration test, accessibility, and SBOM gates must pass.
- High or critical dependency/CVE findings fail the security gate.
- If a high or critical finding remains, `production-deployment-gate` requires `security_reviewer_exception=true` and a non-empty `security_exception_reference`.
- The production environment should require reviewer approval from the Security Reviewer or delegated release approver.

## Review Cadence

Open exceptions must be reviewed before each production deployment and closed when remediation ships. Expired exceptions cannot be reused.
