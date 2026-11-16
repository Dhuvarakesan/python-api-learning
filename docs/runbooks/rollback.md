# Rollback Runbook Placeholder

## Scope

This runbook covers rollback for the patient web app, staff web app, backend API, reminder worker, and database migrations in the managed cloud deployment described by the ARD.

## Production Prerequisites

- Production deployment uses blue/green, rolling, or equivalent managed release strategy.
- Build, lint, test, accessibility, dependency/CVE scan, and SBOM gates pass before deployment.
- Database migrations are reviewed for backward compatibility before production deployment.
- Release artifacts, container images, or deployable bundles are immutable and traceable to the CI run that generated the SBOM.
- Rollback operators have least-privilege access and all actions are logged.

## Rollback Decision Triggers

- Patient booking, secure rescheduling, or staff schedule flows are unavailable.
- Reminder worker causes duplicate, missing, or incorrect appointment-logistics messages.
- API error rates, latency, or verification failure patterns exceed approved alert thresholds.
- Security Reviewer or Engineering Lead requests rollback due to security, privacy, or data-integrity risk.

## Application Rollback Procedure

1. Confirm incident owner, rollback approver, affected environment, and release identifier.
2. Pause production rollout if a progressive deployment is in progress.
3. Redeploy the last known-good frontend, backend API, and worker artifacts.
4. Confirm health checks for patient web, staff web, API, worker, database connectivity, and observability.
5. Validate booking, staff schedule, reminder preference, and audit-event smoke checks.
6. Keep incident communications free of diagnosis, treatment details, or unnecessary PHI.
7. Record timeline, artifact versions, CI run, SBOM reference, operator, and outcome.

## Database Rollback Procedure

Database changes should be backward compatible. If a database rollback is required:

1. Stop or isolate write traffic according to the selected deployment platform procedure.
2. Confirm the migration impact and approved recovery target with Engineering Lead and Security Reviewer.
3. Prefer forward-fix migrations when possible.
4. If restore is required, follow `backup-restore.md` and validate in an isolated environment before production action when time allows.
5. Resume traffic only after API, worker, staff schedule, booking, and audit checks pass.

## Open Architecture Decisions

- Deployment platform and rollout mechanism.
- Artifact storage and promotion strategy.
- Migration review checklist owner.
- Production smoke-test command set.
