# Backup and Restore Runbook Placeholder

## Scope

This runbook covers restore readiness for scheduling-related application data, audit events, reminder preferences, delivery attempts, staff roles, verification state, and integration state stored in the managed PostgreSQL database.

## Production Prerequisites

- Managed PostgreSQL service is HIPAA-eligible and covered by a signed BAA.
- Backup encryption at rest and in transit is enabled.
- Backup retention duration, RPO, and RTO are approved by Engineering Lead and Security Reviewer.
- Restore access is limited to named operators with least-privilege production access.
- Restore activity emits audit logs and operational alerts.

## Backup Controls

- Daily automated backups are required for production scheduling data.
- Point-in-time recovery should be enabled when supported by the selected managed PostgreSQL service.
- Backup success, backup failure, and backup age alerts must be routed to production on-call contacts.
- Backup artifacts must not be exported to unmanaged storage or services without a signed BAA.

## Restore Validation

Before production launch and on the recurring cadence selected by Engineering Lead:

1. Select a backup artifact from the production-like stage or test environment.
2. Restore it into an isolated non-production database.
3. Run database migrations and application smoke checks against the restored database.
4. Verify core booking, staff schedule, reminder preference, and audit-event reads.
5. Record restore duration, data timestamp, operator, environment, and validation outcome.
6. Attach restore evidence to the release-readiness record.

## Open Architecture Decisions

- Cloud provider and managed PostgreSQL product.
- IaC tool and backup resource definitions.
- Retention duration.
- RPO and RTO targets.
- Restore validation cadence.
