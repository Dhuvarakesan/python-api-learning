# Operational Data Retention Policy

## Scope

This policy applies to data retained by `infra/operational-data-store` for custom scheduling, messaging, EHR integration, audit, and observability workflows. The EHR/practice-management system remains the system of record for appointments, patient scheduling data, appointment history, and authoritative patient records.

## Retention Rules

| Data class | Examples | Retention | Required action |
|---|---|---:|---|
| Operational PHI outside the EHR | Slot hold state, reminder scheduling state, delivery status tied to patient contact data, retry records, non-authoritative appointment references, integration workflow state | 90 days | Automatically expire, delete, or anonymize after 90 days unless an approved vendor workflow requires a documented longer period. |
| Audit logs and audit evidence | `audit.booking`, `audit.reschedule`, `audit.reminder`, `audit.configuration`, `audit.schedule_access`, `ops.integration_failure` records and immutable audit exports | 6 years | Retain in the audit/compliance log store for 6 years with integrity controls and access logging. |
| Idempotency keys and correlation records without PHI | Request IDs, correlation IDs, replay protection keys, service routing metadata | Minimum period needed for safe retry and reconciliation, not to exceed operational PHI retention when linked to PHI | Expire with the associated workflow state unless detached from PHI and required for audit traceability. |

## Control Requirements

- Operational stores must separate PHI-bearing workflow records from audit references so audit evidence can be retained for 6 years after operational PHI expires.
- Retention jobs must run at least daily for PHI-bearing operational records and produce auditable success or failure events.
- Backups, replicas, analytics exports, and vendor-managed copies must honor the same 90-day operational PHI retention and 6-year audit log retention windows.
- Any exception that keeps operational PHI outside the EHR for longer than 90 days requires documented workflow justification, owner approval, and review of vendor BAA coverage.
- Retention failures must emit `ops.integration_failure` or equivalent audit/operations evidence for investigation.

## Environment Considerations

- Production retention controls apply to production stores that may contain real PHI.
- Lower environments must not use production PHI. Test data must be synthetic, de-identified, or vendor-provided sandbox data.
- Audit evidence generated in lower environments is retained only for validation and operational troubleshooting unless explicitly promoted as release evidence.
