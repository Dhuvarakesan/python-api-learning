# Audit and Compliance

## Purpose

`services/audit-compliance` owns COMP-008, the service that captures immutable audit events for booking, rescheduling, reminder activity, configuration changes, schedule access, integration failures, and security-relevant events.

## Owned Contracts

- Provides the ARD-API-006 audit and observability event interface.
- Accepts events on `audit.booking`, `audit.reschedule`, `audit.reminder`, `audit.configuration`, `audit.schedule_access`, and `ops.integration_failure`.
- Event payloads include event type, timestamp, actor or verified patient context, appointment reference, correlation ID, source system, outcome, schema version, and minimum necessary metadata.
- Produces centralized audit log records, audit exports, compliance review evidence, and downstream observability signals.

## PHI Handling

- Audit events must minimize PHI while retaining enough evidence for compliance, access review, incident investigation, and booking-change accountability.
- Audit logs are retained for 6 years and protected with encryption, access controls, and reviewable export processes.
- Do not accept OTP codes, message bodies, secrets, access tokens, or unnecessary patient details in audit metadata.
- Operational links to short-lived records must tolerate 90-day expiration of non-audit PHI stores.
