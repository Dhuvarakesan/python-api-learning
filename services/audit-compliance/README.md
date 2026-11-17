# Audit and Compliance

## Purpose

`services/audit-compliance` owns COMP-008, the service that captures immutable audit events for booking, rescheduling, reminder activity, configuration changes, schedule access, integration failures, and security-relevant events.

## Owned Contracts

- Provides the ARD-API-006 audit and observability event interface.
- Accepts events on `audit.booking`, `audit.reschedule`, `audit.reminder`, `audit.configuration`, `audit.schedule_access`, and `ops.integration_failure`.
- Event payloads include event type, timestamp, actor or verified patient context, appointment reference, correlation ID, source system, outcome, schema version, and minimum necessary metadata.
- Produces centralized audit log records, audit exports, compliance review evidence, and downstream observability signals.

## Source Primitives

`src/audit-event-v1.ts` provides `createAuditEventV1` for immutable `audit-event.v1` records. The schema records only the evidence required for booking, rescheduling, reminder, configuration, schedule access, and integration failure accountability.

`src/observability.ts` provides PHI-minimized logging, trace attribute, and metric conventions. Logs and traces must carry `correlation_id` and service/event/outcome references so investigations can join records without patient details.

## PHI Handling

- Audit events must minimize PHI while retaining enough evidence for compliance, access review, incident investigation, and booking-change accountability.
- Audit logs are retained for 6 years and protected with encryption, access controls, and reviewable export processes.
- Do not accept OTP codes, message bodies, secrets, access tokens, or unnecessary patient details in audit metadata.
- Operational links to short-lived records must tolerate 90-day expiration of non-audit PHI stores.

Logging and tracing guidance explicitly excludes OTP codes, one-time passwords, message bodies, secrets, access tokens, refresh tokens, authorization headers, raw EHR payloads, and unnecessary patient details including patient names, dates of birth, full addresses, phone numbers, and email addresses.

## Metrics

The service publishes or forwards these PHI-safe metrics:

- `booking_latency_ms`
- `availability_lookup_latency_ms`
- `ehr_error_rate`
- `hold_count`
- `conflict_count`
- `schedule_view_latency_ms`
- `messaging_delivery_status_total`
- `audit_delivery_failures_total`
