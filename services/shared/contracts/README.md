# Shared Service Contracts

These contracts define the minimum request, event, error, and audit fields used by custom integration services.

## Request Conventions

- Every inbound request and emitted event carries `correlationId`.
- Every mutating operation carries `idempotencyKey`.
- HTTP services use `x-correlation-id` and `idempotency-key` headers.
- Mutating operations include booking, rescheduling, hold release, OTP/reminder sends, and configuration updates.
- Services must propagate the same correlation ID to logs, traces, audit events, vendor calls, and downstream service calls.

## Required Shared Fields

`integration-contracts.ts` defines:

- `correlationId` for traceability across SaaS, custom services, vendors, and EHR calls.
- `idempotencyKey` for conflict-safe mutating integration operations.
- `retry.retryable` with optional retry timing and attempt metadata.
- `userSafeErrorCode` and `userSafeMessage` for patient/staff-safe recovery messaging.
- `audit` metadata for ARD-API-006 topics, actor context, PHI classification, and retention policy.

## Audit Topics

Allowed event topics match ARD-API-006:

- `audit.booking`
- `audit.reschedule`
- `audit.reminder`
- `audit.configuration`
- `audit.schedule_access`
- `ops.integration_failure`

## PHI and Retention Metadata

Operational records must include a retention policy:

- `operational-90-days` for operational PHI or PHI-adjacent state outside the EHR.
- `audit-6-years` for audit evidence.

Payloads should store references and status needed for workflow recovery, not full patient clinical or contact details unless a vendor workflow requires it.
