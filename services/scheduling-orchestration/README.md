# Scheduling Orchestration

## Purpose

`services/scheduling-orchestration` owns COMP-003, the service that coordinates availability reads, slot holds, booking confirmation, rescheduling, hold release, failure recovery, and idempotent scheduling writes.

## Owned Contracts

- Accepts appointment type, provider/resource, desired slot, patient verification context, rescheduling cutoff rules, and normalized EHR availability responses.
- Produces held slot references, confirmed appointment references, rescheduled appointment references, recovery-safe patient-facing errors, and audit events.
- Calls the EHR adapter for real-time availability, hold, booking, and rescheduling operations.
- Uses operational data store contracts for idempotency keys, hold state, retry state, and short-lived appointment references.
- Emits ARD-API-006 events for booking, rescheduling, and integration failures.

## PHI Handling

- The EHR/practice-management system remains the appointment system of record.
- Store only minimum necessary operational PHI outside the EHR, with default retention of 90 days.
- Avoid full appointment history replication; keep non-authoritative references only when required for orchestration and troubleshooting.
- Exclude PHI from logs, traces, and metrics except for minimum necessary protected audit evidence.
