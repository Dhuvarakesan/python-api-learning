# EHR Adapter

## Purpose

`services/ehr-adapter` owns COMP-006, the adapter that encapsulates EHR/practice-management integration using FHIR Scheduling APIs where supported and vendor-specific APIs where required for holds and booking writes.

## Owned Contracts

- Accepts availability queries, hold requests, booking requests, rescheduling requests, appointment read requests, and provider/resource mappings.
- Produces normalized availability, hold confirmations, appointment IDs, conflict responses, and EHR error mappings.
- Shields other services from vendor-specific EHR behavior and keeps system-of-record writes idempotent where supported.
- Emits ARD-API-006 operations events for integration failures and audit-relevant scheduling outcomes.

## PHI Handling

- The EHR/practice-management system remains authoritative for appointments, provider schedules, patient identifiers, appointment status, and appointment history.
- Store no EHR payload copies here unless an implementation task adds an approved short-lived cache.
- Redact EHR errors before logging or returning them to callers; avoid PHI in traces and span attributes.
- Use encrypted transport and secrets management for all EHR credentials and API traffic.
