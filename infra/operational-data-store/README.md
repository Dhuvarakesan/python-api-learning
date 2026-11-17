# Operational Data Store

## Purpose

`infra/operational-data-store` owns COMP-007, the infrastructure area for minimum necessary operational data used by orchestration services.

## Owned Contracts

- Stores idempotency keys, hold state, reminder scheduling state, delivery status, integration state, retry records, audit references, and non-authoritative appointment references.
- Provides state lookup and retry contracts for scheduling orchestration, messaging orchestration, schedule configuration, and audit/compliance services.
- Supports retention controls, encryption at rest, backup configuration, and access boundaries for custom integration services.

## PHI Handling

- Operational PHI outside the EHR is retained for 90 days unless required longer by an approved vendor workflow.
- Appointment history and authoritative patient records remain in the EHR/practice-management system.
- Data models should separate audit references from PHI-bearing operational records so audit evidence can remain available after operational data expires.
- Access must be least-privilege and limited to services with a documented operational need.
