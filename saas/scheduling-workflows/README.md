# SaaS Scheduling Workflows

## Purpose

`saas/scheduling-workflows` holds versionable scheduling SaaS workflow artifacts for patient self-service booking, rescheduling, OTP verification, reminders, staff schedule views, and configuration screens where the selected vendor supports export/import.

## Owned Contracts

- Captures vendor workflow definitions, screen configuration, appointment type flows, reminder flow settings, role-aware staff actions, and integration/webhook mappings.
- Integrates with the integration edge, scheduling orchestration, messaging orchestration, schedule configuration adapter, and audit/compliance service.
- Emits or configures audit-relevant events for booking, rescheduling, reminder activity, configuration changes, and schedule access.

## PHI Handling

- Configure screens and workflows for minimum necessary PHI collection and display.
- Prefer references and vendor-managed protected fields over duplicated exported PHI.
- Do not commit production PHI, secrets, tokens, vendor credentials, or patient examples into this directory.
- Version exported workflow artifacts only when the vendor supports safe export/import without exposing PHI.
