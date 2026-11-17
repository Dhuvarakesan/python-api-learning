# Integration Edge

## Purpose

`services/integration-edge` owns COMP-002, the API gateway and integration edge for controlled traffic between the scheduling SaaS, EHR/practice-management system, messaging vendors, and internal integration services.

## Owned Contracts

- Accepts SaaS webhook/API calls, EHR callbacks, messaging delivery callbacks, and authenticated service requests.
- Routes requests to scheduling orchestration, messaging orchestration, schedule configuration, EHR adapter, and audit/compliance services.
- Applies authentication, authorization, rate limiting, correlation IDs, and normalized integration event envelopes.
- Emits audit and operations events through ARD-API-006 topics where applicable: `audit.booking`, `audit.reschedule`, `audit.reminder`, `audit.configuration`, `audit.schedule_access`, and `ops.integration_failure`.

## PHI Handling

- Treat all appointment, patient contact, callback, and scheduling payloads as PHI or PHI-adjacent unless classified otherwise.
- Keep request/response logs PHI-minimized and redacted; do not log OTP codes, message bodies, access tokens, secrets, or full patient contact details.
- Persist no operational PHI here unless explicitly required by an implementation task; use retention-limited stores for state.
- Use HIPAA-eligible runtimes and vendors with BAAs wherever PHI is handled.
