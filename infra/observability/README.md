# Observability

## Purpose

`infra/observability` owns COMP-009, the infrastructure area for HIPAA-safe logs, metrics, traces, dashboards, and alerts for custom integration services and imported vendor audit/status events.

## Owned Contracts

- Receives application logs, metrics, traces, SaaS/EHR/messaging status events, and deployment events.
- Produces alerts, dashboards, incident diagnostics, uptime/performance evidence, and operations failure visibility.
- Consumes ARD-API-006 `ops.integration_failure` events and summary audit/status exports where available.
- Defines logging, tracing, metric, alerting, and dashboard artifacts for service teams.

## PHI Handling

- Logs, traces, and metrics must be PHI-minimized by design and must not include OTP codes, full message bodies, secrets, access tokens, or full patient contact details.
- Observability vendors or runtimes that receive PHI or PHI-adjacent events must be HIPAA-eligible with BAAs where required.
- Use correlation IDs for diagnostics instead of patient identifiers.
- Retention must follow approved operational and audit policies, with audit logs retained for 6 years where applicable.
