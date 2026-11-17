# Schedule Configuration Adapter

## Purpose

`services/schedule-config-adapter` owns COMP-005, the adapter that normalizes staff/provider schedule views and configuration data between SaaS workflows and the EHR/practice-management system.

## Owned Contracts

- Accepts staff/provider role claims, schedule date/provider filters, appointment type configuration, availability rules, cutoff rules, and reminder timing.
- Produces provider-grouped schedule data, configuration updates, and schedule/configuration access audit events.
- Enforces role-aware access boundaries using federated identity claims supplied by the SaaS or integration edge.
- Emits ARD-API-006 events for configuration changes, schedule access, and integration failures.

## PHI Handling

- Treat schedule views, appointment references, provider assignments, and configuration changes as sensitive operational data.
- Minimize any cached schedule data and keep operational copies outside the EHR no longer than needed, never more than 90 days.
- Do not log patient identifiers, appointment details, or role claim payloads beyond minimum necessary audit metadata.
- Configuration history that supports audit evidence follows the 6-year audit retention policy.
