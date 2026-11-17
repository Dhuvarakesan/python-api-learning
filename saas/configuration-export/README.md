# SaaS Configuration Export

## Purpose

`saas/configuration-export` holds versionable scheduling SaaS configuration exports where the selected vendor supports safe export/import.

## Owned Contracts

- Captures appointment types, provider/resource mappings, availability rules, cutoff rules, reminder timing, access roles, webhook subscriptions, and integration settings.
- Provides reviewable configuration history for SaaS settings that affect scheduling behavior, staff access, messaging, and audit event production.
- Supports promotion between environments when vendor tooling allows reproducible export/import.

## PHI Handling

- Do not store production PHI, patient records, message bodies, secrets, tokens, or vendor credentials in exported configuration.
- Redact or replace PHI-like sample values before committing artifacts.
- Configuration change history supports audit evidence and follows the 6-year audit retention policy where it is part of audit records.
- Use environment-specific secret management outside this repository area for runtime credentials.
