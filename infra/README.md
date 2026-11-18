# Infrastructure Placeholders

This directory reserves the deployment boundary for the clinic booking platform. Production infrastructure is intentionally not provisioned yet because cloud, messaging, identity, scheduling, and deployment vendors are still TBD.

Production deployment is blocked until all vendors that may store, process, transmit, observe, or support PHI are confirmed as HIPAA-eligible and covered by signed Business Associate Agreements (BAAs). The same gate applies to CI/CD services, managed databases, logging, monitoring, alerting, identity providers, email/SMS providers, and backup systems.

## Required Gates

- Dependency and CVE scanning: `npm run security:scan` currently uses `npm audit --audit-level=high` and should be expanded with the selected SCA platform.
- SBOM generation: `npm run security:sbom` emits a CycloneDX SBOM at `sbom.cdx.json` for release evidence and downstream vulnerability tooling.
- IaC validation: `npm run iac:validate` is a placeholder hook. Replace it with `terraform fmt`, `terraform validate`, policy checks, or equivalent vendor-native validation when modules are added.
- Environment promotion: `npm run promotion:check` is a placeholder hook. Replace it with explicit promotion controls for development, staging, and production once deployment environments exist.

## Production Baseline

Before enabling production PHI processing, the deployment design must confirm:

- HTTPS/TLS for all external and internal transit paths.
- Encryption at rest for databases, backups, object storage, logs, and queues.
- RBAC for staff, service accounts, CI/CD identities, and operators.
- PHI-redacted logs, traces, metrics, alerts, and support exports.
- Separate application and migration database credentials.
- Tested backup restore procedures and auditable change history.

Confirmed appointments remain in the existing scheduling system as the system of record for the first release. Infrastructure modules must preserve that boundary and route scheduler interactions through application adapters.
