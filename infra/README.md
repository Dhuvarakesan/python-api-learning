# Infrastructure Placeholders

The infrastructure module will hold IaC, deployment environment definitions, and operational controls once the Engineering Lead selects the cloud platform, CI/CD platform, and IaC tool.

Production infrastructure must satisfy these ARD constraints:

- HIPAA-eligible managed services and signed BAAs for hosting, database, SMS, email, logging, monitoring, backups, and secrets.
- Separate dev, test, stage, and prod environments.
- CI/CD gates for build, lint, type check, tests, accessibility checks, dependency/CVE scan, and SBOM generation before production deployment.
- High and critical dependency/CVE findings block production unless the Security Reviewer exception runbook is followed.
- PostgreSQL backups, restore validation, structured logs, metrics, alerts, and rollback procedures are documented before production launch.

Open decisions to resolve before adding provider-specific IaC:

- Cloud provider and managed services.
- IaC tool.
- Runtime Node.js LTS version.
- PostgreSQL version.
- Backup retention, RPO, and RTO.
- Production deployment strategy.
