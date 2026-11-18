# Database Deployment Requirements

The application stores request, slot, verification, reminder, audit, office-hour, and integration-failure records in a managed relational database.

Deployment environments must provide:

- Encryption at rest from the managed database service or storage layer.
- TLS for database connections in any environment that processes PHI.
- Automated backups with tested point-in-time restore for production.
- Access controls that limit migration execution and application read/write credentials separately.
- PHI-redacted database logs, query sampling, and operational exports.

Encryption keys, backup credentials, and restore material are deployment concerns. They must not be committed as application configuration or embedded in migration files.
