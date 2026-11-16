# Runbooks

These runbooks provide release-readiness placeholders for the clinic booking platform. They align with the ARD deployment model: managed HIPAA-eligible hosting, PostgreSQL database, secrets, logs, metrics, alerts, backups, and CI/CD deployment targets covered by signed BAAs before production use.

Required production runbooks:

- `backup-restore.md` documents backup and restore readiness.
- `rollback.md` documents application and database rollback readiness.
- `security-exception.md` documents the Security Reviewer exception process for high and critical dependency/CVE findings.

Each runbook must be completed with the selected cloud provider, CI/CD platform, IaC tool, environment names, RPO/RTO, retention duration, and reviewer assignments before production launch.
