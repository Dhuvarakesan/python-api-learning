# Rollback

This rollback guide provides controlled release-readiness placeholders. It does
not assume production infrastructure values that remain unresolved.

## Rollback prerequisites

- Cloud provider: TBD.
- Infrastructure-as-code tool: TBD.
- Production BAA coverage for rollback, backup, logging, and deployment systems:
  TBD.
- Release artifact storage location: TBD.
- Database migration strategy: TBD.
- Backup retention duration: TBD.
- Audit log retention duration: TBD.
- Operational log retention duration: TBD.
- Recovery time objective: TBD.
- Recovery point objective: TBD.

## Rollback steps

1. Declare rollback ownership and incident channel: TBD.
2. Identify the last known-good application artifact: TBD.
3. Confirm database migration compatibility and rollback path: TBD.
4. Pause new production promotion while rollback is active: TBD.
5. Deploy the last known-good artifact through the approved pipeline: TBD.
6. Restore or migrate data only through the approved recovery process: TBD.
7. Run smoke tests for `/patient`, `/staff`, and `/staff/admin`: TBD.
8. Confirm audit logging and PHI redaction are functioning: TBD.
9. Capture release evidence and incident notes: TBD.
10. Resume normal deployment flow after approval: TBD.

## Security gate

Critical and High security findings must block release and rollback promotion
unless formally risk-accepted. Risk acceptance must document owner, rationale,
compensating controls, expiration date, and approval evidence.
