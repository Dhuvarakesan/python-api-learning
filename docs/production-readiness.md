# Production Readiness

This checklist records release evidence for the clinic booking application without
assuming production decisions that still require architecture review or formal
approval.

## Quality gates

- Type check: `npm run typecheck`
- Lint: `npm run lint`
- Unit and integration tests: `npm run test`
- Accessibility check: `npm run accessibility`
- Production build: `npm run build`
- Security scan hook: `npm run security:scan`
- SBOM generation hook: `npm run sbom:generate`

Critical and High security findings block release unless they are formally
risk-accepted with owner, rationale, compensating controls, expiration date, and
approval evidence. Risk acceptance must be completed before production deploy.

## Unresolved production decisions

The following ARD-required production decisions are intentionally placeholders:

- Cloud provider: TBD.
- Infrastructure-as-code tool: TBD.
- Production BAA coverage: TBD, including covered services and account boundary.
- Uptime target: TBD.
- Performance target: TBD, including page response and booking workflow targets.
- Data retention durations: TBD for appointments, audit logs, operational logs,
  backups, and deleted or expired patient access artifacts.
- Backup cadence and restore point objective: TBD.
- Recovery time objective: TBD.
- Monitoring and alert routing: TBD.
- Production environment promotion and approval policy: TBD.

## Release evidence

Each release candidate should retain CI output, accessibility report, test
results, production build result, security scan result, generated SBOM, and any
formal risk-acceptance records for unresolved Critical or High findings.
