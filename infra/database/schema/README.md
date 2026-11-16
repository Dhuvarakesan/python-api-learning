# Database Schema Baseline

This schema is managed by SQL migrations in `infra/database/migrations`.

Local migrations do not require production secrets. The runner uses the first
available connection string in this order:

1. `DATABASE_URL`
2. `LOCAL_DATABASE_URL`
3. `postgres://postgres:postgres@localhost:5432/clinic_booking`

Run from the repository root:

```sh
npm run db:migrate -w @clinic/api
```

The baseline migration creates storage for:

- `appointment_type_rules`
- `patient_scheduling_profiles`
- `appointments`
- `booking_requests`
- `reschedule_requests`
- `reminder_preferences`
- `reminder_delivery_attempts`
- `staff_users`
- `staff_roles`
- `verification_attempts`
- `audit_events`

Restricted scheduling, appointment, reminder, verification, and audit tables
include `created_at` and `updated_at` timestamps for audit review and KPI
measurement.
