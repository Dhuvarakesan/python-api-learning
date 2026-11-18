export const initialSchemaSql = `
CREATE TABLE IF NOT EXISTS staff_users (
  id text PRIMARY KEY,
  email text NOT NULL UNIQUE,
  display_name text NOT NULL,
  role text NOT NULL CHECK (role IN ('front_desk', 'clinic_admin', 'technical_support')),
  mfa_enabled boolean NOT NULL DEFAULT false,
  active boolean NOT NULL DEFAULT true,
  last_authenticated_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE staff_users IS 'DATA-007 Staff User and Role. Confidential identity and authorization state.';
COMMENT ON COLUMN staff_users.email IS 'Confidential staff identifier; do not write to application logs.';

CREATE TABLE IF NOT EXISTS appointment_types (
  id text PRIMARY KEY,
  ehr_appointment_type_id text UNIQUE,
  name text NOT NULL,
  is_online_bookable boolean NOT NULL DEFAULT false,
  requires_new_patient_intake boolean NOT NULL DEFAULT false,
  requires_insurance boolean NOT NULL DEFAULT false,
  default_duration_minutes integer NOT NULL CHECK (default_duration_minutes > 0),
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE appointment_types IS 'DATA-006 Appointment Type and Scheduling Configuration. Internal configuration.';

CREATE TABLE IF NOT EXISTS scheduling_configurations (
  id text PRIMARY KEY,
  appointment_type_id text NOT NULL REFERENCES appointment_types(id),
  reschedule_cutoff_hours integer NOT NULL CHECK (reschedule_cutoff_hours >= 0),
  booking_window_days integer NOT NULL CHECK (booking_window_days > 0),
  minimum_notice_hours integer NOT NULL CHECK (minimum_notice_hours >= 0),
  updated_by_staff_user_id text NOT NULL REFERENCES staff_users(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (appointment_type_id)
);

COMMENT ON TABLE scheduling_configurations IS 'DATA-006 Clinic scheduling rules. Internal configuration with audited changes.';

CREATE TABLE IF NOT EXISTS patient_scheduling_profiles (
  id text PRIMARY KEY,
  ehr_patient_id text UNIQUE,
  is_established_patient boolean NOT NULL DEFAULT false,
  legal_name text NOT NULL,
  date_of_birth date NOT NULL,
  phone_number text,
  email text,
  preferred_contact_channel text CHECK (preferred_contact_channel IN ('sms', 'email', 'phone')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE patient_scheduling_profiles IS 'DATA-002 Patient Scheduling Profile. Restricted PHI.';
COMMENT ON COLUMN patient_scheduling_profiles.legal_name IS 'Restricted PHI. Never log field value.';
COMMENT ON COLUMN patient_scheduling_profiles.date_of_birth IS 'Restricted PHI. Never log field value.';
COMMENT ON COLUMN patient_scheduling_profiles.phone_number IS 'Restricted PHI. Never log field value.';
COMMENT ON COLUMN patient_scheduling_profiles.email IS 'Restricted PHI. Never log field value.';

CREATE TABLE IF NOT EXISTS appointments (
  id text PRIMARY KEY,
  patient_profile_id text NOT NULL REFERENCES patient_scheduling_profiles(id),
  appointment_type_id text NOT NULL REFERENCES appointment_types(id),
  provider_external_id text,
  provider_display_name text NOT NULL,
  starts_at timestamptz NOT NULL,
  ends_at timestamptz NOT NULL,
  status text NOT NULL CHECK (status IN ('requested', 'confirmed', 'rescheduled', 'cancelled', 'completed', 'no_show')),
  booking_source text NOT NULL CHECK (booking_source IN ('patient_web', 'staff', 'ehr_sync', 'csv_import')),
  ehr_appointment_id text UNIQUE,
  idempotency_key text UNIQUE,
  cancellation_reason text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CHECK (ends_at > starts_at)
);

COMMENT ON TABLE appointments IS 'DATA-001 Appointment. Restricted scheduling record.';
COMMENT ON COLUMN appointments.provider_display_name IS 'Minimum necessary staff schedule display data; avoid logging with patient context.';
COMMENT ON COLUMN appointments.cancellation_reason IS 'Restricted PHI when patient supplied. Never log field value.';

CREATE INDEX IF NOT EXISTS appointments_day_idx ON appointments (starts_at, status);
CREATE INDEX IF NOT EXISTS appointments_patient_idx ON appointments (patient_profile_id, starts_at DESC);

CREATE TABLE IF NOT EXISTS intake_submissions (
  id text PRIMARY KEY,
  patient_profile_id text NOT NULL REFERENCES patient_scheduling_profiles(id),
  appointment_id text REFERENCES appointments(id),
  submitted_by_actor_id text NOT NULL,
  intake_payload jsonb NOT NULL,
  submitted_at timestamptz NOT NULL DEFAULT now(),
  exported_at timestamptz
);

COMMENT ON TABLE intake_submissions IS 'DATA-003 New-Patient Intake. Restricted PHI workflow store.';
COMMENT ON COLUMN intake_submissions.intake_payload IS 'Restricted PHI. Never log field value.';

CREATE TABLE IF NOT EXISTS insurance_submissions (
  id text PRIMARY KEY,
  patient_profile_id text NOT NULL REFERENCES patient_scheduling_profiles(id),
  appointment_id text REFERENCES appointments(id),
  payer_name text NOT NULL,
  member_id text NOT NULL,
  group_number text,
  policyholder_name text,
  insurance_payload jsonb NOT NULL,
  submitted_at timestamptz NOT NULL DEFAULT now(),
  exported_at timestamptz
);

COMMENT ON TABLE insurance_submissions IS 'DATA-003 Insurance Submission. Restricted PHI workflow store.';
COMMENT ON COLUMN insurance_submissions.payer_name IS 'Restricted PHI. Never log field value.';
COMMENT ON COLUMN insurance_submissions.member_id IS 'Restricted PHI. Never log field value.';
COMMENT ON COLUMN insurance_submissions.group_number IS 'Restricted PHI. Never log field value.';
COMMENT ON COLUMN insurance_submissions.policyholder_name IS 'Restricted PHI. Never log field value.';
COMMENT ON COLUMN insurance_submissions.insurance_payload IS 'Restricted PHI. Never log field value.';

CREATE TABLE IF NOT EXISTS reminder_consents (
  id text PRIMARY KEY,
  patient_profile_id text NOT NULL REFERENCES patient_scheduling_profiles(id),
  channel text NOT NULL CHECK (channel IN ('sms', 'email')),
  status text NOT NULL CHECK (status IN ('granted', 'declined', 'opted_out')),
  consented_at timestamptz,
  opted_out_at timestamptz,
  source text NOT NULL CHECK (source IN ('patient_web', 'staff', 'vendor_callback', 'ehr_sync')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (patient_profile_id, channel)
);

COMMENT ON TABLE reminder_consents IS 'DATA-004 Reminder Consent and Opt-Out. Restricted consent record.';

CREATE TABLE IF NOT EXISTS reminder_events (
  id text PRIMARY KEY,
  appointment_id text NOT NULL REFERENCES appointments(id),
  patient_profile_id text NOT NULL REFERENCES patient_scheduling_profiles(id),
  consent_id text REFERENCES reminder_consents(id),
  channel text NOT NULL CHECK (channel IN ('sms', 'email')),
  scheduled_for timestamptz NOT NULL,
  status text NOT NULL CHECK (status IN ('scheduled', 'not_sent_missing_consent', 'not_sent_opted_out', 'queued', 'sent', 'delivered', 'failed', 'cancelled')),
  vendor_name text,
  vendor_message_id text,
  vendor_correlation_id text,
  failure_reason text,
  attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  last_attempted_at timestamptz,
  delivered_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE reminder_events IS 'DATA-005 Reminder Event and Delivery Status. Restricted operational record.';
COMMENT ON COLUMN reminder_events.failure_reason IS 'May contain vendor-provided PHI. Never log field value.';

CREATE INDEX IF NOT EXISTS reminder_events_due_idx ON reminder_events (scheduled_for, status);
CREATE INDEX IF NOT EXISTS reminder_events_appointment_idx ON reminder_events (appointment_id);

CREATE TABLE IF NOT EXISTS integration_states (
  id text PRIMARY KEY,
  integration_name text NOT NULL,
  resource_type text NOT NULL CHECK (resource_type IN ('appointment', 'patient', 'appointment_type', 'provider', 'export')),
  local_resource_id text,
  external_resource_id text,
  sync_status text NOT NULL CHECK (sync_status IN ('pending', 'synced', 'failed', 'reconciliation_required')),
  retry_count integer NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
  last_synced_at timestamptz,
  next_retry_at timestamptz,
  safe_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE integration_states IS 'Integration sync state for EHR/API/CSV boundaries. Store identifiers and safe metadata only.';
COMMENT ON COLUMN integration_states.safe_metadata IS 'Safe metadata only. Do not store PHI values here.';

CREATE INDEX IF NOT EXISTS integration_states_retry_idx ON integration_states (sync_status, next_retry_at);
CREATE UNIQUE INDEX IF NOT EXISTS integration_states_external_uidx
  ON integration_states (integration_name, resource_type, external_resource_id)
  WHERE external_resource_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS audit_events (
  id text PRIMARY KEY,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  actor_id text,
  actor_type text NOT NULL CHECK (actor_type IN ('patient', 'staff', 'system')),
  action text NOT NULL CHECK (action IN ('booking_created', 'appointment_rescheduled', 'schedule_viewed', 'consent_changed', 'opt_out_recorded', 'configuration_changed', 'authentication_event', 'authorization_denied', 'integration_event', 'reminder_status_changed')),
  target_type text NOT NULL CHECK (target_type IN ('appointment', 'patient_profile', 'reminder_consent', 'configuration', 'staff_user', 'integration_state', 'reminder_event')),
  target_id text,
  outcome text NOT NULL CHECK (outcome IN ('success', 'failure', 'denied')),
  correlation_id text NOT NULL,
  safe_metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE audit_events IS 'DATA-008 Audit Event. Confidential audit store; use resource identifiers and safe metadata instead of PHI field values.';
COMMENT ON COLUMN audit_events.safe_metadata IS 'Safe metadata only. Do not store PHI values here.';

CREATE INDEX IF NOT EXISTS audit_events_occurred_at_idx ON audit_events (occurred_at DESC);
CREATE INDEX IF NOT EXISTS audit_events_target_idx ON audit_events (target_type, target_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS audit_events_actor_idx ON audit_events (actor_type, actor_id, occurred_at DESC);
`;
