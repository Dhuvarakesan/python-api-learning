CREATE TABLE IF NOT EXISTS verification_records (
  id uuid PRIMARY KEY,
  request_id uuid,
  channel text NOT NULL,
  destination_hash text NOT NULL,
  otp_hash text NOT NULL,
  status text NOT NULL,
  attempts integer NOT NULL DEFAULT 0,
  expires_at timestamptz NOT NULL,
  verified_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT verification_records_channel_check CHECK (channel IN ('sms', 'email', 'voice')),
  CONSTRAINT verification_records_status_check CHECK (status IN ('pending', 'verified', 'expired', 'locked'))
);

CREATE INDEX IF NOT EXISTS idx_verification_records_request_id
  ON verification_records (request_id);

CREATE INDEX IF NOT EXISTS idx_verification_records_expires_at
  ON verification_records (expires_at);

CREATE TABLE IF NOT EXISTS appointment_requests (
  id uuid PRIMARY KEY,
  patient_ref text NOT NULL,
  status text NOT NULL,
  reason text,
  requested_start_at timestamptz,
  requested_end_at timestamptz,
  selected_slot_id uuid,
  created_by text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT appointment_requests_status_check CHECK (
    status IN ('draft', 'submitted', 'pending_verification', 'ready_to_schedule', 'scheduled', 'cancelled', 'failed')
  )
);

CREATE INDEX IF NOT EXISTS idx_appointment_requests_patient_ref
  ON appointment_requests (patient_ref);

CREATE INDEX IF NOT EXISTS idx_appointment_requests_status
  ON appointment_requests (status);

CREATE TABLE IF NOT EXISTS appointment_slots (
  id uuid PRIMARY KEY,
  external_slot_id text,
  request_id uuid REFERENCES appointment_requests (id) ON DELETE SET NULL,
  location_id text NOT NULL,
  clinician_id text,
  starts_at timestamptz NOT NULL,
  ends_at timestamptz NOT NULL,
  status text NOT NULL,
  held_until timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT appointment_slots_status_check CHECK (status IN ('available', 'held', 'booked', 'released')),
  CONSTRAINT appointment_slots_time_check CHECK (ends_at > starts_at)
);

CREATE INDEX IF NOT EXISTS idx_appointment_slots_time
  ON appointment_slots (starts_at, ends_at);

CREATE INDEX IF NOT EXISTS idx_appointment_slots_status
  ON appointment_slots (status);

CREATE TABLE IF NOT EXISTS office_hours (
  id uuid PRIMARY KEY,
  location_id text NOT NULL,
  clinician_id text,
  day_of_week smallint NOT NULL,
  opens_at time NOT NULL,
  closes_at time NOT NULL,
  timezone text NOT NULL,
  is_active boolean NOT NULL DEFAULT true,
  effective_from date,
  effective_until date,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT office_hours_day_check CHECK (day_of_week BETWEEN 0 AND 6),
  CONSTRAINT office_hours_time_check CHECK (closes_at > opens_at)
);

CREATE INDEX IF NOT EXISTS idx_office_hours_lookup
  ON office_hours (location_id, clinician_id, day_of_week, is_active);

CREATE TABLE IF NOT EXISTS reminder_events (
  id uuid PRIMARY KEY,
  request_id uuid REFERENCES appointment_requests (id) ON DELETE CASCADE,
  channel text NOT NULL,
  status text NOT NULL,
  scheduled_for timestamptz NOT NULL,
  sent_at timestamptz,
  failure_reason text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT reminder_events_channel_check CHECK (channel IN ('sms', 'email', 'voice')),
  CONSTRAINT reminder_events_status_check CHECK (status IN ('pending', 'sent', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_reminder_events_due
  ON reminder_events (status, scheduled_for);

CREATE TABLE IF NOT EXISTS audit_events (
  id uuid PRIMARY KEY,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  actor_type text NOT NULL,
  actor_id text,
  action text NOT NULL,
  resource_type text NOT NULL,
  resource_id text,
  correlation_id text,
  ip_address inet,
  user_agent text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT audit_events_actor_type_check CHECK (actor_type IN ('patient', 'staff', 'system', 'integration'))
);

CREATE INDEX IF NOT EXISTS idx_audit_events_occurred_at
  ON audit_events (occurred_at);

CREATE INDEX IF NOT EXISTS idx_audit_events_resource
  ON audit_events (resource_type, resource_id);

CREATE INDEX IF NOT EXISTS idx_audit_events_correlation_id
  ON audit_events (correlation_id);

CREATE TABLE IF NOT EXISTS integration_failures (
  id uuid PRIMARY KEY,
  integration_name text NOT NULL,
  operation text NOT NULL,
  status text NOT NULL,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  request_payload_hash text,
  response_status integer,
  error_code text,
  error_message text NOT NULL,
  retry_after timestamptz,
  resolved_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT integration_failures_status_check CHECK (status IN ('open', 'retrying', 'resolved', 'ignored'))
);

CREATE INDEX IF NOT EXISTS idx_integration_failures_status
  ON integration_failures (status, occurred_at);

CREATE INDEX IF NOT EXISTS idx_integration_failures_integration
  ON integration_failures (integration_name, operation);
