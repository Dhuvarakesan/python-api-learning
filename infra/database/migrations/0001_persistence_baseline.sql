CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE staff_roles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  code text NOT NULL UNIQUE,
  display_name text NOT NULL,
  description text,
  is_system_role boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE staff_users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email text NOT NULL UNIQUE,
  display_name text NOT NULL,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'locked')),
  mfa_required boolean NOT NULL DEFAULT true,
  last_login_at timestamptz,
  deactivated_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE staff_user_roles (
  staff_user_id uuid NOT NULL REFERENCES staff_users(id) ON DELETE CASCADE,
  staff_role_id uuid NOT NULL REFERENCES staff_roles(id) ON DELETE RESTRICT,
  assigned_at timestamptz NOT NULL DEFAULT now(),
  assigned_by_staff_user_id uuid REFERENCES staff_users(id) ON DELETE SET NULL,
  PRIMARY KEY (staff_user_id, staff_role_id)
);

CREATE TABLE appointment_type_rules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  appointment_type_external_id text NOT NULL UNIQUE,
  appointment_type_name text NOT NULL,
  is_online_bookable boolean NOT NULL DEFAULT false,
  confirmation_mode text NOT NULL DEFAULT 'manual_approval'
    CHECK (confirmation_mode IN ('instant_confirmation', 'manual_approval')),
  is_online_reschedulable boolean NOT NULL DEFAULT false,
  rescheduling_cutoff_minutes integer NOT NULL DEFAULT 1440 CHECK (rescheduling_cutoff_minutes >= 0),
  pilot_enabled boolean NOT NULL DEFAULT false,
  created_by_staff_user_id uuid REFERENCES staff_users(id) ON DELETE SET NULL,
  updated_by_staff_user_id uuid REFERENCES staff_users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE patient_scheduling_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scheduling_system_patient_id text,
  first_name text NOT NULL,
  last_name text NOT NULL,
  date_of_birth date NOT NULL,
  email text,
  phone_e164 text,
  preferred_contact_channel text CHECK (preferred_contact_channel IN ('sms', 'email', 'both')),
  demographics jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE appointments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  scheduling_system_appointment_id text UNIQUE,
  patient_scheduling_profile_id uuid REFERENCES patient_scheduling_profiles(id) ON DELETE SET NULL,
  appointment_type_rule_id uuid REFERENCES appointment_type_rules(id) ON DELETE SET NULL,
  provider_external_id text,
  provider_display_name text,
  location_name text,
  starts_at timestamptz NOT NULL,
  ends_at timestamptz NOT NULL,
  status text NOT NULL DEFAULT 'scheduled'
    CHECK (status IN ('scheduled', 'confirmed', 'pending', 'cancelled', 'completed', 'no_show')),
  source text NOT NULL DEFAULT 'manual_pilot'
    CHECK (source IN ('manual_pilot', 'scheduling_system', 'online_booking')),
  logistics jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CHECK (ends_at > starts_at)
);

CREATE TABLE booking_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_scheduling_profile_id uuid NOT NULL REFERENCES patient_scheduling_profiles(id) ON DELETE RESTRICT,
  appointment_type_rule_id uuid REFERENCES appointment_type_rules(id) ON DELETE SET NULL,
  appointment_id uuid REFERENCES appointments(id) ON DELETE SET NULL,
  requested_slot_start_at timestamptz,
  requested_slot_end_at timestamptz,
  status text NOT NULL DEFAULT 'submitted'
    CHECK (status IN ('submitted', 'confirmed', 'pending_staff_approval', 'no_availability_request_created', 'declined', 'cancelled')),
  submission_source text NOT NULL DEFAULT 'patient_web',
  idempotency_key text NOT NULL,
  submitted_at timestamptz NOT NULL DEFAULT now(),
  reviewed_at timestamptz,
  reviewed_by_staff_user_id uuid REFERENCES staff_users(id) ON DELETE SET NULL,
  staff_notes text,
  request_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (idempotency_key)
);

CREATE TABLE verification_attempts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_scheduling_profile_id uuid REFERENCES patient_scheduling_profiles(id) ON DELETE SET NULL,
  appointment_id uuid REFERENCES appointments(id) ON DELETE SET NULL,
  purpose text NOT NULL DEFAULT 'rescheduling' CHECK (purpose IN ('rescheduling')),
  destination_channel text NOT NULL CHECK (destination_channel IN ('sms', 'email')),
  destination_fingerprint text NOT NULL,
  status text NOT NULL DEFAULT 'started'
    CHECK (status IN ('started', 'code_sent', 'succeeded', 'failed', 'locked', 'expired')),
  attempts_count integer NOT NULL DEFAULT 0 CHECK (attempts_count >= 0),
  last_attempt_at timestamptz,
  expires_at timestamptz NOT NULL,
  succeeded_at timestamptz,
  failure_reason_category text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE reschedule_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  appointment_id uuid NOT NULL REFERENCES appointments(id) ON DELETE RESTRICT,
  patient_scheduling_profile_id uuid REFERENCES patient_scheduling_profiles(id) ON DELETE SET NULL,
  verification_attempt_id uuid REFERENCES verification_attempts(id) ON DELETE SET NULL,
  appointment_type_rule_id uuid REFERENCES appointment_type_rules(id) ON DELETE SET NULL,
  requested_new_start_at timestamptz NOT NULL,
  requested_new_end_at timestamptz NOT NULL,
  status text NOT NULL DEFAULT 'submitted'
    CHECK (status IN ('submitted', 'confirmed', 'pending_staff_approval', 'blocked_by_cutoff', 'not_eligible', 'declined', 'cancelled')),
  idempotency_key text NOT NULL,
  submitted_at timestamptz NOT NULL DEFAULT now(),
  reviewed_at timestamptz,
  reviewed_by_staff_user_id uuid REFERENCES staff_users(id) ON DELETE SET NULL,
  staff_notes text,
  request_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (idempotency_key),
  CHECK (requested_new_end_at > requested_new_start_at)
);

CREATE TABLE reminder_preferences (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_scheduling_profile_id uuid NOT NULL REFERENCES patient_scheduling_profiles(id) ON DELETE RESTRICT,
  appointment_id uuid REFERENCES appointments(id) ON DELETE CASCADE,
  sms_enabled boolean NOT NULL DEFAULT false,
  email_enabled boolean NOT NULL DEFAULT false,
  sms_destination_fingerprint text,
  email_destination_fingerprint text,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'cancelled')),
  consent_captured_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CHECK (sms_enabled OR email_enabled)
);

CREATE TABLE reminder_delivery_attempts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  reminder_preference_id uuid NOT NULL REFERENCES reminder_preferences(id) ON DELETE CASCADE,
  appointment_id uuid REFERENCES appointments(id) ON DELETE SET NULL,
  channel text NOT NULL CHECK (channel IN ('sms', 'email')),
  provider_message_id text,
  status text NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'sent', 'delivered', 'failed', 'cancelled', 'retry_scheduled')),
  scheduled_for timestamptz NOT NULL,
  attempted_at timestamptz,
  next_retry_at timestamptz,
  retry_count integer NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
  failure_reason_category text,
  idempotency_key text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE audit_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type text NOT NULL,
  actor_type text NOT NULL CHECK (actor_type IN ('patient', 'staff_user', 'system', 'external_provider')),
  actor_staff_user_id uuid REFERENCES staff_users(id) ON DELETE SET NULL,
  patient_scheduling_profile_id uuid REFERENCES patient_scheduling_profiles(id) ON DELETE SET NULL,
  appointment_id uuid REFERENCES appointments(id) ON DELETE SET NULL,
  booking_request_id uuid REFERENCES booking_requests(id) ON DELETE SET NULL,
  reschedule_request_id uuid REFERENCES reschedule_requests(id) ON DELETE SET NULL,
  reminder_delivery_attempt_id uuid REFERENCES reminder_delivery_attempts(id) ON DELETE SET NULL,
  verification_attempt_id uuid REFERENCES verification_attempts(id) ON DELETE SET NULL,
  entity_type text,
  entity_id uuid,
  outcome text NOT NULL DEFAULT 'success' CHECK (outcome IN ('success', 'failure', 'denied')),
  safe_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX appointment_type_rules_bookable_idx ON appointment_type_rules (is_online_bookable, pilot_enabled);
CREATE INDEX patient_scheduling_profiles_name_dob_idx ON patient_scheduling_profiles (last_name, first_name, date_of_birth);
CREATE INDEX appointments_schedule_idx ON appointments (starts_at, provider_external_id, status);
CREATE INDEX booking_requests_status_submitted_idx ON booking_requests (status, submitted_at);
CREATE INDEX reschedule_requests_status_submitted_idx ON reschedule_requests (status, submitted_at);
CREATE INDEX reminder_preferences_appointment_idx ON reminder_preferences (appointment_id);
CREATE INDEX reminder_delivery_attempts_due_idx ON reminder_delivery_attempts (status, scheduled_for);
CREATE INDEX verification_attempts_lookup_idx ON verification_attempts (appointment_id, status, expires_at);
CREATE INDEX audit_events_occurred_idx ON audit_events (occurred_at);
CREATE INDEX audit_events_type_idx ON audit_events (event_type, outcome);

CREATE TRIGGER staff_roles_set_updated_at
  BEFORE UPDATE ON staff_roles
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER staff_users_set_updated_at
  BEFORE UPDATE ON staff_users
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER appointment_type_rules_set_updated_at
  BEFORE UPDATE ON appointment_type_rules
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER patient_scheduling_profiles_set_updated_at
  BEFORE UPDATE ON patient_scheduling_profiles
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER appointments_set_updated_at
  BEFORE UPDATE ON appointments
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER booking_requests_set_updated_at
  BEFORE UPDATE ON booking_requests
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER verification_attempts_set_updated_at
  BEFORE UPDATE ON verification_attempts
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER reschedule_requests_set_updated_at
  BEFORE UPDATE ON reschedule_requests
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER reminder_preferences_set_updated_at
  BEFORE UPDATE ON reminder_preferences
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER reminder_delivery_attempts_set_updated_at
  BEFORE UPDATE ON reminder_delivery_attempts
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER audit_events_set_updated_at
  BEFORE UPDATE ON audit_events
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO staff_roles (code, display_name, description, is_system_role)
VALUES
  ('clinic_admin', 'Clinic Administrator', 'Can manage appointment rules and staff access.', true),
  ('scheduling_staff', 'Scheduling Staff', 'Can review booking and reschedule requests and view the daily schedule.', true)
ON CONFLICT (code) DO NOTHING;

