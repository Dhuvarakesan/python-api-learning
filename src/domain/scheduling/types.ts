export type RestrictedPhi<T> = T & {
  readonly __classification: 'restricted-phi';
};

export type AppointmentStatus =
  | 'requested'
  | 'confirmed'
  | 'rescheduled'
  | 'cancelled'
  | 'completed'
  | 'no_show';

export type BookingSource = 'patient_web' | 'staff' | 'ehr_sync' | 'csv_import';

export type StaffRole =
  | 'front_desk'
  | 'clinic_admin'
  | 'technical_support';

export interface OperationContext {
  actorId: string;
  actorType: 'patient' | 'staff' | 'system';
  correlationId: string;
}

export interface PatientSchedulingProfile {
  id: string;
  ehrPatientId: string | null;
  isEstablishedPatient: boolean;
  legalName: RestrictedPhi<string>;
  dateOfBirth: RestrictedPhi<string>;
  phoneNumber: RestrictedPhi<string | null>;
  email: RestrictedPhi<string | null>;
  preferredContactChannel: 'sms' | 'email' | 'phone' | null;
  createdAt: string;
  updatedAt: string;
}

export interface AppointmentType {
  id: string;
  ehrAppointmentTypeId: string | null;
  name: string;
  isOnlineBookable: boolean;
  requiresNewPatientIntake: boolean;
  requiresInsurance: boolean;
  defaultDurationMinutes: number;
  active: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface SchedulingConfiguration {
  id: string;
  appointmentTypeId: string;
  rescheduleCutoffHours: number;
  bookingWindowDays: number;
  minimumNoticeHours: number;
  updatedByStaffUserId: string;
  createdAt: string;
  updatedAt: string;
}

export interface Appointment {
  id: string;
  patientProfileId: string;
  appointmentTypeId: string;
  providerExternalId: string | null;
  providerDisplayName: string;
  startsAt: string;
  endsAt: string;
  status: AppointmentStatus;
  bookingSource: BookingSource;
  ehrAppointmentId: string | null;
  idempotencyKey: string | null;
  cancellationReason: RestrictedPhi<string | null>;
  createdAt: string;
  updatedAt: string;
}

export interface IntakeSubmission {
  id: string;
  patientProfileId: string;
  appointmentId: string | null;
  submittedByActorId: string;
  intakePayload: RestrictedPhi<Record<string, unknown>>;
  submittedAt: string;
  exportedAt: string | null;
}

export interface InsuranceSubmission {
  id: string;
  patientProfileId: string;
  appointmentId: string | null;
  payerName: RestrictedPhi<string>;
  memberId: RestrictedPhi<string>;
  groupNumber: RestrictedPhi<string | null>;
  policyholderName: RestrictedPhi<string | null>;
  insurancePayload: RestrictedPhi<Record<string, unknown>>;
  submittedAt: string;
  exportedAt: string | null;
}

export interface StaffUser {
  id: string;
  email: string;
  displayName: string;
  role: StaffRole;
  mfaEnabled: boolean;
  active: boolean;
  lastAuthenticatedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface IntegrationState {
  id: string;
  integrationName: string;
  resourceType: 'appointment' | 'patient' | 'appointment_type' | 'provider' | 'export';
  localResourceId: string | null;
  externalResourceId: string | null;
  syncStatus: 'pending' | 'synced' | 'failed' | 'reconciliation_required';
  retryCount: number;
  lastSyncedAt: string | null;
  nextRetryAt: string | null;
  safeMetadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface AuditEvent {
  id: string;
  occurredAt: string;
  actorId: string | null;
  actorType: OperationContext['actorType'];
  action:
    | 'booking_created'
    | 'appointment_rescheduled'
    | 'schedule_viewed'
    | 'consent_changed'
    | 'opt_out_recorded'
    | 'configuration_changed'
    | 'authentication_event'
    | 'authorization_denied'
    | 'integration_event'
    | 'reminder_status_changed';
  targetType:
    | 'appointment'
    | 'patient_profile'
    | 'reminder_consent'
    | 'configuration'
    | 'staff_user'
    | 'integration_state'
    | 'reminder_event';
  targetId: string | null;
  outcome: 'success' | 'failure' | 'denied';
  correlationId: string;
  safeMetadata: Record<string, unknown>;
}
