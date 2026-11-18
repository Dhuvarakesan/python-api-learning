import type { RestrictedPhi } from '../scheduling/types';

export type ReminderChannel = 'sms' | 'email';

export type ReminderConsentStatus = 'granted' | 'declined' | 'opted_out';

export type ReminderDeliveryStatus =
  | 'scheduled'
  | 'not_sent_missing_consent'
  | 'not_sent_opted_out'
  | 'queued'
  | 'sent'
  | 'delivered'
  | 'failed'
  | 'cancelled';

export interface ReminderConsent {
  id: string;
  patientProfileId: string;
  channel: ReminderChannel;
  status: ReminderConsentStatus;
  consentedAt: string | null;
  optedOutAt: string | null;
  source: 'patient_web' | 'staff' | 'vendor_callback' | 'ehr_sync';
  createdAt: string;
  updatedAt: string;
}

export interface ReminderEvent {
  id: string;
  appointmentId: string;
  patientProfileId: string;
  consentId: string | null;
  channel: ReminderChannel;
  scheduledFor: string;
  status: ReminderDeliveryStatus;
  vendorName: string | null;
  vendorMessageId: string | null;
  vendorCorrelationId: string | null;
  failureReason: RestrictedPhi<string | null>;
  attemptCount: number;
  lastAttemptedAt: string | null;
  deliveredAt: string | null;
  createdAt: string;
  updatedAt: string;
}
