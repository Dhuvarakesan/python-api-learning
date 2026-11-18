export type AppointmentStatus =
  | 'requested'
  | 'held'
  | 'confirmed'
  | 'cancelled'
  | 'expired';

export type AppointmentChannel = 'web' | 'phone' | 'staff';

export interface PatientContact {
  readonly firstName: string;
  readonly lastName: string;
  readonly email?: string;
  readonly phone?: string;
}

export interface AppointmentSlot {
  readonly id: string;
  readonly startsAt: string;
  readonly endsAt: string;
  readonly clinicianId: string;
  readonly locationId: string;
}

export interface AppointmentRequest {
  readonly id: string;
  readonly patient: PatientContact;
  readonly requestedSlotId?: string;
  readonly reasonForVisit?: string;
  readonly channel: AppointmentChannel;
  readonly status: AppointmentStatus;
  readonly createdAt: string;
}

export interface StaffUser {
  readonly id: string;
  readonly displayName: string;
  readonly roles: readonly string[];
}

export interface AuditEvent {
  readonly id: string;
  readonly actorId: string;
  readonly action: string;
  readonly occurredAt: string;
  readonly subjectType: 'appointment_request' | 'slot' | 'user';
  readonly subjectId: string;
}
