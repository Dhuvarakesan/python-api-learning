import type {
  Appointment,
  AppointmentType,
  AuditEvent,
  IntegrationState,
  InsuranceSubmission,
  IntakeSubmission,
  OperationContext,
  PatientSchedulingProfile,
  SchedulingConfiguration,
  StaffUser,
} from '../../domain/scheduling/types';
import type { ReminderConsent, ReminderEvent } from '../../domain/reminders/types';

export interface RepositoryWriteOptions {
  context: OperationContext;
}

export interface AppointmentRepository {
  findById(id: string, context: OperationContext): Promise<Appointment | null>;
  findByDay(day: string, context: OperationContext): Promise<Appointment[]>;
  findByPatientProfile(patientProfileId: string, context: OperationContext): Promise<Appointment[]>;
  save(appointment: Appointment, options: RepositoryWriteOptions): Promise<Appointment>;
  updateStatus(
    id: string,
    status: Appointment['status'],
    options: RepositoryWriteOptions,
  ): Promise<Appointment>;
}

export interface PatientProfileRepository {
  findById(id: string, context: OperationContext): Promise<PatientSchedulingProfile | null>;
  findByEhrPatientId(ehrPatientId: string, context: OperationContext): Promise<PatientSchedulingProfile | null>;
  save(profile: PatientSchedulingProfile, options: RepositoryWriteOptions): Promise<PatientSchedulingProfile>;
}

export interface IntakeRepository {
  findByAppointment(appointmentId: string, context: OperationContext): Promise<IntakeSubmission | null>;
  save(submission: IntakeSubmission, options: RepositoryWriteOptions): Promise<IntakeSubmission>;
  markExported(id: string, exportedAt: string, options: RepositoryWriteOptions): Promise<IntakeSubmission>;
}

export interface InsuranceRepository {
  findByAppointment(appointmentId: string, context: OperationContext): Promise<InsuranceSubmission | null>;
  save(submission: InsuranceSubmission, options: RepositoryWriteOptions): Promise<InsuranceSubmission>;
  markExported(id: string, exportedAt: string, options: RepositoryWriteOptions): Promise<InsuranceSubmission>;
}

export interface ConsentRepository {
  findByPatientAndChannel(
    patientProfileId: string,
    channel: ReminderConsent['channel'],
    context: OperationContext,
  ): Promise<ReminderConsent | null>;
  save(consent: ReminderConsent, options: RepositoryWriteOptions): Promise<ReminderConsent>;
  recordOptOut(
    patientProfileId: string,
    channel: ReminderConsent['channel'],
    optedOutAt: string,
    options: RepositoryWriteOptions,
  ): Promise<ReminderConsent>;
}

export interface ReminderEventRepository {
  findDueBefore(cutoff: string, context: OperationContext): Promise<ReminderEvent[]>;
  findByAppointment(appointmentId: string, context: OperationContext): Promise<ReminderEvent[]>;
  save(event: ReminderEvent, options: RepositoryWriteOptions): Promise<ReminderEvent>;
  updateDeliveryStatus(
    id: string,
    status: ReminderEvent['status'],
    deliveryFields: Pick<
      ReminderEvent,
      'vendorMessageId' | 'vendorCorrelationId' | 'failureReason' | 'lastAttemptedAt' | 'deliveredAt'
    >,
    options: RepositoryWriteOptions,
  ): Promise<ReminderEvent>;
}

export interface ConfigurationRepository {
  listAppointmentTypes(context: OperationContext): Promise<AppointmentType[]>;
  findAppointmentType(id: string, context: OperationContext): Promise<AppointmentType | null>;
  saveAppointmentType(appointmentType: AppointmentType, options: RepositoryWriteOptions): Promise<AppointmentType>;
  findSchedulingConfiguration(
    appointmentTypeId: string,
    context: OperationContext,
  ): Promise<SchedulingConfiguration | null>;
  saveSchedulingConfiguration(
    configuration: SchedulingConfiguration,
    options: RepositoryWriteOptions,
  ): Promise<SchedulingConfiguration>;
}

export interface StaffUserRepository {
  findById(id: string, context: OperationContext): Promise<StaffUser | null>;
  findByEmail(email: string, context: OperationContext): Promise<StaffUser | null>;
  save(user: StaffUser, options: RepositoryWriteOptions): Promise<StaffUser>;
  setActive(id: string, active: boolean, options: RepositoryWriteOptions): Promise<StaffUser>;
}

export interface IntegrationStateRepository {
  findById(id: string, context: OperationContext): Promise<IntegrationState | null>;
  findPendingRetry(cutoff: string, context: OperationContext): Promise<IntegrationState[]>;
  save(state: IntegrationState, options: RepositoryWriteOptions): Promise<IntegrationState>;
  markSynced(id: string, externalResourceId: string | null, options: RepositoryWriteOptions): Promise<IntegrationState>;
  markFailed(id: string, nextRetryAt: string | null, options: RepositoryWriteOptions): Promise<IntegrationState>;
}

export interface AuditRepository {
  append(event: AuditEvent): Promise<AuditEvent>;
  findByTarget(targetType: AuditEvent['targetType'], targetId: string, context: OperationContext): Promise<AuditEvent[]>;
  findByCorrelationId(correlationId: string, context: OperationContext): Promise<AuditEvent[]>;
}

export interface ClinicRepositories {
  appointments: AppointmentRepository;
  patientProfiles: PatientProfileRepository;
  intake: IntakeRepository;
  insurance: InsuranceRepository;
  consents: ConsentRepository;
  reminderEvents: ReminderEventRepository;
  configuration: ConfigurationRepository;
  staffUsers: StaffUserRepository;
  integrationStates: IntegrationStateRepository;
  auditEvents: AuditRepository;
}
