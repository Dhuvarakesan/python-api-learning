import type { AuditTopic, PhiClassification, RetentionMetadata } from "./integration-contracts";

export const AUDIT_EVENT_SCHEMA_VERSION = "audit-event.v1" as const;

export type AuditEventSchemaVersion = typeof AUDIT_EVENT_SCHEMA_VERSION;

export type AuditEventType = AuditTopic;

export type AuditActorType = "patient" | "staff" | "provider" | "service" | "vendor";

export type AuditOutcome = "success" | "failure" | "denied" | "timeout" | "partial";

export type AuditSourceSystem =
  | "scheduling-saas"
  | "integration-edge"
  | "scheduling-orchestration"
  | "messaging-orchestration"
  | "schedule-config-adapter"
  | "ehr-adapter"
  | "ehr-practice-management"
  | "audit-compliance"
  | "ci-cd";

export interface VerifiedPatientContext {
  contextType: "verified-patient";
  verifiedPatientRef: string;
  verificationMethod: "otp" | "magic-link" | "portal-session" | "staff-verified";
}

export interface ActorContext {
  contextType: "actor";
  actorType: AuditActorType;
  actorRef: string;
}

export type AuditSubjectContext = ActorContext | VerifiedPatientContext;

export interface AppointmentReference {
  systemOfRecord: "ehr-practice-management";
  appointmentRef?: string;
  holdRef?: string;
  externalVisitTypeRef?: string;
}

export interface MinimumNecessaryAuditMetadata {
  metadataSchema: "minimum-necessary.v1";
  reason: string;
  phiClassification: PhiClassification;
  retention: RetentionMetadata;
  fields: Record<string, string | number | boolean | null>;
}

export interface AuditEventV1 {
  schemaVersion: AuditEventSchemaVersion;
  eventId: string;
  eventType: AuditEventType;
  timestamp: string;
  subject: AuditSubjectContext;
  appointment: AppointmentReference;
  correlationId: string;
  sourceSystem: AuditSourceSystem;
  outcome: AuditOutcome;
  metadata: MinimumNecessaryAuditMetadata;
}

export const AUDIT_EVENT_TYPES: readonly AuditEventType[] = [
  "audit.booking",
  "audit.reschedule",
  "audit.reminder",
  "audit.configuration",
  "audit.schedule_access",
  "ops.integration_failure",
] as const;

export const DISALLOWED_OBSERVABILITY_FIELDS = [
  "otpCode",
  "oneTimePassword",
  "messageBody",
  "smsBody",
  "emailBody",
  "secret",
  "clientSecret",
  "accessToken",
  "refreshToken",
  "authorization",
  "patientName",
  "dateOfBirth",
  "fullAddress",
  "phone",
  "email",
] as const;
