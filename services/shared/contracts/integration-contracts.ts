export type EnvironmentName = "local" | "test" | "staging" | "production";

export type AuditTopic =
  | "audit.booking"
  | "audit.reschedule"
  | "audit.reminder"
  | "audit.configuration"
  | "audit.schedule_access"
  | "ops.integration_failure";

export type PhiClassification = "none" | "phi-adjacent" | "phi";

export type RetentionPolicy = "operational-90-days" | "audit-6-years";

export interface RetentionMetadata {
  policy: RetentionPolicy;
  expiresAt?: string;
  legalHold: boolean;
  reason: string;
}

export interface AuditMetadata {
  topic: AuditTopic;
  action: string;
  actorType: "patient" | "staff" | "provider" | "service" | "vendor";
  actorId?: string;
  occurredAt: string;
  sourceService: string;
  phiClassification: PhiClassification;
  retention: RetentionMetadata;
}

export interface IntegrationRequestContext {
  correlationId: string;
  idempotencyKey?: string;
  environment: EnvironmentName;
  requestedAt: string;
  sourceService: string;
  audit: AuditMetadata;
}

export interface RetryMetadata {
  retryable: boolean;
  retryAfterSeconds?: number;
  maxAttempts?: number;
  attempt?: number;
}

export interface UserSafeError {
  userSafeErrorCode: string;
  userSafeMessage: string;
  retry: RetryMetadata;
  correlationId: string;
  audit: AuditMetadata;
}

export interface IntegrationEvent<TPayload extends Record<string, unknown> = Record<string, unknown>> {
  eventId: string;
  eventType: AuditTopic;
  correlationId: string;
  idempotencyKey?: string;
  emittedAt: string;
  sourceService: string;
  retry: RetryMetadata;
  userSafeErrorCode?: string;
  audit: AuditMetadata;
  payload: TPayload;
}

export interface MutatingIntegrationCommand<TPayload extends Record<string, unknown> = Record<string, unknown>> {
  operation: "book" | "reschedule" | "release-hold" | "send-otp" | "send-reminder" | "update-configuration";
  context: IntegrationRequestContext & { idempotencyKey: string };
  retry: RetryMetadata;
  payload: TPayload;
}

export const CORRELATION_ID_HEADER = "x-correlation-id";
export const IDEMPOTENCY_KEY_HEADER = "idempotency-key";

export function requiresIdempotencyKey(method: string): boolean {
  return ["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase());
}

export function isAuditRetention(policy: RetentionPolicy): boolean {
  return policy === "audit-6-years";
}

export type {
  ActorContext,
  AppointmentReference,
  AuditActorType,
  AuditEventSchemaVersion,
  AuditEventType,
  AuditEventV1,
  AuditOutcome,
  AuditSourceSystem,
  AuditSubjectContext,
  MinimumNecessaryAuditMetadata,
  VerifiedPatientContext,
} from "./audit-event-v1";

export {
  AUDIT_EVENT_SCHEMA_VERSION,
  AUDIT_EVENT_TYPES,
  DISALLOWED_OBSERVABILITY_FIELDS,
} from "./audit-event-v1";
