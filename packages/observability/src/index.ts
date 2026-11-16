import { randomUUID } from "node:crypto";

export type LogLevel = "info" | "warn" | "error";

export type SafeAttributeValue = string | number | boolean | null;

export type SafeAttributes = Record<string, SafeAttributeValue>;

export interface ObservabilityEvent {
  level: LogLevel;
  message: string;
  attributes?: SafeAttributes;
}

export interface StructuredLogEvent extends ObservabilityEvent {
  service: string;
  timestamp: string;
}

export interface LoggerSink {
  info(event: StructuredLogEvent): void;
  warn(event: StructuredLogEvent): void;
  error(event: StructuredLogEvent): void;
}

export interface Logger {
  info(message: string, attributes?: SafeAttributes): void;
  warn(message: string, attributes?: SafeAttributes): void;
  error(message: string, attributes?: SafeAttributes): void;
}

export interface LoggerOptions {
  service: string;
  sink?: LoggerSink;
  now?: () => Date;
}

export type AuditActorType =
  "patient" | "staff_user" | "system" | "external_provider";

export type AuditOutcome = "success" | "failure" | "denied";

export type AuditAction =
  | "staff.access"
  | "role.change"
  | "booking_rule.change"
  | "booking.submission"
  | "reschedule.submission"
  | "verification.attempt"
  | "reminder.delivery_attempt"
  | "integration.write"
  | "security.event";

export interface AuditActor {
  type: AuditActorType;
  staffUserId?: string;
}

export interface AuditEntityReferences {
  patientSchedulingProfileId?: string;
  appointmentId?: string;
  bookingRequestId?: string;
  rescheduleRequestId?: string;
  reminderDeliveryAttemptId?: string;
  verificationAttemptId?: string;
  entityType?: string;
  entityId?: string;
}

export interface AuditEventInput {
  action: AuditAction;
  actor: AuditActor;
  correlationId: string;
  occurredAt?: Date;
  outcome?: AuditOutcome;
  references?: AuditEntityReferences;
  safeMetadata?: SafeAttributes;
}

export interface NormalizedAuditEvent {
  action: AuditAction;
  actor: AuditActor;
  correlationId: string;
  occurredAt: Date;
  outcome: AuditOutcome;
  references: AuditEntityReferences;
  safeMetadata: SafeAttributes;
}

const sensitiveAttributeNamePattern =
  /authorization|cookie|dob|birth|email|name|password|patient|phone|phi|secret|ssn|token|verification.?code|otp/i;

const bearerTokenPattern = /\bBearer\s+[A-Za-z0-9._~+/=-]+/gi;
const longCodePattern = /\b\d{6,10}\b/g;

const consoleJsonSink: LoggerSink = {
  info: (event) => console.info(JSON.stringify(event)),
  warn: (event) => console.warn(JSON.stringify(event)),
  error: (event) => console.error(JSON.stringify(event)),
};

export function createCorrelationId(value?: string): string {
  const trimmed = value?.trim();

  if (trimmed) {
    return trimmed;
  }

  return randomUUID();
}

export function redactSensitiveText(value: string): string {
  return value
    .replace(bearerTokenPattern, "Bearer [REDACTED]")
    .replace(longCodePattern, "[REDACTED_CODE]");
}

export function isSensitiveAttributeName(key: string): boolean {
  return sensitiveAttributeNamePattern.test(key);
}

export function redactSensitiveAttributes(
  attributes: SafeAttributes = {},
): SafeAttributes {
  return Object.fromEntries(
    Object.entries(attributes)
      .filter(([key]) => !isSensitiveAttributeName(key))
      .map(([key, value]) => [
        key,
        typeof value === "string" ? redactSensitiveText(value) : value,
      ]),
  );
}

export function redactSensitiveAttributeNames(
  event: ObservabilityEvent,
): ObservabilityEvent {
  return {
    ...event,
    message: redactSensitiveText(event.message),
    attributes: redactSensitiveAttributes(event.attributes),
  };
}

export function createLogger(options: LoggerOptions): Logger {
  const sink = options.sink ?? consoleJsonSink;
  const now = options.now ?? (() => new Date());

  function write(
    level: LogLevel,
    message: string,
    attributes?: SafeAttributes,
  ): void {
    const event: StructuredLogEvent = {
      service: options.service,
      level,
      message: redactSensitiveText(message),
      timestamp: now().toISOString(),
      attributes: redactSensitiveAttributes(attributes),
    };

    sink[level](event);
  }

  return {
    info: (message, attributes) => write("info", message, attributes),
    warn: (message, attributes) => write("warn", message, attributes),
    error: (message, attributes) => write("error", message, attributes),
  };
}

export function normalizeAuditEvent(
  input: AuditEventInput,
): NormalizedAuditEvent {
  return {
    action: input.action,
    actor: input.actor,
    correlationId: createCorrelationId(input.correlationId),
    occurredAt: input.occurredAt ?? new Date(),
    outcome: input.outcome ?? "success",
    references: input.references ?? {},
    safeMetadata: redactSensitiveAttributes(input.safeMetadata),
  };
}
