import type { ApiErrorResponse, AuditReference, CorrelationId } from "@clinic/contracts";

export interface AvailabilityWindow {
  clinicianId: string;
  startsAt: string;
  endsAt: string;
}

export function isAvailabilityWindowValid(window: AvailabilityWindow): boolean {
  return Date.parse(window.startsAt) < Date.parse(window.endsAt);
}

export type SchedulingDomainErrorCode =
  | "APPOINTMENT_NOT_AVAILABLE"
  | "INVALID_APPOINTMENT_REQUEST"
  | "SCHEDULING_CONFLICT"
  | "SCHEDULING_POLICY_VIOLATION"
  | "SCHEDULING_SERVICE_UNAVAILABLE";

export interface DomainErrorContext {
  correlationId?: CorrelationId;
  audit?: AuditReference;
  retryable?: boolean;
}

export interface PublicSchedulingDomainError {
  code: SchedulingDomainErrorCode;
  message: string;
  retryable: boolean;
  correlationId?: CorrelationId;
  audit?: AuditReference;
}

const defaultErrorMessages: Record<SchedulingDomainErrorCode, string> = {
  APPOINTMENT_NOT_AVAILABLE: "The requested appointment is not available.",
  INVALID_APPOINTMENT_REQUEST: "The appointment request is invalid.",
  SCHEDULING_CONFLICT: "The requested appointment conflicts with current availability.",
  SCHEDULING_POLICY_VIOLATION: "The appointment request does not meet scheduling policy.",
  SCHEDULING_SERVICE_UNAVAILABLE: "Scheduling is temporarily unavailable.",
};

const retryableErrors: ReadonlySet<SchedulingDomainErrorCode> = new Set([
  "SCHEDULING_SERVICE_UNAVAILABLE",
]);

export class SchedulingDomainError extends Error {
  public readonly code: SchedulingDomainErrorCode;
  public readonly correlationId?: CorrelationId;
  public readonly audit?: AuditReference;
  public readonly retryable: boolean;

  public constructor(code: SchedulingDomainErrorCode, context: DomainErrorContext = {}) {
    super(defaultErrorMessages[code]);
    this.name = "SchedulingDomainError";
    this.code = code;
    this.correlationId = context.correlationId;
    this.audit = context.audit;
    this.retryable = context.retryable ?? retryableErrors.has(code);
  }

  public toPublicError(): PublicSchedulingDomainError {
    return {
      code: this.code,
      message: this.message,
      retryable: this.retryable,
      ...(this.correlationId ? { correlationId: this.correlationId } : {}),
      ...(this.audit ? { audit: this.audit } : {}),
    };
  }

  public toApiErrorResponse(correlationId: CorrelationId): ApiErrorResponse {
    return {
      ok: false,
      error: {
        code: this.code,
        message: this.message,
        correlationId: this.correlationId ?? correlationId,
      },
    };
  }
}

export function createSchedulingDomainError(
  code: SchedulingDomainErrorCode,
  context?: DomainErrorContext,
): SchedulingDomainError {
  return new SchedulingDomainError(code, context);
}
