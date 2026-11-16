export type CorrelationId = string & { readonly __brand: "CorrelationId" };

export type ApiActorType = "patient" | "staff" | "system";

export interface AuditReference {
  auditId: string;
  actorType: ApiActorType;
  occurredAt: string;
  correlationId: CorrelationId;
}

export interface ValidationIssue {
  path: string;
  code: string;
  message: string;
}

export interface ApiResponseMeta {
  correlationId: CorrelationId;
}

export interface StaffResponseMeta extends ApiResponseMeta {
  audit?: AuditReference;
}

export interface ApiSuccessResponse<TData, TMeta extends ApiResponseMeta = ApiResponseMeta> {
  ok: true;
  data: TData;
  meta: TMeta;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  correlationId: CorrelationId;
}

export interface ValidationErrorBody extends ApiErrorBody {
  code: "VALIDATION_ERROR";
  issues: ValidationIssue[];
}

export interface ApiErrorResponse<TError extends ApiErrorBody = ApiErrorBody> {
  ok: false;
  error: TError;
}

export type PatientSafeApiResponse<TData> =
  | ApiSuccessResponse<TData>
  | ApiErrorResponse<ApiErrorBody | ValidationErrorBody>;

export type StaffSafeApiResponse<TData> =
  | ApiSuccessResponse<TData, StaffResponseMeta>
  | ApiErrorResponse<ApiErrorBody | ValidationErrorBody>;

export type AppointmentChannel = "in-person" | "telehealth";

export interface AppointmentSlot {
  id: string;
  clinicianId: string;
  startsAt: string;
  endsAt: string;
  channel: AppointmentChannel;
}

export function asCorrelationId(value: string): CorrelationId {
  return value as CorrelationId;
}

export function createSuccessResponse<TData>(
  data: TData,
  correlationId: CorrelationId,
): PatientSafeApiResponse<TData> {
  return {
    ok: true,
    data,
    meta: {
      correlationId,
    },
  };
}

export function createStaffSuccessResponse<TData>(
  data: TData,
  meta: StaffResponseMeta,
): StaffSafeApiResponse<TData> {
  return {
    ok: true,
    data,
    meta,
  };
}

export function createApiErrorResponse(
  code: string,
  message: string,
  correlationId: CorrelationId,
): ApiErrorResponse {
  return {
    ok: false,
    error: {
      code,
      message,
      correlationId,
    },
  };
}

export function createValidationErrorResponse(
  issues: ValidationIssue[],
  correlationId: CorrelationId,
): ApiErrorResponse<ValidationErrorBody> {
  return {
    ok: false,
    error: {
      code: "VALIDATION_ERROR",
      message: "Request validation failed.",
      correlationId,
      issues,
    },
  };
}
