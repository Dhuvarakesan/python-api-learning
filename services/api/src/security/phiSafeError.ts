export interface PatientSafeErrorOptions {
  readonly statusCode?: number;
  readonly code: string;
  readonly patientSafeMessage: string;
  readonly internalDiagnostics?: Record<string, unknown>;
  readonly cause?: unknown;
}

export interface PatientSafeErrorResponse {
  readonly error: string;
  readonly message: string;
}

export class PatientSafeError extends Error {
  public readonly statusCode: number;
  public readonly code: string;
  public readonly patientSafeMessage: string;
  public readonly internalDiagnostics: Readonly<Record<string, unknown>>;

  public constructor(options: PatientSafeErrorOptions) {
    super(options.patientSafeMessage, { cause: options.cause });
    this.name = 'PatientSafeError';
    this.statusCode = options.statusCode ?? 500;
    this.code = options.code;
    this.patientSafeMessage = options.patientSafeMessage;
    this.internalDiagnostics = options.internalDiagnostics ?? {};
  }
}

export function createPatientSafeError(
  options: PatientSafeErrorOptions,
): PatientSafeError {
  return new PatientSafeError(options);
}

export function normalizePatientSafeError(error: unknown): PatientSafeError {
  if (error instanceof PatientSafeError) {
    return error;
  }

  return new PatientSafeError({
    statusCode: 500,
    code: 'internal_server_error',
    patientSafeMessage: 'We could not complete the request.',
    internalDiagnostics: summarizeUnknownError(error),
    cause: error,
  });
}

export function toPatientSafeErrorResponse(
  error: unknown,
): {
  readonly statusCode: number;
  readonly body: PatientSafeErrorResponse;
} {
  const normalized = normalizePatientSafeError(error);

  return {
    statusCode: normalized.statusCode,
    body: {
      error: normalized.code,
      message: normalized.patientSafeMessage,
    },
  };
}

export function summarizeUnknownError(
  error: unknown,
): Record<string, string> {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
    };
  }

  return {
    name: typeof error,
    message: 'Non-error value thrown.',
  };
}
