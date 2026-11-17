import type { PhiClassification, RetentionMetadata } from "../contracts/integration-contracts";

export const OPERATIONAL_PHI_RETENTION_DAYS = 90;
export const AUDIT_RETENTION_YEARS = 6;

export interface OperationalRecordMetadata {
  phiClassification: PhiClassification;
  retention: RetentionMetadata;
  minimumNecessaryJustification: string;
  sourceSystemOfRecord: "ehr-practice-management";
}

export function createOperationalPhiMetadata(expiresAt: string, reason: string): OperationalRecordMetadata {
  return {
    phiClassification: "phi",
    retention: {
      policy: "operational-90-days",
      expiresAt,
      legalHold: false,
      reason,
    },
    minimumNecessaryJustification: reason,
    sourceSystemOfRecord: "ehr-practice-management",
  };
}

export function createAuditMetadata(reason: string): OperationalRecordMetadata {
  return {
    phiClassification: "phi-adjacent",
    retention: {
      policy: "audit-6-years",
      legalHold: false,
      reason,
    },
    minimumNecessaryJustification: reason,
    sourceSystemOfRecord: "ehr-practice-management",
  };
}

export function redactPhiFields<TRecord extends Record<string, unknown>>(record: TRecord): TRecord {
  const redacted = { ...record };
  const fieldNames = ["patientName", "phone", "email", "dateOfBirth", "messageBody", "otpCode"];

  for (const fieldName of fieldNames) {
    if (fieldName in redacted) {
      redacted[fieldName as keyof TRecord] = "[REDACTED]" as TRecord[keyof TRecord];
    }
  }

  return redacted;
}
