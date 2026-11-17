import {
  AUDIT_EVENT_SCHEMA_VERSION,
  DISALLOWED_OBSERVABILITY_FIELDS,
  type AuditEventV1,
  type MinimumNecessaryAuditMetadata,
} from "../../shared/contracts/audit-event-v1";

export type AuditEventInput = Omit<AuditEventV1, "schemaVersion">;

export function createAuditEventV1(input: AuditEventInput): Readonly<AuditEventV1> {
  assertMinimumNecessaryMetadata(input.metadata);

  return Object.freeze({
    schemaVersion: AUDIT_EVENT_SCHEMA_VERSION,
    ...input,
  });
}

export function assertMinimumNecessaryMetadata(metadata: MinimumNecessaryAuditMetadata): void {
  const disallowedKeys = new Set<string>(DISALLOWED_OBSERVABILITY_FIELDS);

  for (const fieldName of Object.keys(metadata.fields)) {
    if (disallowedKeys.has(fieldName)) {
      throw new Error(`Audit metadata field "${fieldName}" is not minimum necessary and must not be recorded.`);
    }
  }
}

export function serializeAuditEvent(event: Readonly<AuditEventV1>): string {
  return JSON.stringify(event);
}
