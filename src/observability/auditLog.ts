import type { AuditEvent, OperationContext } from '../domain/scheduling/types';
import { redactPhi } from './logger';

export interface AuditTarget {
  type: AuditEvent['targetType'];
  id: string | null;
}

export interface AuditRecorder {
  record(event: AuditEvent): Promise<void>;
}

export type AuditRecordInput = {
  actor: Pick<OperationContext, 'actorId' | 'actorType'> | null;
  action: AuditEvent['action'];
  target: AuditTarget;
  outcome: AuditEvent['outcome'];
  correlationId: string;
  safeMetadata?: Record<string, unknown>;
  occurredAt?: Date;
  id?: string;
};

export function buildAuditEvent(input: AuditRecordInput): AuditEvent {
  return {
    id: input.id ?? createAuditId(),
    occurredAt: (input.occurredAt ?? new Date()).toISOString(),
    actorId: input.actor?.actorId ?? null,
    actorType: input.actor?.actorType ?? 'system',
    action: input.action,
    targetType: input.target.type,
    targetId: input.target.id,
    outcome: input.outcome,
    correlationId: input.correlationId,
    safeMetadata: redactPhi(input.safeMetadata ?? {}),
  };
}

export async function recordAuditEvent(
  input: AuditRecordInput,
  recorder: AuditRecorder,
): Promise<AuditEvent> {
  const event = buildAuditEvent(input);
  await recorder.record(event);
  return event;
}

function createAuditId(): string {
  const randomUuid = globalThis.crypto?.randomUUID;

  if (randomUuid) {
    return `audit_${randomUuid.call(globalThis.crypto)}`;
  }

  return `audit_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
}
