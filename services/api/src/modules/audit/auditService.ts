import type {
  AuditActorType,
  AuditEvent,
  AuditEventRepository,
} from '../../repositories/index.js';
import {
  createPhiSafeMetadata,
  type PhiSafeMetadata,
} from '../../security/index.js';

export const AUDIT_EVENT_TYPES = {
  patientSubmission: {
    received: 'patient_submission.received',
    verified: 'patient_submission.verified',
    rejected: 'patient_submission.rejected',
  },
  staffDecision: {
    approved: 'staff_decision.approved',
    declined: 'staff_decision.declined',
    needsFollowUp: 'staff_decision.needs_follow_up',
  },
  reminderFailure: {
    deliveryFailed: 'reminder_failure.delivery_failed',
    retryExhausted: 'reminder_failure.retry_exhausted',
  },
  schedulingFailure: {
    slotUnavailable: 'scheduling_failure.slot_unavailable',
    integrationFailed: 'scheduling_failure.integration_failed',
  },
  maintenanceEvent: {
    migrationStarted: 'maintenance_event.migration_started',
    migrationCompleted: 'maintenance_event.migration_completed',
    configurationChanged: 'maintenance_event.configuration_changed',
  },
  securityEvent: {
    authorizationDenied: 'security_event.authorization_denied',
    authenticationFailed: 'security_event.authentication_failed',
    secretResolutionFailed: 'security_event.secret_resolution_failed',
  },
} as const;

type AuditEventTypeGroups = typeof AUDIT_EVENT_TYPES;
export type AuditEventType =
  AuditEventTypeGroups[keyof AuditEventTypeGroups][keyof AuditEventTypeGroups[keyof AuditEventTypeGroups]];

export interface AuditActorContext {
  readonly type: AuditActorType;
  readonly id?: string;
}

export interface AuditEntityReference {
  readonly type: string;
  readonly id?: string;
}

export interface AuditRequestContext {
  readonly correlationId?: string;
  readonly ipAddress?: string;
  readonly userAgent?: string;
}

export interface WriteAuditEventInput {
  readonly occurredAt?: Date;
  readonly actor: AuditActorContext;
  readonly eventType: AuditEventType;
  readonly entity: AuditEntityReference;
  readonly request?: AuditRequestContext;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface AuditService {
  readonly writeEvent: (input: WriteAuditEventInput) => Promise<AuditEvent>;
}

export function createAuditService(
  repository: AuditEventRepository,
): AuditService {
  return {
    writeEvent: async (input) => {
      const metadata: PhiSafeMetadata = createPhiSafeMetadata(input.metadata);

      return repository.record({
        occurredAt: input.occurredAt,
        actorType: input.actor.type,
        actorId: input.actor.id,
        action: input.eventType,
        resourceType: input.entity.type,
        resourceId: input.entity.id,
        correlationId: input.request?.correlationId,
        ipAddress: input.request?.ipAddress,
        userAgent: input.request?.userAgent,
        metadata,
      });
    },
  };
}
