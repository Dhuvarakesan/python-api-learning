import type {
  AuditEventInput,
  NormalizedAuditEvent,
} from "@clinic/observability";
import { normalizeAuditEvent } from "@clinic/observability";

export interface AuditQueryResult {
  rows: Array<{
    id: string;
    occurred_at: Date | string;
  }>;
}

export interface AuditSqlExecutor {
  query(sql: string, values: unknown[]): Promise<AuditQueryResult>;
}

export interface PersistedAuditEvent {
  auditId: string;
  action: NormalizedAuditEvent["action"];
  actorType: NormalizedAuditEvent["actor"]["type"];
  occurredAt: string;
  correlationId: string;
}

export async function writeAuditEvent(
  executor: AuditSqlExecutor,
  input: AuditEventInput,
): Promise<PersistedAuditEvent> {
  const event = normalizeAuditEvent(input);
  const references = event.references;

  const result = await executor.query(
    `INSERT INTO audit_events (
      event_type,
      actor_type,
      actor_staff_user_id,
      patient_scheduling_profile_id,
      appointment_id,
      booking_request_id,
      reschedule_request_id,
      reminder_delivery_attempt_id,
      verification_attempt_id,
      entity_type,
      entity_id,
      outcome,
      correlation_id,
      safe_metadata,
      occurred_at
    )
    VALUES (
      $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
      $11, $12, $13, $14::jsonb, $15
    )
    RETURNING id, occurred_at`,
    [
      event.action,
      event.actor.type,
      event.actor.staffUserId ?? null,
      references.patientSchedulingProfileId ?? null,
      references.appointmentId ?? null,
      references.bookingRequestId ?? null,
      references.rescheduleRequestId ?? null,
      references.reminderDeliveryAttemptId ?? null,
      references.verificationAttemptId ?? null,
      references.entityType ?? null,
      references.entityId ?? null,
      event.outcome,
      event.correlationId,
      JSON.stringify(event.safeMetadata),
      event.occurredAt,
    ],
  );

  const row = result.rows[0];

  if (!row) {
    throw new Error("Audit event insert did not return an id.");
  }

  return {
    auditId: row.id,
    action: event.action,
    actorType: event.actor.type,
    occurredAt:
      row.occurred_at instanceof Date
        ? row.occurred_at.toISOString()
        : row.occurred_at,
    correlationId: event.correlationId,
  };
}
