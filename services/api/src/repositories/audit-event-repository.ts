import { randomUUID } from 'node:crypto';
import type { DatabaseClient } from '../db/index.js';

export type AuditActorType = 'patient' | 'staff' | 'system' | 'integration';

export interface AuditEvent {
  readonly id: string;
  readonly occurredAt: Date;
  readonly actorType: AuditActorType;
  readonly actorId?: string;
  readonly action: string;
  readonly resourceType: string;
  readonly resourceId?: string;
  readonly correlationId?: string;
  readonly ipAddress?: string;
  readonly userAgent?: string;
  readonly metadata: Readonly<Record<string, unknown>>;
}

export type CreateAuditEventInput = Omit<AuditEvent, 'id' | 'occurredAt'> &
  Partial<Pick<AuditEvent, 'id' | 'occurredAt'>>;

export interface AuditEventRepository {
  readonly record: (event: CreateAuditEventInput) => Promise<AuditEvent>;
}

interface AuditEventRow extends Record<string, unknown> {
  readonly id: string;
  readonly occurred_at: Date;
  readonly actor_type: AuditActorType;
  readonly actor_id: string | null;
  readonly action: string;
  readonly resource_type: string;
  readonly resource_id: string | null;
  readonly correlation_id: string | null;
  readonly ip_address: string | null;
  readonly user_agent: string | null;
  readonly metadata: Record<string, unknown>;
}

export class SqlAuditEventRepository implements AuditEventRepository {
  public constructor(private readonly client: DatabaseClient) {}

  public async record(event: CreateAuditEventInput): Promise<AuditEvent> {
    const id = event.id ?? randomUUID();
    const occurredAt = event.occurredAt ?? new Date();
    const result = await this.client.query<AuditEventRow>(
      `INSERT INTO audit_events (
        id,
        occurred_at,
        actor_type,
        actor_id,
        action,
        resource_type,
        resource_id,
        correlation_id,
        ip_address,
        user_agent,
        metadata
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      RETURNING
        id,
        occurred_at,
        actor_type,
        actor_id,
        action,
        resource_type,
        resource_id,
        correlation_id,
        ip_address::text,
        user_agent,
        metadata`,
      [
        id,
        occurredAt,
        event.actorType,
        event.actorId ?? null,
        event.action,
        event.resourceType,
        event.resourceId ?? null,
        event.correlationId ?? null,
        event.ipAddress ?? null,
        event.userAgent ?? null,
        event.metadata,
      ],
    );

    const row = result.rows[0];
    if (!row) {
      throw new Error('Audit event insert did not return a row.');
    }

    return mapAuditEventRow(row);
  }
}

function mapAuditEventRow(row: AuditEventRow): AuditEvent {
  return {
    id: row.id,
    occurredAt: row.occurred_at,
    actorType: row.actor_type,
    actorId: row.actor_id ?? undefined,
    action: row.action,
    resourceType: row.resource_type,
    resourceId: row.resource_id ?? undefined,
    correlationId: row.correlation_id ?? undefined,
    ipAddress: row.ip_address ?? undefined,
    userAgent: row.user_agent ?? undefined,
    metadata: row.metadata,
  };
}
