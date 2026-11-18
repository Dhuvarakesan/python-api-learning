import { describe, expect, it, vi } from 'vitest';
import { buildAuditEvent, recordAuditEvent, type AuditRecorder } from './auditLog';

describe('auditLog', () => {
  it('records required security and scheduling metadata', async () => {
    const recorder: AuditRecorder = {
      record: vi.fn(async () => undefined),
    };
    const occurredAt = new Date('2026-11-18T10:00:00.000Z');

    const event = await recordAuditEvent(
      {
        id: 'audit-1',
        actor: { actorId: 'staff-1', actorType: 'staff' },
        action: 'configuration_changed',
        target: { type: 'configuration', id: 'config-1' },
        outcome: 'success',
        correlationId: 'corr-1',
        safeMetadata: { field: 'bookingWindowDays' },
        occurredAt,
      },
      recorder,
    );

    expect(event).toMatchObject({
      id: 'audit-1',
      occurredAt: '2026-11-18T10:00:00.000Z',
      actorId: 'staff-1',
      actorType: 'staff',
      action: 'configuration_changed',
      targetType: 'configuration',
      targetId: 'config-1',
      outcome: 'success',
      correlationId: 'corr-1',
      safeMetadata: { field: 'bookingWindowDays' },
    });
    expect(recorder.record).toHaveBeenCalledWith(event);
  });

  it('redacts PHI-like metadata before audit persistence', () => {
    const event = buildAuditEvent({
      id: 'audit-2',
      actor: { actorId: 'patient-1', actorType: 'patient' },
      action: 'booking_created',
      target: { type: 'appointment', id: 'appt-1' },
      outcome: 'success',
      correlationId: 'corr-2',
      safeMetadata: {
        appointmentTypeId: 'follow-up',
        legalName: 'Jane Patient',
        nested: { email: 'jane@example.com' },
      },
      occurredAt: new Date('2026-11-18T11:00:00.000Z'),
    });

    expect(event.safeMetadata).toEqual({
      appointmentTypeId: 'follow-up',
      legalName: '[REDACTED]',
      nested: { email: '[REDACTED]' },
    });
  });
});
