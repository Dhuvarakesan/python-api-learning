import { describe, expect, it } from 'vitest';
import {
  createLogger,
  redactPhi,
  type LoggerSink,
  type StructuredLogEntry,
} from './logger';

describe('logger PHI redaction', () => {
  it('redacts PHI-like payload fields from structured logs', () => {
    const entries: StructuredLogEntry[] = [];
    const sink: LoggerSink = {
      write: (entry) => entries.push(entry),
    };
    const logger = createLogger({
      sink,
      correlationId: 'corr-logger',
      now: () => new Date('2026-11-18T12:00:00.000Z'),
    });

    logger.info('appointment requested', {
      appointmentId: 'appt-1',
      patient: {
        legalName: 'Jane Patient',
        dateOfBirth: '1980-01-01',
        email: 'jane@example.com',
        phoneNumber: '+15551234567',
      },
      intakePayload: { symptom: 'cough' },
    });

    expect(entries).toHaveLength(1);
    expect(entries[0]).toEqual({
      level: 'info',
      message: 'appointment requested',
      timestamp: '2026-11-18T12:00:00.000Z',
      correlationId: 'corr-logger',
      context: {
        appointmentId: 'appt-1',
        patient: {
          legalName: '[REDACTED]',
          dateOfBirth: '[REDACTED]',
          email: '[REDACTED]',
          phoneNumber: '[REDACTED]',
        },
        intakePayload: '[REDACTED]',
      },
    });
  });

  it('redacts nested PHI-like fields while preserving safe operational fields', () => {
    expect(
      redactPhi({
        correlationId: 'corr-1',
        retryCount: 2,
        insurance: {
          memberId: 'ABC123',
          payerCode: 'safe-code',
        },
        cancellationReason: 'medical issue',
      }),
    ).toEqual({
      correlationId: 'corr-1',
      retryCount: 2,
      insurance: {
        memberId: '[REDACTED]',
        payerCode: 'safe-code',
      },
      cancellationReason: '[REDACTED]',
    });
  });
});
