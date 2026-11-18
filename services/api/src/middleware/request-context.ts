import { randomUUID } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';

export const CORRELATION_ID_HEADER = 'x-correlation-id';

export interface RequestContext {
  readonly correlationId: string;
  readonly startedAt: number;
}

export function createRequestContext(
  request: IncomingMessage,
  response: ServerResponse,
): RequestContext {
  const incomingCorrelationId = request.headers[CORRELATION_ID_HEADER];
  const correlationId =
    typeof incomingCorrelationId === 'string' && incomingCorrelationId.trim()
      ? incomingCorrelationId.trim()
      : randomUUID();

  response.setHeader(CORRELATION_ID_HEADER, correlationId);

  return {
    correlationId,
    startedAt: Date.now(),
  };
}
