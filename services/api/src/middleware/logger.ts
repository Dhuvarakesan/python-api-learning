import type { IncomingMessage, ServerResponse } from 'node:http';
import type { RequestContext } from './request-context.js';

export interface StructuredLogEntry {
  readonly level: 'info' | 'warn' | 'error';
  readonly message: string;
  readonly correlationId: string;
  readonly method: string;
  readonly path: string;
  readonly statusCode: number;
  readonly durationMs: number;
}

export interface StructuredLogger {
  readonly info: (entry: StructuredLogEntry) => void;
  readonly warn: (entry: StructuredLogEntry) => void;
  readonly error: (entry: StructuredLogEntry) => void;
}

export const consoleStructuredLogger: StructuredLogger = {
  info: (entry) => console.info(JSON.stringify(entry)),
  warn: (entry) => console.warn(JSON.stringify(entry)),
  error: (entry) => console.error(JSON.stringify(entry)),
};

export function createRequestLogger(
  logger: StructuredLogger = consoleStructuredLogger,
) {
  return (
    request: IncomingMessage,
    response: ServerResponse,
    context: RequestContext,
  ): void => {
    response.once('finish', () => {
      const statusCode = response.statusCode;
      const level =
        statusCode >= 500 ? 'error' : statusCode >= 400 ? 'warn' : 'info';
      const entry: StructuredLogEntry = {
        level,
        message: 'http_request_completed',
        correlationId: context.correlationId,
        method: request.method ?? 'UNKNOWN',
        path: getPathWithoutQuery(request.url),
        statusCode,
        durationMs: Date.now() - context.startedAt,
      };

      logger[level](entry);
    });
  };
}

function getPathWithoutQuery(url: string | undefined): string {
  if (!url) {
    return '/';
  }

  try {
    return new URL(url, 'http://api.local').pathname;
  } catch {
    return '/';
  }
}
