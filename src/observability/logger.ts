export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface StructuredLogEntry {
  level: LogLevel;
  message: string;
  timestamp: string;
  correlationId?: string;
  context?: Record<string, unknown>;
}

export interface LoggerSink {
  write(entry: StructuredLogEntry): void;
}

export interface Logger {
  debug(message: string, context?: Record<string, unknown>): void;
  info(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  error(message: string, context?: Record<string, unknown>): void;
}

const redactedValue = '[REDACTED]';
const maxDepth = 8;

const sensitiveKeyPatterns = [
  /(^|_)phi($|_)/i,
  /patient.*name/i,
  /legal.*name/i,
  /date.*of.*birth/i,
  /^dob$/i,
  /birth.*date/i,
  /phone/i,
  /email/i,
  /address/i,
  /member.*id/i,
  /group.*number/i,
  /policyholder/i,
  /insurance.*payload/i,
  /intake.*payload/i,
  /cancellation.*reason/i,
  /ssn/i,
  /diagnosis/i,
  /symptom/i,
  /medical.*record/i,
  /^mrn$/i,
  /token/i,
  /secret/i,
  /password/i,
];

export function redactPhi<T>(value: T): T {
  return redactValue(value, 0, new WeakSet<object>()) as T;
}

export function createLogger(options: {
  sink?: LoggerSink;
  correlationId?: string;
  now?: () => Date;
} = {}): Logger {
  const sink = options.sink ?? consoleSink;
  const now = options.now ?? (() => new Date());

  const log = (level: LogLevel, message: string, context?: Record<string, unknown>): void => {
    sink.write({
      level,
      message,
      timestamp: now().toISOString(),
      correlationId: options.correlationId,
      context: context ? redactPhi(context) : undefined,
    });
  };

  return {
    debug: (message, context) => log('debug', message, context),
    info: (message, context) => log('info', message, context),
    warn: (message, context) => log('warn', message, context),
    error: (message, context) => log('error', message, context),
  };
}

export function isSensitiveLogKey(key: string): boolean {
  return sensitiveKeyPatterns.some((pattern) => pattern.test(key));
}

function redactValue(value: unknown, depth: number, seen: WeakSet<object>): unknown {
  if (value === null || typeof value !== 'object') {
    return value;
  }

  if (depth >= maxDepth) {
    return '[MAX_DEPTH]';
  }

  if (seen.has(value)) {
    return '[CIRCULAR]';
  }

  seen.add(value);

  if (Array.isArray(value)) {
    const redactedArray = value.map((item) => redactValue(item, depth + 1, seen));
    seen.delete(value);
    return redactedArray;
  }

  const redactedRecord: Record<string, unknown> = {};

  for (const [key, nestedValue] of Object.entries(value)) {
    redactedRecord[key] = isSensitiveLogKey(key)
      ? redactedValue
      : redactValue(nestedValue, depth + 1, seen);
  }

  seen.delete(value);
  return redactedRecord;
}

const consoleSink: LoggerSink = {
  write(entry) {
    const payload = JSON.stringify(entry);

    if (entry.level === 'error') {
      console.error(payload);
      return;
    }

    if (entry.level === 'warn') {
      console.warn(payload);
      return;
    }

    console.log(payload);
  },
};
