export type PhiSafeMetadataValue =
  | string
  | number
  | boolean
  | null
  | readonly PhiSafeMetadataValue[]
  | { readonly [key: string]: PhiSafeMetadataValue };

export type PhiSafeMetadata = Readonly<Record<string, PhiSafeMetadataValue>>;

const SENSITIVE_KEY_PATTERN =
  /(address|birth|clinical|diagnosis|dob|email|insurance|medical|name|note|patient|phone|reason|ssn|symptom)/i;

export function createPhiSafeMetadata(
  metadata: Readonly<Record<string, unknown>> = {},
): PhiSafeMetadata {
  const sanitized: Record<string, PhiSafeMetadataValue> = {};

  for (const [key, value] of Object.entries(metadata)) {
    if (SENSITIVE_KEY_PATTERN.test(key)) {
      sanitized[key] = '[redacted]';
      continue;
    }

    const sanitizedValue = sanitizeMetadataValue(value);
    if (sanitizedValue !== undefined) {
      sanitized[key] = sanitizedValue;
    }
  }

  return sanitized;
}

function sanitizeMetadataValue(value: unknown): PhiSafeMetadataValue | undefined {
  if (value === undefined || typeof value === 'function' || typeof value === 'symbol') {
    return undefined;
  }

  if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }

  if (value instanceof Date) {
    return value.toISOString();
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => sanitizeMetadataValue(item))
      .filter((item): item is PhiSafeMetadataValue => item !== undefined);
  }

  if (typeof value === 'object') {
    return createPhiSafeMetadata(value as Readonly<Record<string, unknown>>);
  }

  return String(value);
}
