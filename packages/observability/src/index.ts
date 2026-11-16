export type LogLevel = "info" | "warn" | "error";

export interface ObservabilityEvent {
  level: LogLevel;
  message: string;
  attributes?: Record<string, string | number | boolean>;
}

export function redactSensitiveAttributeNames(
  event: ObservabilityEvent,
): ObservabilityEvent {
  const attributes = Object.fromEntries(
    Object.entries(event.attributes ?? {}).filter(
      ([key]) => !/patient|phi|token|secret/i.test(key),
    ),
  );

  return {
    ...event,
    attributes,
  };
}
