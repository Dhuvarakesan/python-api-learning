import { DISALLOWED_OBSERVABILITY_FIELDS } from "../../shared/contracts/audit-event-v1";

export const CORRELATION_ID_ATTRIBUTE = "correlation_id" as const;

export const PHI_SAFE_LOGGING_POLICY = {
  requiredFields: ["timestamp", "level", "message", "correlation_id", "source_service", "event_type", "outcome"],
  excludedFields: DISALLOWED_OBSERVABILITY_FIELDS,
  traceAttributes: ["correlation_id", "source_service", "event_type", "appointment_ref", "outcome"],
} as const;

export const OBSERVABILITY_METRICS = {
  bookingLatency: "booking_latency_ms",
  availabilityLookupLatency: "availability_lookup_latency_ms",
  ehrErrorRate: "ehr_error_rate",
  holdCount: "hold_count",
  conflictCount: "conflict_count",
  scheduleViewLatency: "schedule_view_latency_ms",
  messagingDeliveryStatus: "messaging_delivery_status_total",
  auditDeliveryFailures: "audit_delivery_failures_total",
} as const;

export type StructuredLogFields = Record<string, string | number | boolean | null | undefined>;

export function sanitizeStructuredLogFields(fields: StructuredLogFields): StructuredLogFields {
  const sanitized: StructuredLogFields = {};
  const disallowed = new Set<string>(DISALLOWED_OBSERVABILITY_FIELDS);

  for (const [key, value] of Object.entries(fields)) {
    sanitized[key] = disallowed.has(key) ? "[REDACTED]" : value;
  }

  return sanitized;
}

export function createTraceAttributes(fields: StructuredLogFields): StructuredLogFields {
  const sanitized = sanitizeStructuredLogFields(fields);
  const allowed = new Set<string>(PHI_SAFE_LOGGING_POLICY.traceAttributes);
  const attributes: StructuredLogFields = {};

  for (const [key, value] of Object.entries(sanitized)) {
    if (allowed.has(key)) {
      attributes[key] = value;
    }
  }

  return attributes;
}
