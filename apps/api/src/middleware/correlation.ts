import { randomUUID } from "node:crypto";
import type { IncomingMessage } from "node:http";
import { asCorrelationId, type CorrelationId } from "@clinic/contracts";

const correlationHeader = "x-correlation-id";

export function getCorrelationId(request: IncomingMessage): CorrelationId {
  const headerValue = request.headers[correlationHeader];
  const value = Array.isArray(headerValue) ? headerValue[0] : headerValue;

  if (value && value.trim().length > 0) {
    return asCorrelationId(value);
  }

  return asCorrelationId(randomUUID());
}
