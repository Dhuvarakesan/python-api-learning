import { describe, expect, it } from "vitest";
import {
  createLogger,
  normalizeAuditEvent,
  redactSensitiveAttributeNames,
} from "./index.js";
import type { StructuredLogEvent } from "./index.js";

describe("observability redaction", () => {
  it("removes sensitive attribute names and redacts verification-code shaped text", () => {
    const event = redactSensitiveAttributeNames({
      level: "info",
      message: "verification 123456 was submitted",
      attributes: {
        correlationId: "corr-123",
        verificationCode: "123456",
        authorization: "Bearer abc.def",
        outcome: "success",
        detail: "attempt contained 987654",
      },
    });

    expect(event.message).toBe("verification [REDACTED_CODE] was submitted");
    expect(event.attributes).toEqual({
      correlationId: "corr-123",
      outcome: "success",
      detail: "attempt contained [REDACTED_CODE]",
    });
  });

  it("emits structured safe operational log fields", () => {
    const events: StructuredLogEvent[] = [];
    const logger = createLogger({
      service: "clinic-api",
      now: () => new Date("2026-01-01T00:00:00.000Z"),
      sink: {
        info: (event) => events.push(event),
        warn: (event) => events.push(event),
        error: (event) => events.push(event),
      },
    });

    logger.info("sent code 123456", {
      correlationId: "corr-123",
      path: "/api/v1/health",
      secretToken: "abc",
    });

    expect(events).toEqual([
      {
        service: "clinic-api",
        level: "info",
        message: "sent code [REDACTED_CODE]",
        timestamp: "2026-01-01T00:00:00.000Z",
        attributes: {
          correlationId: "corr-123",
          path: "/api/v1/health",
        },
      },
    ]);
  });
});

describe("audit event normalization", () => {
  it("preserves required audit fields and sanitizes metadata", () => {
    const event = normalizeAuditEvent({
      action: "verification.attempt",
      actor: { type: "patient" },
      correlationId: " corr-456 ",
      occurredAt: new Date("2026-01-02T00:00:00.000Z"),
      safeMetadata: {
        channel: "sms",
        verificationCode: "123456",
        failureReason: "submitted 654321",
      },
    });

    expect(event).toMatchObject({
      action: "verification.attempt",
      actor: { type: "patient" },
      correlationId: "corr-456",
      occurredAt: new Date("2026-01-02T00:00:00.000Z"),
      outcome: "success",
      safeMetadata: {
        channel: "sms",
        failureReason: "submitted [REDACTED_CODE]",
      },
    });
  });
});
