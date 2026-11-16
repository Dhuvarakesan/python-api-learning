import { describe, expect, it } from "vitest";
import { writeAuditEvent } from "./index.js";
import type { AuditSqlExecutor } from "./index.js";

describe("writeAuditEvent", () => {
  it("persists required audit fields and safe metadata", async () => {
    const queries: Array<{ sql: string; values: unknown[] }> = [];
    const executor: AuditSqlExecutor = {
      query: async (sql, values) => {
        queries.push({ sql, values });

        return {
          rows: [{ id: "audit-123", occurred_at: "2026-01-03T00:00:00.000Z" }],
        };
      },
    };

    const result = await writeAuditEvent(executor, {
      action: "booking.submission",
      actor: { type: "patient" },
      correlationId: "corr-789",
      occurredAt: new Date("2026-01-03T00:00:00.000Z"),
      references: {
        bookingRequestId: "11111111-1111-1111-1111-111111111111",
      },
      safeMetadata: {
        source: "patient_web",
        verificationCode: "123456",
        attemptNote: "entered 654321",
      },
    });

    expect(result).toEqual({
      auditId: "audit-123",
      action: "booking.submission",
      actorType: "patient",
      occurredAt: "2026-01-03T00:00:00.000Z",
      correlationId: "corr-789",
    });
    expect(queries).toHaveLength(1);
    expect(queries[0].sql).toContain("correlation_id");
    expect(queries[0].values[0]).toBe("booking.submission");
    expect(queries[0].values[1]).toBe("patient");
    expect(queries[0].values[5]).toBe("11111111-1111-1111-1111-111111111111");
    expect(queries[0].values[12]).toBe("corr-789");
    expect(JSON.parse(queries[0].values[13] as string)).toEqual({
      source: "patient_web",
      attemptNote: "entered [REDACTED_CODE]",
    });
  });
});
