import { describe, expect, it } from "vitest";
import {
  createReminderJob,
  createReminderJobLogAttributes,
  ensureReminderJobCorrelationId,
} from "./index.js";

describe("reminder job correlation", () => {
  it("attaches a supplied correlation ID to new jobs", () => {
    expect(
      createReminderJob("appt-123", "2026-01-04T00:00:00.000Z", " corr-123 "),
    ).toEqual({
      appointmentId: "appt-123",
      scheduledFor: "2026-01-04T00:00:00.000Z",
      correlationId: "corr-123",
    });
  });

  it("creates a correlation ID when legacy job payloads do not have one", () => {
    const job = ensureReminderJobCorrelationId({
      appointmentId: "appt-456",
      scheduledFor: "2026-01-04T00:00:00.000Z",
    });

    expect(job.correlationId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  it("returns safe operational log attributes", () => {
    const job = createReminderJob(
      "appt-789",
      "2026-01-04T00:00:00.000Z",
      "corr-789",
    );

    expect(createReminderJobLogAttributes(job)).toEqual({
      correlationId: "corr-789",
      jobType: "appointment_reminder",
      appointmentId: "appt-789",
      scheduledFor: "2026-01-04T00:00:00.000Z",
    });
  });
});
