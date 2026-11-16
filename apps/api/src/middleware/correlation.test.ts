import type { IncomingMessage } from "node:http";
import { describe, expect, it } from "vitest";
import { getCorrelationId } from "./correlation.js";

describe("getCorrelationId", () => {
  it("uses a trimmed inbound correlation ID", () => {
    const request = {
      headers: {
        "x-correlation-id": " corr-123 ",
      },
    } as unknown as IncomingMessage;

    expect(getCorrelationId(request)).toBe("corr-123");
  });

  it("creates a correlation ID when the request does not provide one", () => {
    const request = {
      headers: {},
    } as unknown as IncomingMessage;

    expect(getCorrelationId(request)).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });
});
