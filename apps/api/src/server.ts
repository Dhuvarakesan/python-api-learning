import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { createApiErrorResponse } from "@clinic/contracts";
import { createLogger } from "@clinic/observability";
import { getCorrelationId } from "./middleware/correlation.js";
import { sendJson } from "./middleware/json.js";
import { createHealthResponse, isHealthRoute } from "./routes/health.js";

export const API_VERSION_PREFIX = "/api/v1";

const logger = createLogger({ service: "clinic-api" });

export function createApiServer() {
  return createServer((request: IncomingMessage, response: ServerResponse) => {
    const correlationId = getCorrelationId(request);
    const startedAt = performance.now();
    const requestPath = new URL(request.url ?? "/", "http://clinic-api.local")
      .pathname;

    response.setHeader("x-correlation-id", correlationId);
    response.once("finish", () => {
      logger.info("api.request.completed", {
        correlationId,
        method: request.method ?? "UNKNOWN",
        path: requestPath,
        statusCode: response.statusCode,
        durationMs: Math.round(performance.now() - startedAt),
      });
    });

    if (request.method === "GET" && isHealthRoute(request.url)) {
      sendJson(response, 200, createHealthResponse(correlationId));
      return;
    }

    if (isHealthRoute(request.url)) {
      sendJson(
        response,
        405,
        createApiErrorResponse(
          "METHOD_NOT_ALLOWED",
          "Method not allowed.",
          correlationId,
        ),
      );
      return;
    }

    sendJson(
      response,
      404,
      createApiErrorResponse("NOT_FOUND", "Route not found.", correlationId),
    );
  });
}
