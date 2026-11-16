import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createApiErrorResponse } from "@clinic/contracts";
import { getCorrelationId } from "./middleware/correlation.js";
import { sendJson } from "./middleware/json.js";
import { createHealthResponse, isHealthRoute } from "./routes/health.js";

export const API_VERSION_PREFIX = "/api/v1";

export function createApiServer() {
  return createServer((request: IncomingMessage, response: ServerResponse) => {
    const correlationId = getCorrelationId(request);
    response.setHeader("x-correlation-id", correlationId);

    if (request.method === "GET" && isHealthRoute(request.url)) {
      sendJson(response, 200, createHealthResponse(correlationId));
      return;
    }

    if (isHealthRoute(request.url)) {
      sendJson(
        response,
        405,
        createApiErrorResponse("METHOD_NOT_ALLOWED", "Method not allowed.", correlationId),
      );
      return;
    }

    sendJson(response, 404, createApiErrorResponse("NOT_FOUND", "Route not found.", correlationId));
  });
}
