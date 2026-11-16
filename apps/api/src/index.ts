import { pathToFileURL } from "node:url";
import { createApiServer } from "./server.js";

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number.parseInt(process.env.PORT ?? "3000", 10);
  const host = process.env.HOST ?? "0.0.0.0";

  const server = createApiServer();

  server.listen(port, host, () => {
    console.log(`clinic-api listening on http://${host}:${port}`);
  });
}

export function createHealthResponse(): HealthResponse {
  return {
    status: "ok",
    service: "clinic-api",
  };
}

export { createApiServerDefinition } from "./server.js";
export type {
  ApiRouteDefinition,
  ApiServerDefinition,
  ProtectedStaffRouteDefinition,
} from "./server.js";
