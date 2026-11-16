export interface HealthResponse {
  status: "ok";
  service: "clinic-api";
}

export function createHealthResponse(): HealthResponse {
  return {
    status: "ok",
    service: "clinic-api",
  };
}
