import type { CorrelationId, PatientSafeApiResponse } from "@clinic/contracts";
import { createSuccessResponse } from "@clinic/contracts";

export const HEALTH_ROUTE = "/api/v1/health";

export interface HealthStatus {
  status: "ok";
  service: "clinic-api";
  apiVersion: "v1";
}

export function isHealthRoute(url: string | undefined): boolean {
  if (!url) {
    return false;
  }

  return new URL(url, "http://clinic-api.local").pathname === HEALTH_ROUTE;
}

export function createHealthResponse(
  correlationId: CorrelationId,
): PatientSafeApiResponse<HealthStatus> {
  return createSuccessResponse(
    {
      status: "ok",
      service: "clinic-api",
      apiVersion: "v1",
    },
    correlationId,
  );
}
