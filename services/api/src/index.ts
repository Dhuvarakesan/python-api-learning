import { createServiceConfig } from '@clinic/config';
import type { AppointmentRequest } from '@clinic/shared';
export {
  createApiServer,
  startApiServer,
  type ApiResponse,
  type ApiServerOptions,
  type RouteContext,
  type RouteHandler,
  type RouteRegistry,
} from './server.js';
export {
  CORRELATION_ID_HEADER,
  allowAnonymousAuthorization,
  consoleStructuredLogger,
  createRequestContext,
  createRequestLogger,
  requireRoles,
  type AuthorizationHooks,
  type AuthorizedPrincipal,
  type RequestContext,
  type StaffRole,
  type StructuredLogEntry,
  type StructuredLogger,
} from './middleware/index.js';

export interface ApiService {
  readonly name: 'clinic-api';
  readonly health: () => {
    readonly ok: true;
    readonly service: 'clinic-api';
  };
  readonly summarizeRequest: (request: AppointmentRequest) => string;
}

export function createApiService(
  env: Record<string, string | undefined> = process.env,
): ApiService {
  const config = createServiceConfig(env);

  return {
    name: 'clinic-api',
    health: () => ({
      ok: true,
      service: 'clinic-api',
    }),
    summarizeRequest: (request) =>
      `${request.id}:${request.status}:${config.mode}`,
  };
}
