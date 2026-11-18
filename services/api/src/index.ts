import { createServiceConfig } from '@clinic/config';
import type { AppointmentRequest } from '@clinic/shared';

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
