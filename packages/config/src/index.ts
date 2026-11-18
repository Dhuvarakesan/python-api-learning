export type RuntimeMode = 'development' | 'test' | 'production';

export interface ServiceConfig {
  readonly mode: RuntimeMode;
  readonly apiBaseUrl: string;
  readonly staffBaseUrl: string;
}

export type EnvSource = Record<string, string | undefined>;

export function getRequiredEnv(env: EnvSource, name: string): string {
  const value = env[name];

  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value;
}

export function getOptionalEnv(
  env: EnvSource,
  name: string,
  fallback: string,
): string {
  return env[name] ?? fallback;
}

export function createServiceConfig(env: EnvSource): ServiceConfig {
  const rawMode = getOptionalEnv(env, 'NODE_ENV', 'development');
  const mode: RuntimeMode =
    rawMode === 'production' || rawMode === 'test' ? rawMode : 'development';

  return {
    mode,
    apiBaseUrl: getOptionalEnv(env, 'CLINIC_API_BASE_URL', '/api'),
    staffBaseUrl: getOptionalEnv(env, 'CLINIC_STAFF_BASE_URL', '/staff'),
  };
}
