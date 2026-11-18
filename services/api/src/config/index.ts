import {
  loadSecretReference,
  type EnvSource,
  type SecretReference,
} from '../security/index.js';

export type ApiRuntimeMode = 'development' | 'test' | 'production';

export interface ApiHttpConfig {
  readonly host: string;
  readonly port: number;
}

export interface ApiSecretsConfig {
  readonly databaseUrl?: SecretReference;
  readonly sessionSigningKey?: SecretReference;
  readonly webhookSigningSecret?: SecretReference;
}

export interface ApiConfig {
  readonly mode: ApiRuntimeMode;
  readonly http: ApiHttpConfig;
  readonly secrets: ApiSecretsConfig;
}

export function createApiConfig(
  env: EnvSource = process.env,
): ApiConfig {
  return {
    mode: resolveRuntimeMode(env),
    http: {
      host: env.HOST ?? '0.0.0.0',
      port: parsePort(env.PORT),
    },
    secrets: {
      databaseUrl: loadSecretReference(env, 'DATABASE_URL'),
      sessionSigningKey: loadSecretReference(env, 'SESSION_SIGNING_KEY'),
      webhookSigningSecret: loadSecretReference(env, 'WEBHOOK_SIGNING_SECRET'),
    },
  };
}

function resolveRuntimeMode(env: EnvSource): ApiRuntimeMode {
  const raw = env.CLINIC_ENV ?? env.NODE_ENV ?? 'development';

  if (raw === 'production' || raw === 'test') {
    return raw;
  }

  return 'development';
}

function parsePort(value: string | undefined): number {
  if (!value) {
    return 3000;
  }

  const port = Number.parseInt(value, 10);

  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new Error('PORT must be an integer between 1 and 65535.');
  }

  return port;
}
