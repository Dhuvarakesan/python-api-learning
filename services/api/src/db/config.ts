export type DatabaseEnvironment = 'development' | 'test' | 'production';

export interface DatabaseConfig {
  readonly environment: DatabaseEnvironment;
  readonly databaseUrl: string;
  readonly ssl: boolean;
  readonly migrationsPath: string;
}

export function createDatabaseConfig(
  env: Record<string, string | undefined> = process.env,
): DatabaseConfig {
  const environment = resolveDatabaseEnvironment(env);
  const databaseUrl =
    environment === 'test'
      ? env.TEST_DATABASE_URL ?? env.DATABASE_URL
      : env.DATABASE_URL;

  if (!databaseUrl) {
    throw new Error(
      environment === 'test'
        ? 'TEST_DATABASE_URL or DATABASE_URL is required for database access.'
        : 'DATABASE_URL is required for database access.',
    );
  }

  return {
    environment,
    databaseUrl,
    ssl: resolveSslRequirement(environment, env),
    migrationsPath: env.DB_MIGRATIONS_PATH ?? 'infra/database/migrations',
  };
}

function resolveDatabaseEnvironment(
  env: Record<string, string | undefined>,
): DatabaseEnvironment {
  const raw = env.CLINIC_ENV ?? env.NODE_ENV ?? 'development';

  if (raw === 'production' || raw === 'test') {
    return raw;
  }

  return 'development';
}

function resolveSslRequirement(
  environment: DatabaseEnvironment,
  env: Record<string, string | undefined>,
): boolean {
  if (env.DB_SSL === 'true') {
    return true;
  }

  if (env.DB_SSL === 'false') {
    return false;
  }

  return environment === 'production';
}
