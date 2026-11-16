const DEFAULT_LOCAL_DATABASE_URL =
  "postgres://postgres:postgres@localhost:5432/clinic_booking";

export interface DatabaseConfig {
  url: string;
  ssl: boolean;
  source: "DATABASE_URL" | "LOCAL_DATABASE_URL" | "default-local";
}

export function loadDatabaseConfig(
  env: NodeJS.ProcessEnv = process.env,
): DatabaseConfig {
  if (env.DATABASE_URL) {
    return {
      url: env.DATABASE_URL,
      ssl: env.DATABASE_SSL === "true",
      source: "DATABASE_URL",
    };
  }

  if (env.LOCAL_DATABASE_URL) {
    return {
      url: env.LOCAL_DATABASE_URL,
      ssl: false,
      source: "LOCAL_DATABASE_URL",
    };
  }

  return {
    url: DEFAULT_LOCAL_DATABASE_URL,
    ssl: false,
    source: "default-local",
  };
}

export { DEFAULT_LOCAL_DATABASE_URL };
