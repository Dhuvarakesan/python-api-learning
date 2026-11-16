export interface DatabaseRuntimeConfig {
  connectionStringEnvVar: "DATABASE_URL";
  requireSsl: boolean;
}

export const databaseRuntimeConfig: DatabaseRuntimeConfig = {
  connectionStringEnvVar: "DATABASE_URL",
  requireSsl: true,
};
