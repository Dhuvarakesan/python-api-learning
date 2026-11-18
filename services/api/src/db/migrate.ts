import { createDatabaseConfig } from './config.js';
import { loadMigrations, runMigrations } from './migrations.js';
import { createPostgresClient } from './postgres.js';
import { pathToFileURL } from 'node:url';

export async function migrateDatabase(
  env: Record<string, string | undefined> = process.env,
): Promise<void> {
  const config = createDatabaseConfig(env);
  const migrations = await loadMigrations(config.migrationsPath);
  const client = createPostgresClient(config);

  try {
    const result = await runMigrations(client, migrations);
    console.info(
      JSON.stringify({
        level: 'info',
        message: 'database_migrations_complete',
        applied: result.applied,
        skipped: result.skipped,
      }),
    );
  } finally {
    await client.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  migrateDatabase().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : 'unknown_error';
    console.error(
      JSON.stringify({
        level: 'error',
        message: 'database_migrations_failed',
        error: message,
      }),
    );
    process.exitCode = 1;
  });
}
