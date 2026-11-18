import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { DatabaseClient } from './types.js';

export interface Migration {
  readonly id: string;
  readonly checksum: string;
  readonly sql: string;
}

export interface MigrationResult {
  readonly applied: readonly string[];
  readonly skipped: readonly string[];
}

interface AppliedMigrationRow extends Record<string, unknown> {
  readonly id: string;
  readonly checksum: string;
}

export async function loadMigrations(
  migrationsPath: string,
  cwd: string = process.cwd(),
): Promise<readonly Migration[]> {
  const absolutePath = resolve(cwd, migrationsPath);
  const entries = await readdir(absolutePath, { withFileTypes: true });
  const migrationFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.sql'))
    .map((entry) => entry.name)
    .sort((left, right) => left.localeCompare(right));

  return Promise.all(
    migrationFiles.map(async (fileName) => {
      const sql = await readFile(resolve(absolutePath, fileName), 'utf8');

      return {
        id: fileName,
        checksum: createHash('sha256').update(sql).digest('hex'),
        sql,
      };
    }),
  );
}

export async function runMigrations(
  client: DatabaseClient,
  migrations: readonly Migration[],
): Promise<MigrationResult> {
  await ensureMigrationTable(client);

  const applied: string[] = [];
  const skipped: string[] = [];
  const existing = await loadAppliedMigrations(client);

  for (const migration of migrations) {
    const existingChecksum = existing.get(migration.id);

    if (existingChecksum === migration.checksum) {
      skipped.push(migration.id);
      continue;
    }

    if (existingChecksum && existingChecksum !== migration.checksum) {
      throw new Error(
        `Migration ${migration.id} checksum changed after it was applied.`,
      );
    }

    await runInTransaction(client, async (transactionClient) => {
      await transactionClient.query(migration.sql);
      await transactionClient.query(
        `INSERT INTO schema_migrations (id, checksum, applied_at)
         VALUES ($1, $2, now())`,
        [migration.id, migration.checksum],
      );
    });

    applied.push(migration.id);
    existing.set(migration.id, migration.checksum);
  }

  return { applied, skipped };
}

async function runInTransaction(
  client: DatabaseClient,
  work: (client: DatabaseClient) => Promise<void>,
): Promise<void> {
  if (client.transaction) {
    await client.transaction(work);
    return;
  }

  await client.query('BEGIN');
  try {
    await work(client);
    await client.query('COMMIT');
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  }
}

async function ensureMigrationTable(client: DatabaseClient): Promise<void> {
  await client.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      id text PRIMARY KEY,
      checksum text NOT NULL,
      applied_at timestamptz NOT NULL DEFAULT now()
    )
  `);
}

async function loadAppliedMigrations(
  client: DatabaseClient,
): Promise<Map<string, string>> {
  const result = await client.query<AppliedMigrationRow>(
    'SELECT id, checksum FROM schema_migrations ORDER BY id',
  );

  return new Map(
    result.rows.map((row) => [String(row.id), String(row.checksum)]),
  );
}
