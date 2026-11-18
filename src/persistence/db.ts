import { Pool, type PoolClient, type PoolConfig, type QueryResult, type QueryResultRow } from 'pg';

import { initialSchemaSql } from './schema';

type DatabaseEnvironment = NodeJS.ProcessEnv;

export interface SafeDatabaseConfigSummary {
  host: string;
  port: number;
  database: string;
  user: string;
  ssl: boolean;
}

export interface Database {
  query<T extends QueryResultRow = QueryResultRow>(
    text: string,
    values?: unknown[],
  ): Promise<QueryResult<T>>;
  withTransaction<T>(operation: (client: PoolClient) => Promise<T>): Promise<T>;
  migrate(): Promise<void>;
  close(): Promise<void>;
}

export function readDatabaseConfig(env: DatabaseEnvironment = process.env): PoolConfig {
  const connectionString = env.DATABASE_URL;
  const sslEnabled = env.DB_SSL === 'true' || env.NODE_ENV === 'production';

  if (connectionString) {
    return {
      connectionString,
      ssl: sslEnabled ? { rejectUnauthorized: env.DB_SSL_REJECT_UNAUTHORIZED !== 'false' } : undefined,
      max: Number(env.DB_POOL_MAX ?? 10),
      application_name: env.DB_APPLICATION_NAME ?? 'clinic-booking-app',
    };
  }

  const database = env.DB_NAME;
  const user = env.DB_USER;

  if (!database || !user) {
    throw new Error('Database configuration requires DATABASE_URL or DB_NAME and DB_USER.');
  }

  return {
    host: env.DB_HOST ?? 'localhost',
    port: Number(env.DB_PORT ?? 5432),
    database,
    user,
    password: env.DB_PASSWORD,
    ssl: sslEnabled ? { rejectUnauthorized: env.DB_SSL_REJECT_UNAUTHORIZED !== 'false' } : undefined,
    max: Number(env.DB_POOL_MAX ?? 10),
    application_name: env.DB_APPLICATION_NAME ?? 'clinic-booking-app',
  };
}

export function summarizeDatabaseConfig(config: PoolConfig): SafeDatabaseConfigSummary {
  return {
    host: config.host ?? (config.connectionString ? '<connection-string-host>' : 'localhost'),
    port: Number(config.port ?? 5432),
    database: config.database ?? '<connection-string-database>',
    user: config.user ?? '<connection-string-user>',
    ssl: Boolean(config.ssl),
  };
}

export function createPool(config: PoolConfig = readDatabaseConfig()): Pool {
  return new Pool(config);
}

export async function migrateDatabase(pool: Pool): Promise<void> {
  const client = await pool.connect();

  try {
    await client.query('BEGIN');
    await client.query(initialSchemaSql);
    await client.query('COMMIT');
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}

export function createDatabase(pool: Pool = createPool()): Database {
  return {
    query<T extends QueryResultRow = QueryResultRow>(
      text: string,
      values?: unknown[],
    ): Promise<QueryResult<T>> {
      return pool.query<T>(text, values);
    },

    async withTransaction<T>(operation: (client: PoolClient) => Promise<T>): Promise<T> {
      const client = await pool.connect();

      try {
        await client.query('BEGIN');
        const result = await operation(client);
        await client.query('COMMIT');
        return result;
      } catch (error) {
        await client.query('ROLLBACK');
        throw error;
      } finally {
        client.release();
      }
    },

    migrate(): Promise<void> {
      return migrateDatabase(pool);
    },

    close(): Promise<void> {
      return pool.end();
    },
  };
}
