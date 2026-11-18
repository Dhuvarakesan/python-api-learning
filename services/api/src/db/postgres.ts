import pg from 'pg';
import type { DatabaseClient, DatabaseQueryResult, DatabaseValue } from './types.js';
import type { DatabaseConfig } from './config.js';

const { Pool } = pg;

export interface ManagedDatabaseClient extends DatabaseClient {
  readonly close: () => Promise<void>;
}

export function createPostgresClient(config: DatabaseConfig): ManagedDatabaseClient {
  const pool = new Pool({
    connectionString: config.databaseUrl,
    ssl: config.ssl ? { rejectUnauthorized: true } : undefined,
  });

  return {
    query: async <Row extends Record<string, unknown> = Record<string, unknown>>(
      sql: string,
      values: readonly DatabaseValue[] = [],
    ): Promise<DatabaseQueryResult<Row>> => {
      const result = await pool.query<Row>(sql, [...values]);

      return {
        rows: result.rows,
        rowCount: result.rowCount ?? result.rows.length,
      };
    },
    transaction: async <Result>(
      work: (client: DatabaseClient) => Promise<Result>,
    ): Promise<Result> => {
      const poolClient = await pool.connect();
      const transactionClient: DatabaseClient = {
        query: async <
          Row extends Record<string, unknown> = Record<string, unknown>,
        >(
          sql: string,
          values: readonly DatabaseValue[] = [],
        ): Promise<DatabaseQueryResult<Row>> => {
          const result = await poolClient.query<Row>(sql, [...values]);

          return {
            rows: result.rows,
            rowCount: result.rowCount ?? result.rows.length,
          };
        },
      };

      try {
        await poolClient.query('BEGIN');
        const result = await work(transactionClient);
        await poolClient.query('COMMIT');
        return result;
      } catch (error) {
        await poolClient.query('ROLLBACK');
        throw error;
      } finally {
        poolClient.release();
      }
    },
    close: () => pool.end(),
  };
}
