export type DatabaseValue =
  | string
  | number
  | boolean
  | Date
  | null
  | Record<string, unknown>
  | readonly unknown[];

export interface DatabaseQueryResult<Row extends Record<string, unknown>> {
  readonly rows: readonly Row[];
  readonly rowCount: number;
}

export interface DatabaseClient {
  readonly query: <Row extends Record<string, unknown> = Record<string, unknown>>(
    sql: string,
    values?: readonly DatabaseValue[],
  ) => Promise<DatabaseQueryResult<Row>>;
  readonly transaction?: <Result>(
    work: (client: DatabaseClient) => Promise<Result>,
  ) => Promise<Result>;
}
