import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const migrationsDir = dirname(fileURLToPath(import.meta.url));
const defaultLocalDatabaseUrl =
  "postgres://postgres:postgres@localhost:5432/clinic_booking";
const databaseUrl =
  process.env.DATABASE_URL ??
  process.env.LOCAL_DATABASE_URL ??
  defaultLocalDatabaseUrl;

function runPsql(args, options = {}) {
  const result = spawnSync(
    "psql",
    ["--set", "ON_ERROR_STOP=1", databaseUrl, ...args],
    {
      encoding: "utf8",
      stdio: options.capture ? "pipe" : "inherit",
    },
  );

  if (result.error) {
    throw new Error(`Unable to run psql: ${result.error.message}`);
  }

  if (result.status !== 0) {
    throw new Error(`psql exited with status ${result.status}`);
  }

  return result.stdout ?? "";
}

function ensureMigrationTable() {
  runPsql([
    "--command",
    `CREATE TABLE IF NOT EXISTS schema_migrations (
      version text PRIMARY KEY,
      applied_at timestamptz NOT NULL DEFAULT now()
    );`,
  ]);
}

function appliedVersions() {
  const output = runPsql(
    [
      "--tuples-only",
      "--no-align",
      "--command",
      "SELECT version FROM schema_migrations ORDER BY version;",
    ],
    { capture: true },
  );

  return new Set(
    output
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
  );
}

function migrationFiles() {
  return readdirSync(migrationsDir)
    .filter((fileName) => /^\d+_.+\.sql$/.test(fileName))
    .sort();
}

function applyMigration(fileName) {
  const sql = readFileSync(join(migrationsDir, fileName), "utf8");
  const version = fileName.replace(/\.sql$/, "");
  const escapedVersion = version.replaceAll("'", "''");

  runPsql([
    "--command",
    `BEGIN;
${sql}
INSERT INTO schema_migrations (version)
VALUES ('${escapedVersion}')
ON CONFLICT (version) DO NOTHING;
COMMIT;`,
  ]);
}

ensureMigrationTable();

const applied = appliedVersions();
const pending = migrationFiles().filter(
  (fileName) => !applied.has(fileName.replace(/\.sql$/, "")),
);

if (pending.length === 0) {
  console.log("No pending database migrations.");
  process.exit(0);
}

for (const fileName of pending) {
  console.log(`Applying database migration ${fileName}`);
  applyMigration(fileName);
}

console.log(`Applied ${pending.length} database migration(s).`);
