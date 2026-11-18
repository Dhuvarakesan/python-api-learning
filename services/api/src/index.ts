import { createServiceConfig } from '@clinic/config';
import type { AppointmentRequest } from '@clinic/shared';
export {
  createApiServer,
  startApiServer,
  type ApiResponse,
  type ApiServerOptions,
  type RouteContext,
  type RouteHandler,
  type RouteRegistry,
} from './server.js';
export {
  CORRELATION_ID_HEADER,
  allowAnonymousAuthorization,
  consoleStructuredLogger,
  createRequestContext,
  createRequestLogger,
  requireRoles,
  type AuthorizationHooks,
  type AuthorizedPrincipal,
  type RequestContext,
  type StaffRole,
  type StructuredLogEntry,
  type StructuredLogger,
} from './middleware/index.js';
export {
  createDatabaseConfig,
  createPostgresClient,
  loadMigrations,
  migrateDatabase,
  runMigrations,
  type DatabaseClient,
  type DatabaseConfig,
  type DatabaseEnvironment,
  type DatabaseQueryResult,
  type DatabaseValue,
  type ManagedDatabaseClient,
  type Migration,
  type MigrationResult,
} from './db/index.js';
export {
  SqlAuditEventRepository,
  type AuditActorType,
  type AuditEvent,
  type AuditEventRepository,
  type CreateAuditEventInput,
} from './repositories/index.js';
export {
  createApiConfig,
  type ApiConfig,
  type ApiHttpConfig,
  type ApiRuntimeMode,
  type ApiSecretsConfig,
} from './config/index.js';
export {
  PatientSafeError,
  createPatientSafeError,
  createPhiSafeMetadata,
  describeSecret,
  loadSecretReference,
  normalizePatientSafeError,
  requireSecretReference,
  resolveSecretValue,
  summarizeUnknownError,
  toPatientSafeErrorResponse,
  type EnvSource,
  type ManagedSecretResolver,
  type PatientSafeErrorOptions,
  type PatientSafeErrorResponse,
  type PhiSafeMetadata,
  type PhiSafeMetadataValue,
  type SecretReference,
  type SecretSource,
} from './security/index.js';
export {
  AUDIT_EVENT_TYPES,
  createAuditService,
  type AuditActorContext,
  type AuditEntityReference,
  type AuditEventType,
  type AuditRequestContext,
  type AuditService,
  type WriteAuditEventInput,
} from './modules/audit/index.js';

export interface ApiService {
  readonly name: 'clinic-api';
  readonly health: () => {
    readonly ok: true;
    readonly service: 'clinic-api';
  };
  readonly summarizeRequest: (request: AppointmentRequest) => string;
}

export function createApiService(
  env: Record<string, string | undefined> = process.env,
): ApiService {
  const config = createServiceConfig(env);

  return {
    name: 'clinic-api',
    health: () => ({
      ok: true,
      service: 'clinic-api',
    }),
    summarizeRequest: (request) =>
      `${request.id}:${request.status}:${config.mode}`,
  };
}
