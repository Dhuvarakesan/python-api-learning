import type { RouteRegistry } from '../../server.js';

export function registerAuditRoutes(_registry: RouteRegistry): void {
  // Audit routes are registered by later feature tasks.
}

export {
  AUDIT_EVENT_TYPES,
  createAuditService,
  type AuditActorContext,
  type AuditEntityReference,
  type AuditEventType,
  type AuditRequestContext,
  type AuditService,
  type WriteAuditEventInput,
} from './auditService.js';
