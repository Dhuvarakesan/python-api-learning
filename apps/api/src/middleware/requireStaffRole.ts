import {
  type StaffRole,
  type StaffSession,
  staffHasAnyRole,
  validateStaffSession,
} from "@clinic/auth";
import {
  type ObservabilityEvent,
  redactSensitiveAttributeNames,
} from "@clinic/observability";

export interface StaffRequestContext {
  method?: string;
  path?: string;
  requestId?: string;
  staffSession?: StaffSession;
}

export interface StaffAccessDeniedResponse {
  status: 401 | 403;
  body: {
    error: "staff_authentication_required" | "staff_authorization_failed";
    message: string;
  };
  auditEvent: ObservabilityEvent;
}

export interface StaffAccessGrantedResponse {
  status: 200;
  staffSession: StaffSession;
}

export type StaffAccessResult =
  StaffAccessGrantedResponse | StaffAccessDeniedResponse;

export type StaffAuditSink = (event: ObservabilityEvent) => void;

export interface RequireStaffRoleOptions {
  audit?: StaffAuditSink;
  now?: Date;
  production?: boolean;
}

export function createUnauthorizedStaffAuditEvent(
  context: StaffRequestContext,
  requiredRoles: readonly StaffRole[],
  reason: string,
): ObservabilityEvent {
  return redactSensitiveAttributeNames({
    level: "warn",
    message: "staff_access_denied",
    attributes: {
      reason,
      method: context.method ?? "unknown",
      path: context.path ?? "unknown",
      requestId: context.requestId ?? "unknown",
      requiredRoles: requiredRoles.join(","),
      staffRole: context.staffSession?.identity.role ?? "none",
      clinicId: context.staffSession?.identity.clinicId ?? "unknown",
    },
  });
}

export function requireStaffRole(
  requiredRoles: readonly StaffRole[],
  options: RequireStaffRoleOptions = {},
): (context: StaffRequestContext) => StaffAccessResult {
  return (context) => {
    const denied = (
      status: 401 | 403,
      reason: string,
    ): StaffAccessDeniedResponse => {
      const auditEvent = createUnauthorizedStaffAuditEvent(
        context,
        requiredRoles,
        reason,
      );

      options.audit?.(auditEvent);

      return {
        status,
        body: {
          error:
            status === 401
              ? "staff_authentication_required"
              : "staff_authorization_failed",
          message:
            status === 401
              ? "A valid staff session is required."
              : "The staff session does not have the required role.",
        },
        auditEvent,
      };
    };

    if (!context.staffSession) {
      return denied(401, "missing_staff_session");
    }

    const validation = validateStaffSession(context.staffSession, options.now, {
      production: options.production,
    });

    if (!validation.ok) {
      return denied(401, validation.reason ?? "invalid_staff_session");
    }

    if (!staffHasAnyRole(context.staffSession, requiredRoles)) {
      return denied(403, "insufficient_staff_role");
    }

    return {
      status: 200,
      staffSession: context.staffSession,
    };
  };
}
