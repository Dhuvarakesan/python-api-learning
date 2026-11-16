export type PrincipalRole = "patient" | "staff" | "system";

export const STAFF_ROLES = [
  "clinic_administrator",
  "scheduling_staff",
] as const;

export type StaffRole = (typeof STAFF_ROLES)[number];

export type StaffMfaMethod = "totp" | "webauthn" | "sms" | "email";

export interface AuthenticatedPrincipal {
  id: string;
  role: PrincipalRole;
}

export interface StaffIdentity {
  staffId: string;
  clinicId: string;
  role: StaffRole;
  authProviderSubject: string;
  displayName?: string;
}

export interface StaffMfaState {
  required: boolean;
  verifiedAt?: string;
  enrolledMethods: StaffMfaMethod[];
}

export interface StaffAccountPolicy {
  sharedProductionAccount: boolean;
}

export interface StaffSession {
  sessionId: string;
  issuedAt: string;
  expiresAt: string;
  identity: StaffIdentity;
  mfa: StaffMfaState;
  accountPolicy: StaffAccountPolicy;
}

export interface StaffSessionValidationResult {
  ok: boolean;
  reason?: "expired" | "mfa_required" | "shared_production_account";
}

export function canManageSchedule(principal: AuthenticatedPrincipal): boolean {
  return principal.role === "staff" || principal.role === "system";
}

export function isStaffRole(value: string): value is StaffRole {
  return STAFF_ROLES.includes(value as StaffRole);
}

export function staffHasAnyRole(
  session: StaffSession,
  requiredRoles: readonly StaffRole[],
): boolean {
  return requiredRoles.includes(session.identity.role);
}

export function validateStaffSession(
  session: StaffSession,
  now: Date = new Date(),
  options: { production?: boolean } = {},
): StaffSessionValidationResult {
  if (Date.parse(session.expiresAt) <= now.getTime()) {
    return {
      ok: false,
      reason: "expired",
    };
  }

  if (session.mfa.required && !session.mfa.verifiedAt) {
    return {
      ok: false,
      reason: "mfa_required",
    };
  }

  if (options.production && session.accountPolicy.sharedProductionAccount) {
    return {
      ok: false,
      reason: "shared_production_account",
    };
  }

  return {
    ok: true,
  };
}
