import type { StaffRole, StaffUser } from '../domain/scheduling/types';

export type StaffPermission =
  | 'appointment:read'
  | 'appointment:write'
  | 'schedule:configure'
  | 'staff:manage'
  | 'audit:read'
  | 'integration:manage';

export type AuthorizationOutcome =
  | { allowed: true }
  | { allowed: false; reason: 'missing_session' | 'inactive_staff' | 'mfa_required' | 'insufficient_role' };

export interface StaffPrincipal {
  id: string;
  role: StaffRole;
  active: boolean;
  mfaVerified: boolean;
}

const rolePermissions: Record<StaffRole, ReadonlySet<StaffPermission>> = {
  front_desk: new Set(['appointment:read', 'appointment:write']),
  clinic_admin: new Set([
    'appointment:read',
    'appointment:write',
    'schedule:configure',
    'staff:manage',
    'audit:read',
    'integration:manage',
  ]),
  technical_support: new Set(['audit:read', 'integration:manage']),
};

const roleRank: Record<StaffRole, number> = {
  front_desk: 1,
  technical_support: 2,
  clinic_admin: 3,
};

export function principalFromStaffUser(staffUser: StaffUser, mfaVerified = false): StaffPrincipal {
  return {
    id: staffUser.id,
    role: staffUser.role,
    active: staffUser.active,
    mfaVerified,
  };
}

export function hasRole(principal: StaffPrincipal | null | undefined, role: StaffRole): boolean {
  return Boolean(principal?.active && principal.role === role);
}

export function hasMinimumRole(
  principal: StaffPrincipal | null | undefined,
  role: StaffRole,
): boolean {
  return Boolean(principal?.active && roleRank[principal.role] >= roleRank[role]);
}

export function hasPermission(
  principal: StaffPrincipal | null | undefined,
  permission: StaffPermission,
): boolean {
  return Boolean(principal?.active && rolePermissions[principal.role].has(permission));
}

export function authorizeStaff(
  principal: StaffPrincipal | null | undefined,
  permission?: StaffPermission,
): AuthorizationOutcome {
  if (!principal) {
    return { allowed: false, reason: 'missing_session' };
  }

  if (!principal.active) {
    return { allowed: false, reason: 'inactive_staff' };
  }

  if (!principal.mfaVerified) {
    return { allowed: false, reason: 'mfa_required' };
  }

  if (permission && !hasPermission(principal, permission)) {
    return { allowed: false, reason: 'insufficient_role' };
  }

  return { allowed: true };
}

export function authorizeAdmin(principal: StaffPrincipal | null | undefined): AuthorizationOutcome {
  const staffOutcome = authorizeStaff(principal);

  if (!staffOutcome.allowed) {
    return staffOutcome;
  }

  return hasRole(principal, 'clinic_admin')
    ? { allowed: true }
    : { allowed: false, reason: 'insufficient_role' };
}

export function assertAuthorized(outcome: AuthorizationOutcome): asserts outcome is { allowed: true } {
  if (!outcome.allowed) {
    throw new Error(`Unauthorized: ${outcome.reason}`);
  }
}
