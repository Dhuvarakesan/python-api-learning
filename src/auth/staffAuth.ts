import type { StaffUser } from '../domain/scheduling/types';
import {
  authorizeAdmin,
  authorizeStaff,
  principalFromStaffUser,
  type AuthorizationOutcome,
  type StaffPermission,
  type StaffPrincipal,
} from './rbac';

export interface StaffSession {
  id: string;
  staffUserId: string;
  expiresAt: string;
  mfaVerifiedAt: string | null;
  createdAt: string;
}

export interface StaffSessionRecord {
  session: StaffSession;
  staffUser: StaffUser;
}

export interface StaffSessionStore {
  findByToken(sessionToken: string): Promise<StaffSessionRecord | null>;
  touch?(sessionId: string, authenticatedAt: string): Promise<void>;
}

export interface MfaVerifier {
  verify(input: { staffUserId: string; factorId?: string; code: string; correlationId: string }): Promise<boolean>;
}

export interface StaffAuthContext {
  principal: StaffPrincipal;
  session: StaffSession;
  staffUser: StaffUser;
}

export interface StaffAuthenticationResult {
  authenticated: boolean;
  context: StaffAuthContext | null;
  reason?: 'missing_token' | 'invalid_session' | 'expired_session' | 'inactive_staff';
}

export async function authenticateStaffSession(
  sessionToken: string | null | undefined,
  store: StaffSessionStore,
  now: Date = new Date(),
): Promise<StaffAuthenticationResult> {
  if (!sessionToken) {
    return { authenticated: false, context: null, reason: 'missing_token' };
  }

  const record = await store.findByToken(sessionToken);

  if (!record) {
    return { authenticated: false, context: null, reason: 'invalid_session' };
  }

  if (Date.parse(record.session.expiresAt) <= now.getTime()) {
    return { authenticated: false, context: null, reason: 'expired_session' };
  }

  if (!record.staffUser.active) {
    return { authenticated: false, context: null, reason: 'inactive_staff' };
  }

  await store.touch?.(record.session.id, now.toISOString());

  return {
    authenticated: true,
    context: {
      principal: principalFromStaffUser(record.staffUser, Boolean(record.session.mfaVerifiedAt)),
      session: record.session,
      staffUser: record.staffUser,
    },
  };
}

export async function verifyStaffMfa(
  input: {
    staffUserId: string;
    code: string;
    correlationId: string;
    factorId?: string;
  },
  verifier: MfaVerifier,
): Promise<boolean> {
  if (!input.code.trim()) {
    return false;
  }

  return verifier.verify(input);
}

export function requireStaff(
  context: StaffAuthContext | null | undefined,
  permission?: StaffPermission,
): AuthorizationOutcome {
  return authorizeStaff(context?.principal, permission);
}

export function requireAdmin(context: StaffAuthContext | null | undefined): AuthorizationOutcome {
  return authorizeAdmin(context?.principal);
}
