import { describe, expect, it } from 'vitest';
import {
  authorizeAdmin,
  authorizeStaff,
  hasPermission,
  type StaffPrincipal,
} from './rbac';

const admin: StaffPrincipal = {
  id: 'staff-admin',
  role: 'clinic_admin',
  active: true,
  mfaVerified: true,
};

const frontDesk: StaffPrincipal = {
  id: 'staff-front',
  role: 'front_desk',
  active: true,
  mfaVerified: true,
};

describe('rbac helpers', () => {
  it('allows authorized staff actions', () => {
    expect(authorizeStaff(frontDesk, 'appointment:write')).toEqual({ allowed: true });
    expect(hasPermission(admin, 'schedule:configure')).toBe(true);
  });

  it('denies unauthorized staff and admin actions', () => {
    expect(authorizeStaff(frontDesk, 'staff:manage')).toEqual({
      allowed: false,
      reason: 'insufficient_role',
    });
    expect(authorizeAdmin(frontDesk)).toEqual({
      allowed: false,
      reason: 'insufficient_role',
    });
    expect(authorizeStaff({ ...admin, mfaVerified: false }, 'staff:manage')).toEqual({
      allowed: false,
      reason: 'mfa_required',
    });
  });
});
