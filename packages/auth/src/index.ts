export type PrincipalRole = "patient" | "staff" | "system";

export interface AuthenticatedPrincipal {
  id: string;
  role: PrincipalRole;
}

export function canManageSchedule(principal: AuthenticatedPrincipal): boolean {
  return principal.role === "staff" || principal.role === "system";
}
