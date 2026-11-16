import { type StaffRole } from "@clinic/auth";
import {
  requireStaffRole,
  type RequireStaffRoleOptions,
  type StaffRequestContext,
} from "./middleware/requireStaffRole.js";
import {
  getStaffSessionRoute,
  type StaffSessionRouteResult,
} from "./routes/staff/session.js";

export interface ApiRouteDefinition {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: string;
  requiredStaffRoles?: readonly StaffRole[];
}

export interface ProtectedStaffRouteDefinition extends ApiRouteDefinition {
  requiredStaffRoles: readonly StaffRole[];
  authorize: ReturnType<typeof requireStaffRole>;
  handle: (context: StaffRequestContext) => StaffSessionRouteResult;
}

export interface ApiServerDefinition {
  routes: {
    staffSession: ProtectedStaffRouteDefinition;
  };
}

const STAFF_ROUTE_ROLES = [
  "clinic_administrator",
  "scheduling_staff",
] as const satisfies readonly StaffRole[];

export function createApiServerDefinition(
  options: RequireStaffRoleOptions = {},
): ApiServerDefinition {
  return {
    routes: {
      staffSession: {
        method: "GET",
        path: "/staff/session",
        requiredStaffRoles: STAFF_ROUTE_ROLES,
        authorize: requireStaffRole(STAFF_ROUTE_ROLES, options),
        handle: (context) =>
          getStaffSessionRoute({
            staffSession: context.staffSession,
            now: options.now,
            production: options.production,
          }),
      },
    },
  };
}
