import { type StaffSession, validateStaffSession } from "@clinic/auth";

export interface StaffSessionRouteContext {
  staffSession?: StaffSession;
  now?: Date;
  production?: boolean;
}

export interface StaffSessionResponse {
  authenticated: true;
  staff: {
    staffId: string;
    clinicId: string;
    role: StaffSession["identity"]["role"];
    displayName?: string;
  };
  mfa: {
    required: boolean;
    verified: boolean;
    enrolledMethods: StaffSession["mfa"]["enrolledMethods"];
  };
  expiresAt: string;
}

export interface StaffSessionErrorResponse {
  authenticated: false;
  error: "staff_session_required" | "staff_session_invalid";
}

export type StaffSessionRouteResult =
  | {
      status: 200;
      body: StaffSessionResponse;
    }
  | {
      status: 401;
      body: StaffSessionErrorResponse;
    };

export function createStaffSessionResponse(
  session: StaffSession,
): StaffSessionResponse {
  return {
    authenticated: true,
    staff: {
      staffId: session.identity.staffId,
      clinicId: session.identity.clinicId,
      role: session.identity.role,
      displayName: session.identity.displayName,
    },
    mfa: {
      required: session.mfa.required,
      verified: Boolean(session.mfa.verifiedAt),
      enrolledMethods: session.mfa.enrolledMethods,
    },
    expiresAt: session.expiresAt,
  };
}

export function getStaffSessionRoute(
  context: StaffSessionRouteContext,
): StaffSessionRouteResult {
  if (!context.staffSession) {
    return {
      status: 401,
      body: {
        authenticated: false,
        error: "staff_session_required",
      },
    };
  }

  const validation = validateStaffSession(context.staffSession, context.now, {
    production: context.production,
  });

  if (!validation.ok) {
    return {
      status: 401,
      body: {
        authenticated: false,
        error: "staff_session_invalid",
      },
    };
  }

  return {
    status: 200,
    body: createStaffSessionResponse(context.staffSession),
  };
}
