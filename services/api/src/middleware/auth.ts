import type { IncomingMessage, ServerResponse } from 'node:http';
import type { RequestContext } from './request-context.js';

export type StaffRole = 'admin' | 'clinician' | 'scheduler' | 'auditor';

export interface AuthorizedPrincipal {
  readonly id: string;
  readonly roles: readonly StaffRole[];
}

export interface AuthorizationHooks {
  readonly authorizeRequest: (
    request: IncomingMessage,
    context: RequestContext,
  ) => Promise<AuthorizedPrincipal | null> | AuthorizedPrincipal | null;
}

export const allowAnonymousAuthorization: AuthorizationHooks = {
  authorizeRequest: () => null,
};

export function requireRoles(roles: readonly StaffRole[]) {
  return async (
    request: IncomingMessage,
    response: ServerResponse,
    context: RequestContext,
    hooks: AuthorizationHooks = allowAnonymousAuthorization,
  ): Promise<AuthorizedPrincipal | null> => {
    const principal = await hooks.authorizeRequest(request, context);

    if (!principal) {
      response.writeHead(401, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ error: 'unauthorized' }));
      return null;
    }

    const hasRole = roles.some((role) => principal.roles.includes(role));
    if (!hasRole) {
      response.writeHead(403, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ error: 'forbidden' }));
      return null;
    }

    return principal;
  };
}
