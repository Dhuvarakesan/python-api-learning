import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { pathToFileURL } from 'node:url';
import {
  allowAnonymousAuthorization,
  createRequestContext,
  createRequestLogger,
  type AuthorizationHooks,
  type RequestContext,
  type StructuredLogger,
} from './middleware/index.js';
import { registerAuditRoutes } from './modules/audit/index.js';
import { registerRequestRoutes } from './modules/requests/index.js';
import { registerSchedulingIntegrationRoutes } from './modules/scheduling-integration/index.js';
import { registerVerificationRoutes } from './modules/verification/index.js';

export interface ApiResponse {
  readonly statusCode?: number;
  readonly headers?: Readonly<Record<string, string>>;
  readonly body?: unknown;
}

export interface RouteContext {
  readonly request: IncomingMessage;
  readonly response: ServerResponse;
  readonly requestContext: RequestContext;
  readonly authorization: AuthorizationHooks;
}

export type RouteHandler = (
  context: RouteContext,
) => ApiResponse | Promise<ApiResponse>;

export interface RouteRegistry {
  readonly get: (path: string, handler: RouteHandler) => void;
  readonly post: (path: string, handler: RouteHandler) => void;
  readonly put: (path: string, handler: RouteHandler) => void;
  readonly patch: (path: string, handler: RouteHandler) => void;
  readonly delete: (path: string, handler: RouteHandler) => void;
}

export interface ApiServerOptions {
  readonly logger?: StructuredLogger;
  readonly authorization?: AuthorizationHooks;
}

type HttpMethod = 'DELETE' | 'GET' | 'PATCH' | 'POST' | 'PUT';

type RegisteredRoute = Readonly<{
  method: HttpMethod;
  path: string;
  handler: RouteHandler;
}>;

export function createApiServer(options: ApiServerOptions = {}) {
  const routes: RegisteredRoute[] = [];
  const register = createRouteRegistry(routes);

  register.get('/health', () => ({
    body: {
      ok: true,
      service: 'clinic-api',
    },
  }));

  registerAuditRoutes(register);
  registerVerificationRoutes(register);
  registerRequestRoutes(register);
  registerSchedulingIntegrationRoutes(register);

  const requestLogger = createRequestLogger(options.logger);
  const authorization = options.authorization ?? allowAnonymousAuthorization;

  return createServer(async (request, response) => {
    const requestContext = createRequestContext(request, response);
    requestLogger(request, response, requestContext);

    const route = findRoute(routes, request.method, request.url);
    if (!route) {
      writeJsonResponse(response, {
        statusCode: 404,
        body: { error: 'not_found' },
      });
      return;
    }

    try {
      const apiResponse = await route.handler({
        request,
        response,
        requestContext,
        authorization,
      });

      if (!response.writableEnded) {
        writeJsonResponse(response, apiResponse);
      }
    } catch (error) {
      if (!response.writableEnded) {
        writeJsonResponse(response, {
          statusCode: 500,
          body: { error: 'internal_server_error' },
        });
      }
    }
  });
}

export function startApiServer(options: ApiServerOptions = {}) {
  const server = createApiServer(options);
  const port = Number.parseInt(process.env.PORT ?? '3000', 10);
  const host = process.env.HOST ?? '0.0.0.0';

  server.listen(port, host, () => {
    console.info(
      JSON.stringify({
        level: 'info',
        message: 'api_server_started',
        service: 'clinic-api',
        host,
        port,
      }),
    );
  });

  return server;
}

function createRouteRegistry(routes: RegisteredRoute[]): RouteRegistry {
  const addRoute = (
    method: HttpMethod,
    path: string,
    handler: RouteHandler,
  ): void => {
    routes.push({ method, path, handler });
  };

  return {
    get: (path, handler) => addRoute('GET', path, handler),
    post: (path, handler) => addRoute('POST', path, handler),
    put: (path, handler) => addRoute('PUT', path, handler),
    patch: (path, handler) => addRoute('PATCH', path, handler),
    delete: (path, handler) => addRoute('DELETE', path, handler),
  };
}

function findRoute(
  routes: readonly RegisteredRoute[],
  method: string | undefined,
  url: string | undefined,
): RegisteredRoute | undefined {
  if (!method || !url) {
    return undefined;
  }

  const pathname = new URL(url, 'http://api.local').pathname;
  return routes.find((route) => route.method === method && route.path === pathname);
}

function writeJsonResponse(
  response: ServerResponse,
  apiResponse: ApiResponse,
): void {
  const statusCode = apiResponse.statusCode ?? 200;
  const headers = {
    'content-type': 'application/json',
    ...apiResponse.headers,
  };

  response.writeHead(statusCode, headers);
  response.end(JSON.stringify(apiResponse.body ?? {}));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  startApiServer();
}
