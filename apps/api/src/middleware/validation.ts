import type { IncomingMessage } from "node:http";
import type { ValidationIssue } from "@clinic/contracts";

export interface ValidationResult<TValue> {
  value?: TValue;
  issues: ValidationIssue[];
}

export type RequestValidator<TValue> = (value: unknown) => ValidationResult<TValue>;

export async function readJsonBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];

  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  if (chunks.length === 0) {
    return undefined;
  }

  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
}

export function requiredString(value: unknown, path: string): ValidationResult<string> {
  if (typeof value === "string" && value.trim().length > 0) {
    return {
      value,
      issues: [],
    };
  }

  return {
    issues: [
      {
        path,
        code: "REQUIRED_STRING",
        message: "Expected a non-empty string.",
      },
    ],
  };
}
