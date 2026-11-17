import type { SecretReference } from "../config/environment";

export interface SecretManager {
  getSecret(reference: SecretReference): Promise<string>;
}

export class UnconfiguredSecretManager implements SecretManager {
  async getSecret(reference: SecretReference): Promise<string> {
    throw new Error(
      `Secret manager is not configured for ${reference.provider}:${reference.name}. Use a managed vault or cloud secret manager adapter.`,
    );
  }
}

export function assertNoInlineSecret(name: string, value: string | undefined): void {
  if (!value) {
    return;
  }

  if (looksLikeSecret(value)) {
    throw new Error(`${name} appears to contain an inline secret. Use a managed secret reference instead.`);
  }
}

export function looksLikeSecret(value: string): boolean {
  const normalized = value.trim();
  return (
    normalized.length >= 24 &&
    (/^(sk|pk|ghp|eyJ|AKIA|ASIA)[A-Za-z0-9_-]/.test(normalized) ||
      /token|secret|password|credential/i.test(normalized))
  );
}
