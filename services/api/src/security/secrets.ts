export type SecretSource = 'environment' | 'managed-reference';

export interface SecretReference {
  readonly source: SecretSource;
  readonly envName: string;
  readonly reference?: string;
}

export interface ManagedSecretResolver {
  readonly resolve: (reference: string) => Promise<string> | string;
}

export type EnvSource = Record<string, string | undefined>;

export function loadSecretReference(
  env: EnvSource,
  envName: string,
): SecretReference | undefined {
  const managedReference = env[`${envName}_REF`] ?? env[`${envName}_SECRET_REF`];

  if (managedReference?.trim()) {
    return {
      source: 'managed-reference',
      envName,
      reference: managedReference.trim(),
    };
  }

  if (env[envName]?.trim()) {
    return {
      source: 'environment',
      envName,
    };
  }

  return undefined;
}

export function requireSecretReference(
  env: EnvSource,
  envName: string,
): SecretReference {
  const secret = loadSecretReference(env, envName);

  if (!secret) {
    throw new Error(`Missing required secret: ${envName}`);
  }

  return secret;
}

export async function resolveSecretValue(
  env: EnvSource,
  secret: SecretReference,
  resolver?: ManagedSecretResolver,
): Promise<string> {
  if (secret.source === 'environment') {
    const value = env[secret.envName];
    if (!value) {
      throw new Error(`Missing required secret value: ${secret.envName}`);
    }

    return value;
  }

  if (!secret.reference) {
    throw new Error(`Missing managed secret reference: ${secret.envName}`);
  }

  if (!resolver) {
    throw new Error(`No managed secret resolver configured: ${secret.envName}`);
  }

  return resolver.resolve(secret.reference);
}

export function describeSecret(secret: SecretReference): Record<string, string> {
  return {
    envName: secret.envName,
    source: secret.source,
  };
}
