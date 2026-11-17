import type { EnvironmentName } from "../contracts/integration-contracts";

export type ConfigValue = string | number | boolean;

export interface SecretReference {
  provider: "managed-vault" | "cloud-secret-manager" | "local-dev";
  name: string;
  version?: string;
}

export interface SharedServiceConfig {
  environment: EnvironmentName;
  serviceName: string;
  ehrBaseUrl?: string;
  schedulingSaasBaseUrl?: string;
  smsVendorBaseUrl?: string;
  emailVendorBaseUrl?: string;
  observabilityEndpoint?: string;
  secretRefs: {
    ehrClientCredential?: SecretReference;
    schedulingSaasCredential?: SecretReference;
    smsVendorCredential?: SecretReference;
    emailVendorCredential?: SecretReference;
    observabilityCredential?: SecretReference;
  };
  values: Record<string, ConfigValue>;
}

export interface EnvironmentSource {
  [name: string]: string | undefined;
}

const validEnvironments: EnvironmentName[] = ["local", "test", "staging", "production"];

export function loadSharedServiceConfig(env: EnvironmentSource): SharedServiceConfig {
  const environment = parseEnvironment(env.APP_ENV);
  const serviceName = requireValue(env.SERVICE_NAME, "SERVICE_NAME");

  return {
    environment,
    serviceName,
    ehrBaseUrl: env.EHR_BASE_URL,
    schedulingSaasBaseUrl: env.SCHEDULING_SAAS_BASE_URL,
    smsVendorBaseUrl: env.SMS_VENDOR_BASE_URL,
    emailVendorBaseUrl: env.EMAIL_VENDOR_BASE_URL,
    observabilityEndpoint: env.OBSERVABILITY_ENDPOINT,
    secretRefs: {
      ehrClientCredential: readSecretReference(env.EHR_CLIENT_CREDENTIAL_SECRET_REF),
      schedulingSaasCredential: readSecretReference(env.SCHEDULING_SAAS_CREDENTIAL_SECRET_REF),
      smsVendorCredential: readSecretReference(env.SMS_VENDOR_CREDENTIAL_SECRET_REF),
      emailVendorCredential: readSecretReference(env.EMAIL_VENDOR_CREDENTIAL_SECRET_REF),
      observabilityCredential: readSecretReference(env.OBSERVABILITY_CREDENTIAL_SECRET_REF),
    },
    values: {
      auditRetentionYears: 6,
      operationalPhiRetentionDays: 90,
      requireIdempotencyForMutations: true,
      redactPhiLogs: true,
    },
  };
}

export function readSecretReference(value: string | undefined): SecretReference | undefined {
  if (!value) {
    return undefined;
  }

  const [provider, name, version] = value.split(":");
  if (!provider || !name) {
    throw new Error("Secret references must use provider:name or provider:name:version format.");
  }

  if (!["managed-vault", "cloud-secret-manager", "local-dev"].includes(provider)) {
    throw new Error(`Unsupported secret provider '${provider}'.`);
  }

  return {
    provider: provider as SecretReference["provider"],
    name,
    version,
  };
}

function parseEnvironment(value: string | undefined): EnvironmentName {
  const normalized = value ?? "local";
  if (!validEnvironments.includes(normalized as EnvironmentName)) {
    throw new Error(`Unsupported APP_ENV '${normalized}'.`);
  }

  return normalized as EnvironmentName;
}

function requireValue(value: string | undefined, name: string): string {
  if (!value) {
    throw new Error(`${name} is required.`);
  }

  return value;
}
