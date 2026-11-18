export type PatientAccessMethod = 'magic_link' | 'one_time_code';

export interface PatientAccessChallenge {
  id: string;
  patientProfileId: string;
  method: PatientAccessMethod;
  secretHash: string;
  expiresAt: string;
  usedAt: string | null;
  createdAt: string;
}

export interface PatientAccessChallengeStore {
  findActiveChallenge(input: {
    patientProfileId: string;
    method: PatientAccessMethod;
  }): Promise<PatientAccessChallenge | null>;
  markUsed(challengeId: string, usedAt: string): Promise<void>;
}

export interface PatientSecretHasher {
  hash(secret: string): Promise<string> | string;
}

export type PatientAccessValidation =
  | { valid: true; patientProfileId: string; challengeId: string }
  | {
      valid: false;
      reason: 'missing_secret' | 'not_found' | 'expired' | 'already_used' | 'secret_mismatch';
    };

export async function validatePatientAccess(
  input: {
    patientProfileId: string;
    method: PatientAccessMethod;
    secret: string;
    now?: Date;
  },
  store: PatientAccessChallengeStore,
  hasher: PatientSecretHasher,
): Promise<PatientAccessValidation> {
  const secret = input.secret.trim();

  if (!secret) {
    return { valid: false, reason: 'missing_secret' };
  }

  const challenge = await store.findActiveChallenge({
    patientProfileId: input.patientProfileId,
    method: input.method,
  });

  if (!challenge) {
    return { valid: false, reason: 'not_found' };
  }

  if (challenge.usedAt) {
    return { valid: false, reason: 'already_used' };
  }

  const now = input.now ?? new Date();

  if (Date.parse(challenge.expiresAt) <= now.getTime()) {
    return { valid: false, reason: 'expired' };
  }

  const secretHash = await hasher.hash(secret);

  if (!timingSafeEqual(secretHash, challenge.secretHash)) {
    return { valid: false, reason: 'secret_mismatch' };
  }

  await store.markUsed(challenge.id, now.toISOString());

  return {
    valid: true,
    patientProfileId: challenge.patientProfileId,
    challengeId: challenge.id,
  };
}

export function getPatientAccessExpiry(createdAt: Date, ttlMinutes: number): string {
  return new Date(createdAt.getTime() + ttlMinutes * 60_000).toISOString();
}

function timingSafeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) {
    return false;
  }

  let mismatch = 0;

  for (let index = 0; index < left.length; index += 1) {
    mismatch |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }

  return mismatch === 0;
}
