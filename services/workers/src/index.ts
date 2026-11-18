import { createServiceConfig } from '@clinic/config';
import type { AppointmentRequest } from '@clinic/shared';

export interface WorkerJob {
  readonly id: string;
  readonly kind: 'appointment-reminder' | 'slot-release';
  readonly appointmentRequest: AppointmentRequest;
}

export interface WorkerService {
  readonly name: 'clinic-workers';
  readonly enqueue: (job: WorkerJob) => Promise<string>;
}

export function createWorkerService(
  env: Record<string, string | undefined> = process.env,
): WorkerService {
  const config = createServiceConfig(env);

  return {
    name: 'clinic-workers',
    enqueue: async (job) => `${config.mode}:${job.kind}:${job.id}`,
  };
}
