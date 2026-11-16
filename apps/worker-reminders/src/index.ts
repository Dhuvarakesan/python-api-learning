import {
  createCorrelationId,
  type SafeAttributes,
} from "@clinic/observability";

export interface ReminderJob {
  appointmentId: string;
  scheduledFor: string;
  correlationId: string;
}

export type ReminderJobInput = Omit<ReminderJob, "correlationId"> & {
  correlationId?: string;
};

export function createReminderJob(
  appointmentId: string,
  scheduledFor: string,
  correlationId?: string,
): ReminderJob {
  return {
    appointmentId,
    scheduledFor,
    correlationId: createCorrelationId(correlationId),
  };
}

export function ensureReminderJobCorrelationId(
  job: ReminderJobInput,
): ReminderJob {
  return {
    ...job,
    correlationId: createCorrelationId(job.correlationId),
  };
}

export function createReminderJobLogAttributes(
  job: ReminderJob,
): SafeAttributes {
  return {
    correlationId: job.correlationId,
    jobType: "appointment_reminder",
    appointmentId: job.appointmentId,
    scheduledFor: job.scheduledFor,
  };
}
