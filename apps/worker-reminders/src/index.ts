export interface ReminderJob {
  appointmentId: string;
  scheduledFor: string;
}

export function createReminderJob(
  appointmentId: string,
  scheduledFor: string,
): ReminderJob {
  return {
    appointmentId,
    scheduledFor,
  };
}
