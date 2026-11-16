export interface SchedulingSystemAppointment {
  externalId: string;
  startsAt: string;
  endsAt: string;
}

export interface SchedulingSystemClient {
  listAppointments(clinicianId: string): Promise<SchedulingSystemAppointment[]>;
}
