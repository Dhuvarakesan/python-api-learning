export type AppointmentChannel = "in-person" | "telehealth";

export interface AppointmentSlot {
  id: string;
  clinicianId: string;
  startsAt: string;
  endsAt: string;
  channel: AppointmentChannel;
}
