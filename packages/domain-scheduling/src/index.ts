export interface AvailabilityWindow {
  clinicianId: string;
  startsAt: string;
  endsAt: string;
}

export function isAvailabilityWindowValid(window: AvailabilityWindow): boolean {
  return Date.parse(window.startsAt) < Date.parse(window.endsAt);
}
