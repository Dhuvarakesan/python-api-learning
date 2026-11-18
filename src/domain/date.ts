export type DateString = `${number}-${number}-${number}`;

export type MonthString = `${number}-${number}`;

export const DATE_FORMAT_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

export const MONTH_FORMAT_PATTERN = /^\d{4}-\d{2}$/;

const padDatePart = (value: number) => value.toString().padStart(2, '0');

export const toDateString = (date: Date): DateString => {
  const year = date.getFullYear();
  const month = padDatePart(date.getMonth() + 1);
  const day = padDatePart(date.getDate());

  return `${year}-${month}-${day}` as DateString;
};

export const toMonthString = (date: Date): MonthString => {
  const year = date.getFullYear();
  const month = padDatePart(date.getMonth() + 1);

  return `${year}-${month}` as MonthString;
};

export const getTodayDateString = () => toDateString(new Date());

export const getCurrentMonthString = () => toMonthString(new Date());

export const isDateString = (value: string): value is DateString =>
  DATE_FORMAT_PATTERN.test(value);

export const isMonthString = (value: string): value is MonthString =>
  MONTH_FORMAT_PATTERN.test(value);
