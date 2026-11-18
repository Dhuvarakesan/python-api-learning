import type { ExpenseCategory } from './categories';
import type { DateString, MonthString } from './date';

export interface Expense {
  id: string;
  amount: number;
  date: DateString;
  category: ExpenseCategory;
  description?: string;
  createdAt: DateString;
  updatedAt: DateString;
}

export interface MonthlyBudget {
  month: MonthString;
  amount: number;
  createdAt: DateString;
  updatedAt: DateString;
}

export interface FilterState {
  month: MonthString;
  category: ExpenseCategory | null;
}
