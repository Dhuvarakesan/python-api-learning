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

export interface ExpenseFormValues {
  amount: string;
  date: DateString | '';
  category: ExpenseCategory | '';
  description: string;
}

export interface NormalizedExpenseInput {
  amount: number;
  date: DateString;
  category: ExpenseCategory;
  description?: string;
}

export type ExpenseFormField = keyof ExpenseFormValues;

export type ExpenseValidationErrors = Partial<
  Record<ExpenseFormField, string>
>;

export type ExpenseValidationResult =
  | {
      isValid: true;
      value: NormalizedExpenseInput;
      errors: ExpenseValidationErrors;
    }
  | {
      isValid: false;
      value: null;
      errors: ExpenseValidationErrors;
    };
