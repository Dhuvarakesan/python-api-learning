import { getTodayDateString } from './date';
import type { Expense, ExpenseFormValues } from './types';

export const createExpenseFormValues = (
  expense?: Expense,
): ExpenseFormValues => {
  if (expense) {
    return {
      amount: expense.amount.toFixed(2),
      date: expense.date,
      category: expense.category,
      description: expense.description ?? '',
    };
  }

  return {
    amount: '',
    date: getTodayDateString(),
    category: '',
    description: '',
  };
};
