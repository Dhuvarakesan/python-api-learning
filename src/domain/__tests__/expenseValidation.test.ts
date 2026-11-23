import { describe, expect, it } from 'vitest';

import {
  MAX_EXPENSE_DESCRIPTION_LENGTH,
  validateExpenseAmount,
  validateExpenseInput,
} from '../expenseValidation';
import type { ExpenseFormValues } from '../types';

const validExpenseValues: ExpenseFormValues = {
  amount: '12.50',
  date: '2026-11-23',
  category: 'Food',
  description: 'Lunch',
};

describe('expense validation', () => {
  it('requires a positive amount with up to two decimal places', () => {
    expect(validateExpenseAmount('')).toBe('Enter an amount.');
    expect(validateExpenseAmount('0')).toBe('Amount must be greater than zero.');
    expect(validateExpenseAmount('-4')).toBe(
      'Enter a positive amount with up to two decimal places.',
    );
    expect(validateExpenseAmount('10.999')).toBe(
      'Enter a positive amount with up to two decimal places.',
    );
    expect(validateExpenseAmount(' 10.50 ')).toBeNull();
  });

  it('normalizes valid input and trims optional description text', () => {
    const result = validateExpenseInput({
      ...validExpenseValues,
      amount: ' 99.9 ',
      description: '  Groceries and snacks  ',
    });

    expect(result).toEqual({
      isValid: true,
      value: {
        amount: 99.9,
        date: '2026-11-23',
        category: 'Food',
        description: 'Groceries and snacks',
      },
      errors: {},
    });
  });

  it('omits an empty description after trimming', () => {
    const result = validateExpenseInput({
      ...validExpenseValues,
      description: '   ',
    });

    expect(result.isValid).toBe(true);
    expect(result.value).toEqual({
      amount: 12.5,
      date: '2026-11-23',
      category: 'Food',
    });
  });

  it('reports date and category validation errors', () => {
    expect(
      validateExpenseInput({
        ...validExpenseValues,
        date: '',
        category: '',
      }).errors,
    ).toEqual({
      date: 'Choose a date.',
      category: 'Choose a category.',
    });

    expect(
      validateExpenseInput({
        ...validExpenseValues,
        date: '23-11-2026' as ExpenseFormValues['date'],
        category: 'Travel' as ExpenseFormValues['category'],
      }).errors,
    ).toEqual({
      date: 'Enter a date in YYYY-MM-DD format.',
      category: 'Choose a valid category.',
    });
  });

  it('rejects descriptions over the maximum length', () => {
    const result = validateExpenseInput({
      ...validExpenseValues,
      description: 'x'.repeat(MAX_EXPENSE_DESCRIPTION_LENGTH + 1),
    });

    expect(result).toEqual({
      isValid: false,
      value: null,
      errors: {
        description: `Description must be ${MAX_EXPENSE_DESCRIPTION_LENGTH} characters or less.`,
      },
    });
  });
});
