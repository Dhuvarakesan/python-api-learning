import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { Expense, ExpenseFormValues } from '../domain/types';
import { useSpendWiseState } from './useSpendWiseState';

const validExpenseInput: ExpenseFormValues = {
  amount: '12.50',
  date: '2026-11-22',
  category: 'Food',
  description: 'Lunch',
};

const createExpense = (overrides: Partial<Expense>): Expense => ({
  id: 'expense-1',
  amount: 10,
  date: '2026-11-20',
  category: 'Food',
  description: 'Original',
  createdAt: '2026-11-20',
  updatedAt: '2026-11-20',
  ...overrides,
});

describe('useSpendWiseState', () => {
  it('adds a valid expense with stable metadata and success feedback', () => {
    const saveExpenses = vi.fn();
    const { result } = renderHook(() =>
      useSpendWiseState({
        repository: { saveExpenses },
        idFactory: () => 'expense-added',
        nowFactory: () => '2026-11-23',
      }),
    );

    let addedExpense: Expense | null = null;

    act(() => {
      addedExpense = result.current.addExpense({
        ...validExpenseInput,
        description: '  Lunch  ',
      });
    });

    expect(addedExpense).toEqual({
      id: 'expense-added',
      amount: 12.5,
      date: '2026-11-22',
      category: 'Food',
      description: 'Lunch',
      createdAt: '2026-11-23',
      updatedAt: '2026-11-23',
    });
    expect(result.current.expenses).toEqual([addedExpense]);
    expect(result.current.feedback).toEqual({
      type: 'success',
      message: 'Expense added.',
    });
    expect(result.current.validationErrors).toEqual({});
    expect(result.current.revision).toBe(1);
    expect(saveExpenses).toHaveBeenCalledWith([addedExpense]);
  });

  it('updates only the targeted expense after valid input', () => {
    const untouchedExpense = createExpense({
      id: 'expense-a',
      amount: 25,
      category: 'Bills',
    });
    const targetExpense = createExpense({
      id: 'expense-b',
      amount: 7,
      category: 'Transport',
      createdAt: '2026-11-19',
      updatedAt: '2026-11-19',
    });
    const { result } = renderHook(() =>
      useSpendWiseState({
        initialExpenses: [untouchedExpense, targetExpense],
        nowFactory: () => '2026-11-23',
      }),
    );

    let updatedExpense: Expense | null = null;

    act(() => {
      updatedExpense = result.current.updateExpense('expense-b', {
        amount: '18.75',
        date: '2026-11-21',
        category: 'Health',
        description: 'Pharmacy',
      });
    });

    expect(result.current.expenses[0]).toBe(untouchedExpense);
    expect(result.current.expenses[1]).toEqual({
      id: 'expense-b',
      amount: 18.75,
      date: '2026-11-21',
      category: 'Health',
      description: 'Pharmacy',
      createdAt: '2026-11-19',
      updatedAt: '2026-11-23',
    });
    expect(result.current.expenses[1]).toEqual(updatedExpense);
    expect(result.current.feedback).toEqual({
      type: 'success',
      message: 'Expense updated.',
    });
    expect(result.current.revision).toBe(1);
  });

  it('leaves the original expense unchanged when edit is canceled', () => {
    const originalExpense = createExpense({
      id: 'expense-c',
      amount: 42,
      description: 'Keep me',
    });
    const { result } = renderHook(() =>
      useSpendWiseState({ initialExpenses: [originalExpense] }),
    );

    act(() => {
      expect(result.current.beginEdit('expense-c')).toBe(true);
    });

    expect(result.current.editingExpense).toEqual(originalExpense);

    act(() => {
      result.current.cancelEdit();
    });

    expect(result.current.editingExpenseId).toBeNull();
    expect(result.current.editingExpense).toBeNull();
    expect(result.current.expenses).toEqual([originalExpense]);
    expect(result.current.expenses[0]).toBe(originalExpense);
    expect(result.current.revision).toBe(0);
  });
});
