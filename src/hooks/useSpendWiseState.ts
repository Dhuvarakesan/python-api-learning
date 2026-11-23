import { useCallback, useMemo, useState } from 'react';

import { getTodayDateString, type DateString } from '../domain/date';
import { validateExpenseInput } from '../domain/expenseValidation';
import type {
  Expense,
  ExpenseFormValues,
  ExpenseValidationErrors,
} from '../domain/types';

export type SpendWiseFeedback =
  | {
      type: 'success' | 'error';
      message: string;
    }
  | null;

export interface SpendWiseExpenseRepository {
  saveExpenses(expenses: Expense[]): void | Promise<void>;
}

export interface SpendWiseStateOptions {
  initialExpenses?: Expense[];
  repository?: SpendWiseExpenseRepository;
  idFactory?: () => string;
  nowFactory?: () => DateString;
}

export interface SpendWiseState {
  expenses: Expense[];
  editingExpenseId: string | null;
  editingExpense: Expense | null;
  feedback: SpendWiseFeedback;
  validationErrors: ExpenseValidationErrors;
  revision: number;
  addExpense(input: ExpenseFormValues): Expense | null;
  updateExpense(id: string, input: ExpenseFormValues): Expense | null;
  beginEdit(expenseId: string): boolean;
  cancelEdit(): void;
  clearFeedback(): void;
}

const createExpenseId = (): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }

  return `expense-${Date.now().toString(36)}`;
};

export const useSpendWiseState = (
  options: SpendWiseStateOptions = {},
): SpendWiseState => {
  const {
    initialExpenses = [],
    repository,
    idFactory = createExpenseId,
    nowFactory = getTodayDateString,
  } = options;
  const [expenses, setExpenses] = useState<Expense[]>(() => [
    ...initialExpenses,
  ]);
  const [editingExpenseId, setEditingExpenseId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<SpendWiseFeedback>(null);
  const [validationErrors, setValidationErrors] =
    useState<ExpenseValidationErrors>({});
  const [revision, setRevision] = useState(0);

  const persistExpenses = useCallback(
    (nextExpenses: Expense[]) => {
      if (!repository) {
        return;
      }

      try {
        const result = repository.saveExpenses(nextExpenses);
        if (result instanceof Promise) {
          result.catch(() => {
            setFeedback({
              type: 'error',
              message: 'Expense saved locally, but persistence failed.',
            });
          });
        }
      } catch {
        setFeedback({
          type: 'error',
          message: 'Expense saved locally, but persistence failed.',
        });
      }
    },
    [repository],
  );

  const commitExpenses = useCallback(
    (nextExpenses: Expense[]) => {
      setExpenses(nextExpenses);
      setRevision((currentRevision) => currentRevision + 1);
      persistExpenses(nextExpenses);
    },
    [persistExpenses],
  );

  const addExpense = useCallback(
    (input: ExpenseFormValues): Expense | null => {
      const validation = validateExpenseInput(input);

      if (!validation.isValid) {
        setValidationErrors(validation.errors);
        setFeedback({
          type: 'error',
          message: 'Review the expense details and try again.',
        });
        return null;
      }

      const timestamp = nowFactory();
      const expense: Expense = {
        id: idFactory(),
        ...validation.value,
        createdAt: timestamp,
        updatedAt: timestamp,
      };
      const nextExpenses = [expense, ...expenses];

      commitExpenses(nextExpenses);
      setValidationErrors({});
      setFeedback({ type: 'success', message: 'Expense added.' });

      return expense;
    },
    [commitExpenses, expenses, idFactory, nowFactory],
  );

  const updateExpense = useCallback(
    (id: string, input: ExpenseFormValues): Expense | null => {
      const existingExpense = expenses.find((expense) => expense.id === id);

      if (!existingExpense) {
        setFeedback({ type: 'error', message: 'Expense was not found.' });
        return null;
      }

      const validation = validateExpenseInput(input);

      if (!validation.isValid) {
        setValidationErrors(validation.errors);
        setFeedback({
          type: 'error',
          message: 'Review the expense details and try again.',
        });
        return null;
      }

      const updatedExpense: Expense = {
        ...existingExpense,
        ...validation.value,
        updatedAt: nowFactory(),
      };
      const nextExpenses = expenses.map((expense) =>
        expense.id === id ? updatedExpense : expense,
      );

      commitExpenses(nextExpenses);
      setEditingExpenseId(null);
      setValidationErrors({});
      setFeedback({ type: 'success', message: 'Expense updated.' });

      return updatedExpense;
    },
    [commitExpenses, expenses, nowFactory],
  );

  const beginEdit = useCallback(
    (expenseId: string): boolean => {
      const hasExpense = expenses.some((expense) => expense.id === expenseId);

      if (!hasExpense) {
        setFeedback({ type: 'error', message: 'Expense was not found.' });
        return false;
      }

      setEditingExpenseId(expenseId);
      setValidationErrors({});
      setFeedback(null);
      return true;
    },
    [expenses],
  );

  const cancelEdit = useCallback(() => {
    setEditingExpenseId(null);
    setValidationErrors({});
    setFeedback(null);
  }, []);

  const clearFeedback = useCallback(() => {
    setFeedback(null);
  }, []);

  const editingExpense = useMemo(
    () =>
      editingExpenseId
        ? expenses.find((expense) => expense.id === editingExpenseId) ?? null
        : null,
    [editingExpenseId, expenses],
  );

  return {
    expenses,
    editingExpenseId,
    editingExpense,
    feedback,
    validationErrors,
    revision,
    addExpense,
    updateExpense,
    beginEdit,
    cancelEdit,
    clearFeedback,
  };
};
