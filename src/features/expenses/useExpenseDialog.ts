import { useCallback, useMemo, useRef, useState } from 'react';

import { createExpenseFormValues } from '../../domain/expenseFactory';
import type { Expense, ExpenseFormValues } from '../../domain/types';

type ExpenseDialogMode = 'add' | 'edit';

interface ExpenseDialogState {
  mode: ExpenseDialogMode;
  expense: Expense | null;
}

export interface ExpenseDialogController {
  isOpen: boolean;
  mode: ExpenseDialogMode;
  editingExpense: Expense | null;
  values: ExpenseFormValues;
  openAddDialog(opener?: HTMLElement | null): void;
  openEditDialog(expense: Expense, opener?: HTMLElement | null): void;
  closeDialog(): void;
  updateField<Field extends keyof ExpenseFormValues>(
    field: Field,
    value: ExpenseFormValues[Field],
  ): void;
}

export const useExpenseDialog = (): ExpenseDialogController => {
  const [dialogState, setDialogState] = useState<ExpenseDialogState | null>(
    null,
  );
  const [values, setValues] = useState<ExpenseFormValues>(() =>
    createExpenseFormValues(),
  );
  const restoreFocusElementRef = useRef<HTMLElement | null>(null);

  const restoreFocus = useCallback(() => {
    restoreFocusElementRef.current?.focus();
    restoreFocusElementRef.current = null;
  }, []);

  const openAddDialog = useCallback((opener?: HTMLElement | null) => {
    restoreFocusElementRef.current = opener ?? null;
    setValues(createExpenseFormValues());
    setDialogState({ mode: 'add', expense: null });
  }, []);

  const openEditDialog = useCallback(
    (expense: Expense, opener?: HTMLElement | null) => {
      restoreFocusElementRef.current = opener ?? null;
      setValues(createExpenseFormValues(expense));
      setDialogState({ mode: 'edit', expense });
    },
    [],
  );

  const closeDialog = useCallback(() => {
    setDialogState(null);
    setValues(createExpenseFormValues());
    window.requestAnimationFrame(restoreFocus);
  }, [restoreFocus]);

  const updateField = useCallback(
    <Field extends keyof ExpenseFormValues>(
      field: Field,
      value: ExpenseFormValues[Field],
    ) => {
      setValues((currentValues) => ({
        ...currentValues,
        [field]: field === 'description' ? value.slice(0, 120) : value,
      }));
    },
    [],
  );

  return useMemo(
    () => ({
      isOpen: dialogState !== null,
      mode: dialogState?.mode ?? 'add',
      editingExpense: dialogState?.expense ?? null,
      values,
      openAddDialog,
      openEditDialog,
      closeDialog,
      updateField,
    }),
    [
      closeDialog,
      dialogState,
      openAddDialog,
      openEditDialog,
      updateField,
      values,
    ],
  );
};
