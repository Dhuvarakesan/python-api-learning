import { useEffect, useRef, type FormEvent, type KeyboardEvent } from 'react';

import { Button } from '../../components/ui/Button';
import type {
  ExpenseFormField,
  ExpenseFormValues,
  ExpenseValidationErrors,
} from '../../domain/types';
import { ExpenseFormFields } from './ExpenseFormFields';

interface ExpenseDialogProps {
  open: boolean;
  mode: 'add' | 'edit';
  values: ExpenseFormValues;
  errors: ExpenseValidationErrors;
  onCancel(): void;
  onSubmit(): void;
  onFieldChange<Field extends ExpenseFormField>(
    field: Field,
    value: ExpenseFormValues[Field],
  ): void;
}

const focusableSelector = [
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'a[href]',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

export function ExpenseDialog({
  open,
  mode,
  values,
  errors,
  onCancel,
  onSubmit,
  onFieldChange,
}: ExpenseDialogProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const titleId = 'expense-dialog-title';

  useEffect(() => {
    if (!open) {
      return;
    }

    const firstInput = dialogRef.current?.querySelector<HTMLInputElement>(
      '#expense-amount',
    );
    firstInput?.focus();
  }, [open]);

  if (!open) {
    return null;
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onCancel();
      return;
    }

    if (event.key !== 'Tab' || !dialogRef.current) {
      return;
    }

    const focusableElements = Array.from(
      dialogRef.current.querySelectorAll<HTMLElement>(focusableSelector),
    );

    if (focusableElements.length === 0) {
      return;
    }

    const firstElement = focusableElements[0]!;
    const lastElement = focusableElements[focusableElements.length - 1]!;

    if (event.shiftKey && document.activeElement === firstElement) {
      event.preventDefault();
      lastElement.focus();
    } else if (!event.shiftKey && document.activeElement === lastElement) {
      event.preventDefault();
      firstElement.focus();
    }
  };

  return (
    <div
      aria-labelledby={titleId}
      aria-modal="true"
      className="dialog-backdrop"
      onKeyDown={handleKeyDown}
      role="dialog"
    >
      <div className="expense-dialog" ref={dialogRef}>
        <form className="expense-dialog__form" onSubmit={handleSubmit}>
          <div className="expense-dialog__header">
            <div>
              <p className="section-eyebrow">Expense entry</p>
              <h2 id={titleId}>
                {mode === 'edit' ? 'Edit expense' : 'Add expense'}
              </h2>
            </div>
          </div>

          <ExpenseFormFields
            errors={errors}
            onFieldChange={onFieldChange}
            values={values}
          />

          <div className="expense-dialog__actions">
            <Button onClick={onCancel} variant="secondary">
              Cancel
            </Button>
            <Button type="submit" variant="primary">
              Save expense
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
