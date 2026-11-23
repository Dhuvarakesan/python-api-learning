import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useState } from 'react';

import { getTodayDateString } from '../../../domain/date';
import type { Expense, ExpenseFormValues } from '../../../domain/types';
import { validateExpenseInput } from '../../../domain/expenseValidation';
import { ExpenseDialog } from '../ExpenseDialog';
import { useExpenseDialog } from '../useExpenseDialog';

const editExpense: Expense = {
  id: 'expense-edit',
  amount: 42.5,
  date: '2026-11-20',
  category: 'Transport',
  description: 'Airport taxi',
  createdAt: '2026-11-20',
  updatedAt: '2026-11-21',
};

const requestAnimationFrameSpy = vi
  .spyOn(window, 'requestAnimationFrame')
  .mockImplementation((callback: FrameRequestCallback) => {
    callback(0);
    return 0;
  });

afterEach(() => {
  vi.useRealTimers();
  requestAnimationFrameSpy.mockClear();
});

function ExpenseDialogHarness({
  onCancel = vi.fn(),
  onSave = vi.fn(),
}: {
  onCancel?: () => void;
  onSave?: (values: ExpenseFormValues) => void;
}) {
  const dialog = useExpenseDialog();
  const [errors, setErrors] = useState({});

  const submitExpense = () => {
    const validation = validateExpenseInput(dialog.values);

    if (!validation.isValid) {
      setErrors(validation.errors);
      return;
    }

    setErrors({});
    onSave(dialog.values);
    dialog.closeDialog();
  };

  const cancelExpense = () => {
    setErrors({});
    onCancel();
    dialog.closeDialog();
  };

  return (
    <>
      <button
        onClick={(event) => dialog.openAddDialog(event.currentTarget)}
        type="button"
      >
        Open add
      </button>
      <button
        onClick={(event) =>
          dialog.openEditDialog(editExpense, event.currentTarget)
        }
        type="button"
      >
        Open edit
      </button>
      <ExpenseDialog
        errors={errors}
        mode={dialog.mode}
        onCancel={cancelExpense}
        onFieldChange={(field, value) => dialog.updateField(field, value)}
        onSubmit={submitExpense}
        open={dialog.isOpen}
        values={dialog.values}
      />
    </>
  );
}

describe('ExpenseDialog', () => {
  it('opens in add mode with default values and focus on amount', async () => {
    const user = userEvent.setup();

    render(<ExpenseDialogHarness />);

    await user.click(screen.getByRole('button', { name: 'Open add' }));

    expect(
      screen.getByRole('heading', { name: 'Add expense' }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Amount (₹)')).toHaveValue('');
    expect(screen.getByLabelText('Date')).toHaveValue(getTodayDateString());
    expect(screen.getByLabelText('Category')).toHaveValue('');
    expect(screen.getByLabelText('Description')).toHaveValue('');
    expect(screen.getByLabelText('Amount (₹)')).toHaveFocus();
  });

  it('opens in edit mode with the selected expense values prefilled', async () => {
    const user = userEvent.setup();

    render(<ExpenseDialogHarness />);

    await user.click(screen.getByRole('button', { name: 'Open edit' }));

    expect(
      screen.getByRole('heading', { name: 'Edit expense' }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Amount (₹)')).toHaveValue('42.50');
    expect(screen.getByLabelText('Date')).toHaveValue('2026-11-20');
    expect(screen.getByLabelText('Category')).toHaveValue('Transport');
    expect(screen.getByLabelText('Description')).toHaveValue('Airport taxi');
  });

  it('shows field messages on invalid submit', async () => {
    const user = userEvent.setup();

    render(<ExpenseDialogHarness />);

    await user.click(screen.getByRole('button', { name: 'Open add' }));
    await user.clear(screen.getByLabelText('Date'));
    await user.click(screen.getByRole('button', { name: 'Save expense' }));

    expect(screen.getByText('Enter an amount.')).toBeInTheDocument();
    expect(screen.getByText('Choose a date.')).toBeInTheDocument();
    expect(screen.getByText('Choose a category.')).toBeInTheDocument();
    expect(screen.getByLabelText('Amount (₹)')).toHaveAttribute(
      'aria-invalid',
      'true',
    );
  });

  it('saves valid add-mode values and closes the dialog', async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();

    render(<ExpenseDialogHarness onSave={onSave} />);

    await user.click(screen.getByRole('button', { name: 'Open add' }));
    await user.type(screen.getByLabelText('Amount (₹)'), '15.25');
    await user.clear(screen.getByLabelText('Date'));
    await user.type(screen.getByLabelText('Date'), '2026-11-22');
    await user.selectOptions(screen.getByLabelText('Category'), 'Food');
    await user.type(screen.getByLabelText('Description'), 'Lunch');
    await user.click(screen.getByRole('button', { name: 'Save expense' }));

    expect(onSave).toHaveBeenCalledWith({
      amount: '15.25',
      date: '2026-11-22',
      category: 'Food',
      description: 'Lunch',
    });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('cancels without saving and restores focus to the opener', async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    const onSave = vi.fn();

    render(<ExpenseDialogHarness onCancel={onCancel} onSave={onSave} />);

    const opener = screen.getByRole('button', { name: 'Open edit' });
    await user.click(opener);
    await user.clear(screen.getByLabelText('Amount (₹)'));
    await user.type(screen.getByLabelText('Amount (₹)'), '99.00');
    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onSave).not.toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await waitFor(() => expect(opener).toHaveFocus());
  });

  it('supports keyboard escape close and tab wrapping inside the dialog', async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();

    render(<ExpenseDialogHarness onCancel={onCancel} />);

    await user.click(screen.getByRole('button', { name: 'Open add' }));
    expect(screen.getByLabelText('Amount (₹)')).toHaveFocus();

    await user.keyboard('{Shift>}{Tab}{/Shift}');
    expect(screen.getByRole('button', { name: 'Save expense' })).toHaveFocus();

    await user.keyboard('{Tab}');
    expect(screen.getByLabelText('Amount (₹)')).toHaveFocus();

    await user.keyboard('{Escape}');
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('keeps user-provided markup-like descriptions as plain text', async () => {
    const user = userEvent.setup();
    const description = '<img src=x onerror=alert(1)>Coffee';

    render(<ExpenseDialogHarness />);

    await user.click(screen.getByRole('button', { name: 'Open add' }));
    await user.type(screen.getByLabelText('Description'), description);

    expect(screen.getByLabelText('Description')).toHaveValue(description);
    expect(document.querySelector('img')).toBeNull();
  });
});
