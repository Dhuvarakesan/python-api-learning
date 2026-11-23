import { isExpenseCategory } from './categories';
import { isDateString } from './date';
import type {
  ExpenseFormValues,
  ExpenseValidationErrors,
  ExpenseValidationResult,
  NormalizedExpenseInput,
} from './types';

export const MAX_EXPENSE_DESCRIPTION_LENGTH = 120;

const AMOUNT_PATTERN = /^\d+(?:\.\d{1,2})?$/;

export const normalizeExpenseAmount = (amount: string): number =>
  Number(Number(amount).toFixed(2));

export const validateExpenseAmount = (amount: string): string | null => {
  const normalizedAmount = amount.trim();

  if (normalizedAmount.length === 0) {
    return 'Enter an amount.';
  }

  if (!AMOUNT_PATTERN.test(normalizedAmount)) {
    return 'Enter a positive amount with up to two decimal places.';
  }

  if (normalizeExpenseAmount(normalizedAmount) <= 0) {
    return 'Amount must be greater than zero.';
  }

  return null;
};

export const normalizeExpenseInput = (
  values: ExpenseFormValues,
): NormalizedExpenseInput => {
  const amountError = validateExpenseAmount(values.amount);

  if (amountError) {
    throw new Error(amountError);
  }

  if (!isDateString(values.date)) {
    throw new Error('Enter a date in YYYY-MM-DD format.');
  }

  if (!isExpenseCategory(values.category)) {
    throw new Error('Choose a valid category.');
  }

  const description = values.description.trim();

  return {
    amount: normalizeExpenseAmount(values.amount.trim()),
    date: values.date,
    category: values.category,
    ...(description.length > 0 ? { description } : {}),
  };
};

export const validateExpenseInput = (
  values: ExpenseFormValues,
): ExpenseValidationResult => {
  const errors: ExpenseValidationErrors = {};
  const amountError = validateExpenseAmount(values.amount);
  const description = values.description.trim();

  if (amountError) {
    errors.amount = amountError;
  }

  if (!values.date) {
    errors.date = 'Choose a date.';
  } else if (!isDateString(values.date)) {
    errors.date = 'Enter a date in YYYY-MM-DD format.';
  }

  if (!values.category) {
    errors.category = 'Choose a category.';
  } else if (!isExpenseCategory(values.category)) {
    errors.category = 'Choose a valid category.';
  }

  if (description.length > MAX_EXPENSE_DESCRIPTION_LENGTH) {
    errors.description = `Description must be ${MAX_EXPENSE_DESCRIPTION_LENGTH} characters or less.`;
  }

  if (Object.keys(errors).length > 0) {
    return {
      isValid: false,
      value: null,
      errors,
    };
  }

  return {
    isValid: true,
    value: normalizeExpenseInput(values),
    errors,
  };
};
