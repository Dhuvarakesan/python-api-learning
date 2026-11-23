import { EXPENSE_CATEGORIES } from '../../domain/categories';
import { MAX_EXPENSE_DESCRIPTION_LENGTH } from '../../domain/expenseValidation';
import type {
  ExpenseFormField,
  ExpenseFormValues,
  ExpenseValidationErrors,
} from '../../domain/types';

interface ExpenseFormFieldsProps {
  values: ExpenseFormValues;
  errors: ExpenseValidationErrors;
  onFieldChange<Field extends ExpenseFormField>(
    field: Field,
    value: ExpenseFormValues[Field],
  ): void;
}

const describedBy = (fieldId: string, hasError: boolean) =>
  hasError ? `${fieldId}-error` : undefined;

export function ExpenseFormFields({
  values,
  errors,
  onFieldChange,
}: ExpenseFormFieldsProps) {
  return (
    <div className="expense-form-fields">
      <div className="field-group">
        <label htmlFor="expense-amount">Amount (₹)</label>
        <input
          aria-describedby={describedBy('expense-amount', Boolean(errors.amount))}
          aria-invalid={Boolean(errors.amount)}
          autoComplete="off"
          id="expense-amount"
          inputMode="decimal"
          name="amount"
          onChange={(event) => onFieldChange('amount', event.target.value)}
          placeholder="0.00"
          type="text"
          value={values.amount}
        />
        {errors.amount ? (
          <p className="field-error" id="expense-amount-error">
            {errors.amount}
          </p>
        ) : null}
      </div>

      <div className="field-group">
        <label htmlFor="expense-date">Date</label>
        <input
          aria-describedby={describedBy('expense-date', Boolean(errors.date))}
          aria-invalid={Boolean(errors.date)}
          id="expense-date"
          name="date"
          onChange={(event) =>
            onFieldChange(
              'date',
              event.target.value as ExpenseFormValues['date'],
            )
          }
          type="date"
          value={values.date}
        />
        {errors.date ? (
          <p className="field-error" id="expense-date-error">
            {errors.date}
          </p>
        ) : null}
      </div>

      <div className="field-group">
        <label htmlFor="expense-category">Category</label>
        <select
          aria-describedby={describedBy(
            'expense-category',
            Boolean(errors.category),
          )}
          aria-invalid={Boolean(errors.category)}
          id="expense-category"
          name="category"
          onChange={(event) =>
            onFieldChange(
              'category',
              event.target.value as ExpenseFormValues['category'],
            )
          }
          value={values.category}
        >
          <option value="">Choose category</option>
          {EXPENSE_CATEGORIES.map((category) => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>
        {errors.category ? (
          <p className="field-error" id="expense-category-error">
            {errors.category}
          </p>
        ) : null}
      </div>

      <div className="field-group field-group--wide">
        <label htmlFor="expense-description">Description</label>
        <textarea
          aria-describedby={describedBy(
            'expense-description',
            Boolean(errors.description),
          )}
          aria-invalid={Boolean(errors.description)}
          id="expense-description"
          maxLength={MAX_EXPENSE_DESCRIPTION_LENGTH}
          name="description"
          onChange={(event) =>
            onFieldChange('description', event.target.value)
          }
          rows={3}
          value={values.description}
        />
        <div className="field-footer">
          {errors.description ? (
            <p className="field-error" id="expense-description-error">
              {errors.description}
            </p>
          ) : (
            <span />
          )}
          <span aria-live="polite">
            {values.description.length}/{MAX_EXPENSE_DESCRIPTION_LENGTH}
          </span>
        </div>
      </div>
    </div>
  );
}
