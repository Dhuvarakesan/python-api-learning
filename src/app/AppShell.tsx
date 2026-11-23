import { useMemo, useState } from 'react';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Toast } from '../components/ui/Toast';
import { formatInrCurrency } from '../domain/currency';
import type { Expense } from '../domain/types';
import { ExpenseDialog } from '../features/expenses/ExpenseDialog';
import { useExpenseDialog } from '../features/expenses/useExpenseDialog';
import { useSpendWiseState } from '../hooks/useSpendWiseState';
import {
  defaultScreenId,
  primaryNavigation,
  type PrimaryNavigationItem,
  type PrimaryScreenId,
} from './navigation';

const screenStats: Record<PrimaryScreenId, string[]> = {
  'monthly-summary': ['This month', 'Category mix', 'Budget pace'],
  'expense-history': ['Recent records', 'Month filter', 'Category filter'],
  'monthly-budget': ['Monthly target', 'Remaining amount', 'Budget health'],
  'csv-export': ['CSV preview', 'Download action', 'Local file copy'],
};

function ScreenPreview({ item }: { item: PrimaryNavigationItem }) {
  return (
    <Card elevated aria-labelledby={`${item.id}-title`} className="screen-card">
      <p className="section-eyebrow">{item.eyebrow}</p>
      <div className="screen-card__header">
        <div>
          <h2 id={`${item.id}-title`}>{item.title}</h2>
          <p>{item.description}</p>
        </div>
        <span className="screen-card__status">Local only</span>
      </div>
      <div className="metric-grid" aria-label={`${item.label} preview`}>
        {screenStats[item.id].map((stat) => (
          <div className="metric" key={stat}>
            <span>{stat}</span>
            <strong>Ready</strong>
          </div>
        ))}
      </div>
    </Card>
  );
}

function ExpenseList({
  expenses,
  onEditExpense,
}: {
  expenses: Expense[];
  onEditExpense(expense: Expense, opener: HTMLElement): void;
}) {
  return (
    <Card className="expenses-panel" aria-labelledby="expenses-panel-title">
      <div className="expenses-panel__header">
        <div>
          <p className="section-eyebrow">Expense records</p>
          <h2 id="expenses-panel-title">Recent expenses</h2>
        </div>
        <span>{expenses.length} saved</span>
      </div>

      {expenses.length === 0 ? (
        <p className="expenses-empty">
          Added expenses will appear here for review and editing.
        </p>
      ) : (
        <ul className="expense-list" aria-label="Recent expenses">
          {expenses.map((expense) => (
            <li className="expense-list__item" key={expense.id}>
              <div>
                <strong>{formatInrCurrency(expense.amount)}</strong>
                <span>
                  {expense.category} · {expense.date}
                </span>
                {expense.description ? <p>{expense.description}</p> : null}
              </div>
              <Button
                aria-label={`Edit ${expense.category} expense from ${expense.date}`}
                onClick={(event) =>
                  onEditExpense(expense, event.currentTarget)
                }
                variant="secondary"
              >
                Edit
              </Button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function AppShell() {
  const [activeScreenId, setActiveScreenId] =
    useState<PrimaryScreenId>(defaultScreenId);
  const [toastMessage, setToastMessage] = useState('');
  const spendWiseState = useSpendWiseState();
  const expenseDialog = useExpenseDialog();

  const activeScreen = useMemo(
    () =>
      primaryNavigation.find((item) => item.id === activeScreenId) ??
      primaryNavigation.find((item) => item.id === defaultScreenId)!,
    [activeScreenId],
  );

  const openAddExpenseDialog = (opener: HTMLElement) => {
    spendWiseState.cancelEdit();
    setToastMessage('');
    expenseDialog.openAddDialog(opener);
  };

  const openEditExpenseDialog = (expense: Expense, opener: HTMLElement) => {
    if (!spendWiseState.beginEdit(expense.id)) {
      return;
    }

    setToastMessage('');
    expenseDialog.openEditDialog(expense, opener);
  };

  const closeExpenseDialog = () => {
    spendWiseState.cancelEdit();
    expenseDialog.closeDialog();
  };

  const saveExpense = () => {
    const savedExpense =
      expenseDialog.mode === 'edit' && expenseDialog.editingExpense
        ? spendWiseState.updateExpense(
            expenseDialog.editingExpense.id,
            expenseDialog.values,
          )
        : spendWiseState.addExpense(expenseDialog.values);

    if (savedExpense) {
      expenseDialog.closeDialog();
    }
  };

  const navigateTo = (screenId: PrimaryScreenId) => {
    setActiveScreenId(screenId);
    const screen = primaryNavigation.find((item) => item.id === screenId);
    if (screen) {
      setToastMessage(`${screen.label} opened.`);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="skip-link" href="#primary-content">
          Skip to content
        </a>
        <div className="brand" aria-label="SpendWise home">
          <span className="brand__mark" aria-hidden="true">
            S
          </span>
          <div>
            <p>SpendWise</p>
            <span>Personal expense tracker</span>
          </div>
        </div>
        <Button
          variant="primary"
          onClick={(event) => openAddExpenseDialog(event.currentTarget)}
        >
          Add expense
        </Button>
      </header>

      <div className="shell-layout">
        <aside className="sidebar" aria-label="Primary">
          <nav className="primary-nav" aria-label="SpendWise screens">
            {primaryNavigation.map((item) => (
              <button
                aria-current={item.id === activeScreenId ? 'page' : undefined}
                className="nav-item"
                key={item.id}
                onClick={() => navigateTo(item.id)}
                type="button"
              >
                <span>{item.label}</span>
                <small>{item.eyebrow}</small>
              </button>
            ))}
          </nav>
          <Card className="privacy-card">
            <p className="section-eyebrow">Privacy</p>
            <p>
              SpendWise keeps records in this browser only. Clearing browser
              site data may remove saved expenses, budgets, and exports.
            </p>
          </Card>
        </aside>

        <main className="main-content" id="primary-content" tabIndex={-1}>
          <section className="page-intro" aria-labelledby="app-title">
            <p className="section-eyebrow">Local budget clarity</p>
            <h1 id="app-title">SpendWise</h1>
            <p>
              Track expenses, check the month, tune your budget, and export a
              CSV without login prompts or external account connections.
            </p>
          </section>

          <ScreenPreview item={activeScreen} />

          <ExpenseList
            expenses={spendWiseState.expenses}
            onEditExpense={openEditExpenseDialog}
          />

          <section className="action-row" aria-label="Current screen actions">
            <Button
              variant="primary"
              onClick={(event) => openAddExpenseDialog(event.currentTarget)}
            >
              Add expense
            </Button>
            <Button
              aria-label={`Refresh ${activeScreen.label}`}
              onClick={() =>
                setToastMessage(`${activeScreen.label} is already up to date.`)
              }
            >
              Refresh view
            </Button>
          </section>
        </main>
      </div>

      <ExpenseDialog
        errors={spendWiseState.validationErrors}
        mode={expenseDialog.mode}
        onCancel={closeExpenseDialog}
        onFieldChange={expenseDialog.updateField}
        onSubmit={saveExpense}
        open={expenseDialog.isOpen}
        values={expenseDialog.values}
      />

      <Toast
        message={spendWiseState.feedback?.message ?? toastMessage}
        open={Boolean(spendWiseState.feedback?.message ?? toastMessage)}
      />
    </div>
  );
}
