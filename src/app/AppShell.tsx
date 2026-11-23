import { useMemo, useState } from 'react';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Toast } from '../components/ui/Toast';
import { EXPENSE_CATEGORIES } from '../domain/categories';
import { formatInrCurrency } from '../domain/currency';
import {
  getCurrentMonthString,
  type MonthString,
} from '../domain/date';
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

const DEFAULT_MONTHLY_BUDGET = 50000;

interface ScreenMetric {
  label: string;
  value: string;
}

const getExpenseMonth = (expense: Expense): MonthString =>
  expense.date.slice(0, 7) as MonthString;

const sortExpensesByDateDescending = (expenses: Expense[]): Expense[] =>
  [...expenses].sort((left, right) => {
    const dateComparison = right.date.localeCompare(left.date);

    if (dateComparison !== 0) {
      return dateComparison;
    }

    return right.updatedAt.localeCompare(left.updatedAt);
  });

const getBudgetHealth = (spent: number): string => {
  if (spent === 0) {
    return 'No spend yet';
  }

  const spentRatio = spent / DEFAULT_MONTHLY_BUDGET;

  if (spentRatio > 1) {
    return 'Over budget';
  }

  if (spentRatio >= 0.85) {
    return 'Watch pace';
  }

  return 'On track';
};

function ScreenPreview({
  item,
  metrics,
  selectedMonth,
}: {
  item: PrimaryNavigationItem;
  metrics: ScreenMetric[];
  selectedMonth: MonthString;
}) {
  return (
    <Card elevated aria-labelledby={`${item.id}-title`} className="screen-card">
      <p className="section-eyebrow">{item.eyebrow}</p>
      <div className="screen-card__header">
        <div>
          <h2 id={`${item.id}-title`}>{item.title}</h2>
          <p>{item.description}</p>
        </div>
        <span className="screen-card__status">{selectedMonth}</span>
      </div>
      <div className="metric-grid" aria-label={`${item.label} preview`}>
        {metrics.map((metric) => (
          <div className="metric" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </div>
        ))}
      </div>
    </Card>
  );
}

function ExpenseList({
  expenses,
  onEditExpense,
  selectedMonth,
  totalExpenseCount,
}: {
  expenses: Expense[];
  selectedMonth: MonthString;
  totalExpenseCount: number;
  onEditExpense(expense: Expense, opener: HTMLElement): void;
}) {
  return (
    <Card className="expenses-panel" aria-labelledby="expenses-panel-title">
      <div className="expenses-panel__header">
        <div>
          <p className="section-eyebrow">Expense records</p>
          <h2 id="expenses-panel-title">Recent expenses</h2>
        </div>
        <span>
          {expenses.length} in {selectedMonth}
          {totalExpenseCount !== expenses.length
            ? ` / ${totalExpenseCount} saved`
            : ' saved'}
        </span>
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
  const [selectedMonth, setSelectedMonth] = useState<MonthString>(() =>
    getCurrentMonthString(),
  );
  const [toastMessage, setToastMessage] = useState('');
  const spendWiseState = useSpendWiseState();
  const expenseDialog = useExpenseDialog();

  const activeScreen = useMemo(
    () =>
      primaryNavigation.find((item) => item.id === activeScreenId) ??
      primaryNavigation.find((item) => item.id === defaultScreenId)!,
    [activeScreenId],
  );

  const derivedViews = useMemo(() => {
    const monthlyExpenses = sortExpensesByDateDescending(
      spendWiseState.expenses.filter(
        (expense) => getExpenseMonth(expense) === selectedMonth,
      ),
    );
    const monthlyTotal = monthlyExpenses.reduce(
      (total, expense) => total + expense.amount,
      0,
    );
    const categoryTotals = EXPENSE_CATEGORIES.map((category) => ({
      category,
      total: monthlyExpenses
        .filter((expense) => expense.category === category)
        .reduce((total, expense) => total + expense.amount, 0),
    }))
      .filter((categoryTotal) => categoryTotal.total > 0)
      .sort((left, right) => right.total - left.total);
    const topCategory = categoryTotals[0];
    const remainingBudget = DEFAULT_MONTHLY_BUDGET - monthlyTotal;
    const csvRows = spendWiseState.expenses.length + 1;
    const metricsByScreen: Record<PrimaryScreenId, ScreenMetric[]> = {
      'monthly-summary': [
        {
          label: 'This month',
          value: formatInrCurrency(monthlyTotal),
        },
        {
          label: 'Category mix',
          value: topCategory
            ? `${topCategory.category} ${formatInrCurrency(topCategory.total)}`
            : 'No categories',
        },
        {
          label: 'Expense count',
          value: `${monthlyExpenses.length} saved`,
        },
      ],
      'expense-history': [
        {
          label: 'Recent records',
          value: `${monthlyExpenses.length} visible`,
        },
        {
          label: 'Month filter',
          value: selectedMonth,
        },
        {
          label: 'Category totals',
          value: `${categoryTotals.length} active`,
        },
      ],
      'monthly-budget': [
        {
          label: 'Monthly target',
          value: formatInrCurrency(DEFAULT_MONTHLY_BUDGET),
        },
        {
          label: 'Remaining amount',
          value:
            remainingBudget >= 0
              ? formatInrCurrency(remainingBudget)
              : `${formatInrCurrency(Math.abs(remainingBudget))} over`,
        },
        {
          label: 'Budget health',
          value: getBudgetHealth(monthlyTotal),
        },
      ],
      'csv-export': [
        {
          label: 'CSV preview',
          value: `${csvRows} rows`,
        },
        {
          label: 'Download action',
          value: spendWiseState.expenses.length > 0 ? 'Ready' : 'No records',
        },
        {
          label: 'Local file copy',
          value: `${spendWiseState.expenses.length} records`,
        },
      ],
    };

    return {
      metricsByScreen,
      monthlyExpenses,
    };
  }, [selectedMonth, spendWiseState.expenses]);

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
      setSelectedMonth(getExpenseMonth(savedExpense));
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

          <ScreenPreview
            item={activeScreen}
            metrics={derivedViews.metricsByScreen[activeScreen.id]}
            selectedMonth={selectedMonth}
          />

          <ExpenseList
            expenses={derivedViews.monthlyExpenses}
            onEditExpense={openEditExpenseDialog}
            selectedMonth={selectedMonth}
            totalExpenseCount={spendWiseState.expenses.length}
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
