export type PrimaryScreenId =
  | 'monthly-summary'
  | 'expense-history'
  | 'monthly-budget'
  | 'csv-export';

export interface PrimaryNavigationItem {
  id: PrimaryScreenId;
  label: string;
  eyebrow: string;
  title: string;
  description: string;
}

export const primaryNavigation: PrimaryNavigationItem[] = [
  {
    id: 'monthly-summary',
    label: 'Monthly summary',
    eyebrow: 'Overview',
    title: 'Monthly summary',
    description:
      'See spending totals, category patterns, and budget progress for the current month.',
  },
  {
    id: 'expense-history',
    label: 'Expense history',
    eyebrow: 'Records',
    title: 'Expense history',
    description:
      'Review locally saved expense records with filters for month and category.',
  },
  {
    id: 'monthly-budget',
    label: 'Monthly budget',
    eyebrow: 'Planning',
    title: 'Monthly budget',
    description:
      'Set a monthly target and keep everyday spending aligned with it.',
  },
  {
    id: 'csv-export',
    label: 'CSV export',
    eyebrow: 'Portable data',
    title: 'CSV export',
    description:
      'Download local expense records as a CSV file whenever you need a copy.',
  },
];

export const defaultScreenId: PrimaryScreenId = 'monthly-summary';
