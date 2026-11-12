export type TaskStatus = 'pending' | 'completed';

export type TaskPriority = 'low' | 'medium' | 'high';

export interface Task {
  id: string;
  title: string;
  details: string;
  status: TaskStatus;
  priority: TaskPriority;
  dueDate: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface TaskFormInput {
  title: string;
  details: string;
  priority: TaskPriority;
  dueDate: string | null;
}

export type TaskFormErrors = Partial<Record<keyof TaskFormInput, string>>;
