interface ToastProps {
  message: string;
  open: boolean;
}

export function Toast({ message, open }: ToastProps) {
  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      className={`toast${open ? ' toast--open' : ''}`}
      role="status"
    >
      {message}
    </div>
  );
}
