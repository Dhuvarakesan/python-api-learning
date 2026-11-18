import type { HTMLAttributes, ReactNode } from 'react';

interface CardProps extends HTMLAttributes<HTMLElement> {
  children: ReactNode;
  elevated?: boolean;
}

export function Card({
  children,
  className,
  elevated = false,
  ...props
}: CardProps) {
  const classes = ['card', elevated ? 'card--elevated' : '', className]
    .filter(Boolean)
    .join(' ');

  return (
    <section className={classes} {...props}>
      {children}
    </section>
  );
}
