import Link from 'next/link';
import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
} from 'react';

type Tone = 'primary' | 'secondary' | 'ghost';

export function Button({
  children,
  className,
  type = 'button',
  variant = 'primary',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  variant?: Tone;
}) {
  return (
    <button
      className={['button', `button--${variant}`, className].filter(Boolean).join(' ')}
      type={type}
      {...props}
    >
      {children}
    </button>
  );
}

export function ButtonLink({
  children,
  href,
  variant = 'primary',
}: {
  children: ReactNode;
  href: string;
  variant?: Tone;
}) {
  return (
    <Link className={`button button--${variant}`} href={href}>
      {children}
    </Link>
  );
}

export function Card({
  children,
  eyebrow,
  title,
}: {
  children: ReactNode;
  eyebrow?: string;
  title: string;
}) {
  return (
    <article className="card">
      {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
      <h2>{title}</h2>
      <div className="card__body">{children}</div>
    </article>
  );
}

export function FormField({
  label,
  name,
  type = 'text',
  ...props
}: InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  name: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input id={name} name={name} type={type} {...props} />
    </label>
  );
}

export function Notice({
  children,
  title,
}: {
  children: ReactNode;
  title: string;
}) {
  return (
    <aside className="notice" role="note" aria-label={title}>
      <strong>{title}</strong>
      <p>{children}</p>
    </aside>
  );
}

export function Tag({ children }: { children: ReactNode }) {
  return <span className="tag">{children}</span>;
}

export function Switch({ label, name }: { label: string; name: string }) {
  return (
    <label className="switch">
      <input name={name} type="checkbox" />
      <span aria-hidden="true" />
      <strong>{label}</strong>
    </label>
  );
}

export function DataTable({
  caption,
  headers,
  rows,
}: {
  caption: string;
  headers: string[];
  rows: string[][];
}) {
  return (
    <div className="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header} scope="col">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.join('-')}>
              {row.map((cell) => (
                <td key={cell}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Dialog({
  children,
  summary,
}: {
  children: ReactNode;
  summary: string;
}) {
  return (
    <details className="dialog">
      <summary>{summary}</summary>
      <div role="group" aria-label={summary}>
        {children}
      </div>
    </details>
  );
}
