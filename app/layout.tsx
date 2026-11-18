import type { Metadata } from 'next';
import Link from 'next/link';
import type { ReactNode } from 'react';
import '../styles/globals.css';

const personaLinks = [
  { href: '/', label: 'Home' },
  { href: '/patient', label: 'Patient' },
  { href: '/staff', label: 'Staff' },
  { href: '/staff/admin', label: 'Admin' },
];

export const metadata: Metadata = {
  title: 'Clinic Booking',
  description: 'Clinic appointment booking foundations for patients and staff.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to main content
        </a>
        <header className="topbar">
          <Link className="brand" href="/" aria-label="Clinic Booking home">
            <span className="brand__mark" aria-hidden="true">
              CB
            </span>
            <span>
              <strong>Clinic Booking</strong>
              <small>Patient and staff scheduling</small>
            </span>
          </Link>
          <nav className="persona-nav" aria-label="Persona navigation">
            {personaLinks.map((item) => (
              <Link key={item.href} href={item.href}>
                {item.label}
              </Link>
            ))}
          </nav>
        </header>
        <main id="main-content" className="main-shell" tabIndex={-1}>
          {children}
        </main>
      </body>
    </html>
  );
}
