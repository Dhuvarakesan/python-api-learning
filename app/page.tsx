import Link from 'next/link';
import { ButtonLink, Card, Notice, Tag } from '../components/ui';

const routes = [
  {
    href: '/patient',
    title: 'Patient portal',
    description: 'Appointment search, booking intake, and passwordless access will start here.',
    tag: 'Patient',
  },
  {
    href: '/staff',
    title: 'Staff workspace',
    description: 'Daily schedule management, visit readiness, and MFA entry points will start here.',
    tag: 'Staff',
  },
  {
    href: '/staff/admin',
    title: 'Admin console',
    description: 'RBAC setup, clinic configuration, and operational audit views will start here.',
    tag: 'Admin',
  },
];

export default function HomePage() {
  return (
    <div className="screen-stack">
      <section className="hero-panel" aria-labelledby="home-title">
        <Tag>Foundation</Tag>
        <h1 id="home-title">Clinic booking application shell</h1>
        <p>
          A responsive App Router scaffold for patient booking, staff workflows,
          and administrative controls.
        </p>
        <div className="hero-actions" aria-label="Primary routes">
          <ButtonLink href="/patient">Open patient portal</ButtonLink>
          <Link className="text-link" href="/staff">
            Staff workspace
          </Link>
        </div>
      </section>

      <Notice title="Ready for feature slices">
        This scaffold includes route placeholders, shared navigation, accessible
        controls, and theme-ready styling hooks for later tasks.
      </Notice>

      <section className="card-grid" aria-label="Persona route placeholders">
        {routes.map((route) => (
          <Card key={route.href} title={route.title} eyebrow={route.tag}>
            <p>{route.description}</p>
            <ButtonLink href={route.href} variant="secondary">
              Open {route.tag.toLowerCase()}
            </ButtonLink>
          </Card>
        ))}
      </section>
    </div>
  );
}
