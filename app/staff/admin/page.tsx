import { Button, Card, DataTable, Notice, Tag } from '../../../components/ui';

const permissions = [
  ['Scheduler', 'Appointments', 'Manage booking queues'],
  ['Provider', 'Visits', 'Review assigned visits'],
  ['Admin', 'Configuration', 'Manage staff and clinic settings'],
];

export default function AdminPage() {
  return (
    <div className="screen-stack">
      <section className="page-header" aria-labelledby="admin-title">
        <Tag>Admin</Tag>
        <h1 id="admin-title">Admin console</h1>
        <p>
          Placeholder for clinic configuration, RBAC controls, and audit-ready
          administration.
        </p>
      </section>

      <div className="two-column">
        <Card title="Role model" eyebrow="RBAC">
          <DataTable
            caption="Sample role permissions"
            headers={['Role', 'Area', 'Responsibility']}
            rows={permissions}
          />
        </Card>
        <Card title="Clinic setup" eyebrow="Configuration">
          <p>Administrative setup actions will be implemented in later specs.</p>
          <Button>Prepare setup</Button>
        </Card>
      </div>

      <Notice title="Audit readiness">
        This route reserves the administrative surface for future audit log and
        security review workflows.
      </Notice>
    </div>
  );
}
