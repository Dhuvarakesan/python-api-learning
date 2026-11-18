import { Button, Card, DataTable, Dialog, Notice, Tag } from '../../components/ui';

const rows = [
  ['8:30 AM', 'New patient', 'Check-in pending'],
  ['9:15 AM', 'Follow-up', 'Room ready'],
  ['10:00 AM', 'Consultation', 'Provider assigned'],
];

export default function StaffPage() {
  return (
    <div className="screen-stack">
      <section className="page-header" aria-labelledby="staff-title">
        <Tag>Staff</Tag>
        <h1 id="staff-title">Staff workspace</h1>
        <p>
          Placeholder for authenticated schedule operations, MFA prompts, and
          clinic-day coordination.
        </p>
      </section>

      <div className="two-column">
        <Card title="Today at a glance" eyebrow="Schedule">
          <DataTable
            caption="Sample staff schedule"
            headers={['Time', 'Visit type', 'Status']}
            rows={rows}
          />
        </Card>
        <Card title="Staff access" eyebrow="Security">
          <p>Role-aware workflows and MFA checkpoints will be composed here.</p>
          <div className="button-row">
            <Button>Start MFA check</Button>
            <Dialog summary="View access note">
              Staff access placeholders are intentionally static until auth
              integrations are introduced.
            </Dialog>
          </div>
        </Card>
      </div>

      <Notice title="Operational shell">
        The staff route is available for subsequent schedule and visit
        management tasks.
      </Notice>
    </div>
  );
}
