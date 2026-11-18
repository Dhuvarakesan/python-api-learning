import { Button, Card, FormField, Notice, Switch, Tag } from '../../components/ui';

export default function PatientPage() {
  return (
    <div className="screen-stack">
      <section className="page-header" aria-labelledby="patient-title">
        <Tag>Patient</Tag>
        <h1 id="patient-title">Patient portal</h1>
        <p>
          Placeholder for appointment discovery, booking requests, and
          passwordless patient access.
        </p>
      </section>

      <div className="two-column">
        <Card title="Book an appointment" eyebrow="Intake">
          <form className="form-grid" action="#">
            <FormField label="Reason for visit" name="visit-reason" placeholder="Annual physical" />
            <FormField label="Preferred date" name="preferred-date" type="date" />
            <Switch label="Text me appointment updates" name="patient-updates" />
            <Button type="submit">Continue booking</Button>
          </form>
        </Card>
        <Notice title="Passwordless access">
          Future slices can attach magic links or one-time passcodes here while
          preserving the patient-first route shell.
        </Notice>
      </div>
    </div>
  );
}
