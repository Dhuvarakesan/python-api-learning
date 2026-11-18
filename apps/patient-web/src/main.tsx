import React from 'react';
import { createRoot } from 'react-dom/client';
import { createServiceConfig } from '@clinic/config';
import type { AppointmentRequest } from '@clinic/shared';

const sampleRequest: AppointmentRequest = {
  id: 'request-demo',
  patient: {
    firstName: 'Avery',
    lastName: 'Patient',
  },
  channel: 'web',
  status: 'requested',
  createdAt: new Date().toISOString(),
};

const config = createServiceConfig(import.meta.env);

function PatientApp() {
  return (
    <main>
      <h1>Patient booking</h1>
      <p>Request status: {sampleRequest.status}</p>
      <p>API endpoint: {config.apiBaseUrl}</p>
    </main>
  );
}

createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <PatientApp />
  </React.StrictMode>,
);
