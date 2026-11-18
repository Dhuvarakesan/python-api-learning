import React from 'react';
import { createRoot } from 'react-dom/client';
import { createServiceConfig } from '@clinic/config';
import type { StaffUser } from '@clinic/shared';

const staffUser: StaffUser = {
  id: 'staff-demo',
  displayName: 'Clinic Coordinator',
  roles: ['scheduler'],
};

const config = createServiceConfig(import.meta.env);

function StaffApp() {
  return (
    <main>
      <h1>Staff scheduling</h1>
      <p>Signed in as {staffUser.displayName}</p>
      <p>Staff endpoint: {config.staffBaseUrl}</p>
    </main>
  );
}

createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <StaffApp />
  </React.StrictMode>,
);
