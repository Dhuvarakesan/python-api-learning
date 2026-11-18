import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import HomePage from '../../app/page';
import PatientPage from '../../app/patient/page';
import StaffPage from '../../app/staff/page';
import AdminPage from '../../app/staff/admin/page';

describe('route rendering', () => {
  it('renders the home route with patient, staff, and admin navigation', () => {
    render(<HomePage />);

    expect(
      screen.getByRole('heading', { name: 'Clinic booking application shell' }),
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open patient portal' })).toHaveAttribute(
      'href',
      '/patient',
    );
    expect(screen.getByRole('link', { name: 'Staff workspace' })).toHaveAttribute(
      'href',
      '/staff',
    );
    expect(screen.getByRole('link', { name: 'Open admin' })).toHaveAttribute(
      'href',
      '/staff/admin',
    );
  });

  it('renders the patient route placeholder', () => {
    render(<PatientPage />);

    expect(screen.getByRole('heading', { name: 'Patient portal' })).toBeInTheDocument();
    expect(screen.getByLabelText('Reason for visit')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Continue booking' }),
    ).toBeInTheDocument();
  });

  it('renders the staff route placeholder', () => {
    render(<StaffPage />);

    expect(screen.getByRole('heading', { name: 'Staff workspace' })).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Sample staff schedule' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start MFA check' })).toBeInTheDocument();
  });

  it('renders the admin route placeholder', () => {
    render(<AdminPage />);

    expect(screen.getByRole('heading', { name: 'Admin console' })).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Sample role permissions' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Prepare setup' })).toBeInTheDocument();
  });
});
