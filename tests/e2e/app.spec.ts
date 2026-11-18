import { expect, test } from '@playwright/test';

test('loads the app shell and persona routes', async ({ page }) => {
  await page.goto('/');

  await expect(
    page.getByRole('heading', { name: 'Clinic booking application shell' }),
  ).toBeVisible();

  await page.getByRole('link', { name: 'Patient', exact: true }).click();
  await expect(page).toHaveURL('/patient');
  await expect(page.getByRole('heading', { name: 'Patient portal' })).toBeVisible();

  await page.getByRole('link', { name: 'Staff', exact: true }).click();
  await expect(page).toHaveURL('/staff');
  await expect(page.getByRole('heading', { name: 'Staff workspace' })).toBeVisible();

  await page.getByRole('link', { name: 'Admin', exact: true }).click();
  await expect(page).toHaveURL('/staff/admin');
  await expect(page.getByRole('heading', { name: 'Admin console' })).toBeVisible();
});
