export const INR_CURRENCY_CODE = 'INR';

export const INR_LOCALE = 'en-IN';

export const inrCurrencyFormatter = new Intl.NumberFormat(INR_LOCALE, {
  currency: INR_CURRENCY_CODE,
  style: 'currency',
});

export const formatInrCurrency = (amount: number) =>
  inrCurrencyFormatter.format(amount);
