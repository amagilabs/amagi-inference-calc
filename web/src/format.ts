export const fmtTok = (x: number): string =>
  x >= 100 ? x.toFixed(0) : x.toFixed(1);

export const fmtSec = (s: number | null): string => {
  if (s === null) return 'n/a';
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${s.toFixed(0)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
};

const usdFmt = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});

export const fmtUsd = (x: number | null): string =>
  x === null ? 'no price' : usdFmt.format(x);

export const fmtGiB = (x: number): string => `${x.toFixed(1)} GiB`;

export const fmtTokens = (n: number): string =>
  n >= 1024 && n % 1024 === 0 ? `${n / 1024}K` : String(n);

export const billions = (params: number): string => {
  const b = params / 1e9;
  return b >= 10 ? `${b.toFixed(0)}B` : `${b.toFixed(1)}B`;
};
