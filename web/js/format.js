// Μορφοποίηση αριθμών/ημερομηνιών στα ελληνικά (Intl el-GR) και κοινές ετικέτες.
import { html } from './dom.js';

const LOCALE = 'el-GR';
const cache = new Map();
function nf(opts) {
  const key = JSON.stringify(opts);
  if (!cache.has(key)) cache.set(key, new Intl.NumberFormat(LOCALE, opts));
  return cache.get(key);
}

export const DASH = '—';
const MINUS = '−';
const fixMinus = (s) => s.replace(/-/g, MINUS);

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const toNum = (v) => (isNum(v) ? v : (typeof v === 'string' && v.trim() !== '' && Number.isFinite(+v) ? +v : null));

/** Ποσοστό από κλάσμα: 0.0235 → «2,35%» (με πρόσημο αν sign). */
export function pct(v, { digits = 1, sign = false } = {}) {
  v = toNum(v);
  if (v == null) return DASH;
  return fixMinus(nf({ style: 'percent', minimumFractionDigits: digits, maximumFractionDigits: digits, signDisplay: sign ? 'exceptZero' : 'auto' }).format(v));
}

/** Πιθανότητα χωρίς δεκαδικά: 0.413 → «41%». */
export const prob = (v) => pct(v, { digits: 0 });

/** Ποσό σε ευρώ: «1.234,56 €». */
export function eur(v, { digits = 2, compact = false, sign = false } = {}) {
  v = toNum(v);
  if (v == null) return DASH;
  if (compact && Math.abs(v) >= 10000) {
    return fixMinus(nf({ style: 'currency', currency: 'EUR', notation: 'compact', maximumFractionDigits: 1, signDisplay: sign ? 'exceptZero' : 'auto' }).format(v));
  }
  return fixMinus(nf({ style: 'currency', currency: 'EUR', minimumFractionDigits: digits, maximumFractionDigits: digits, signDisplay: sign ? 'exceptZero' : 'auto' }).format(v));
}

/** Τιμή μετοχής: 2 δεκαδικά (3 κάτω από 1 €). */
export function price(v, { unit = true } = {}) {
  v = toNum(v);
  if (v == null) return DASH;
  const d = Math.abs(v) < 1 ? 3 : 2;
  const s = nf({ minimumFractionDigits: d, maximumFractionDigits: d }).format(v);
  return fixMinus(unit ? `${s} €` : s);
}

export function num(v, digits = 2, { sign = false } = {}) {
  v = toNum(v);
  if (v == null) return DASH;
  return fixMinus(nf({ minimumFractionDigits: digits, maximumFractionDigits: digits, signDisplay: sign ? 'exceptZero' : 'auto' }).format(v));
}

export function int(v) {
  v = toNum(v);
  if (v == null) return DASH;
  return fixMinus(nf({ maximumFractionDigits: 0 }).format(v));
}

export function compact(v) {
  v = toNum(v);
  if (v == null) return DASH;
  return fixMinus(nf({ notation: 'compact', maximumFractionDigits: 1 }).format(v));
}

/** «YYYY-MM-DD» (ή ISO) → Date τοπικής ζώνης χωρίς μετατόπιση. */
export function parseDate(s) {
  if (!s) return null;
  if (s instanceof Date) return Number.isNaN(s.getTime()) ? null : s;
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:$|T)/.exec(String(s));
  if (m && String(s).length === 10) return new Date(+m[1], +m[2] - 1, +m[3]);
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

const DF = {
  full: { day: '2-digit', month: '2-digit', year: 'numeric' },
  short: { day: 'numeric', month: 'short' },
  long: { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' },
  month: { month: 'short' },
  monthYear: { month: 'short', year: 'numeric' },
  year: { year: 'numeric' },
  time: { hour: '2-digit', minute: '2-digit' },
  dateTime: { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' },
};
const dfCache = new Map();
export function date(s, style = 'full') {
  const d = parseDate(s);
  if (!d) return DASH;
  if (!dfCache.has(style)) dfCache.set(style, new Intl.DateTimeFormat(LOCALE, DF[style] || DF.full));
  return dfCache.get(style).format(d);
}

/** Σχετικός χρόνος: «πριν από 5 λεπτά». */
export function ago(s) {
  const d = parseDate(s);
  if (!d) return DASH;
  const rtf = new Intl.RelativeTimeFormat('el', { numeric: 'auto' });
  const sec = (d.getTime() - Date.now()) / 1000;
  const abs = Math.abs(sec);
  if (abs < 60) return rtf.format(Math.round(sec), 'second');
  if (abs < 3600) return rtf.format(Math.round(sec / 60), 'minute');
  if (abs < 86400) return rtf.format(Math.round(sec / 3600), 'hour');
  return rtf.format(Math.round(sec / 86400), 'day');
}

/** Μετρική πράκτορα με βάση το fmt του API. */
export function metric(value, fmt) {
  if (value == null || value === '') return DASH;
  switch (fmt) {
    case 'pct': return pct(value, { digits: 2 });
    case 'eur': {
      const v = toNum(value);
      if (v == null) return String(value);
      return Math.abs(v) >= 100000 ? eur(v, { compact: true }) : (Math.abs(v) < 1000 ? price(v) : eur(v, { digits: 0 }));
    }
    case 'num': return num(value, 2);
    case 'int': return int(value);
    case 'date': return date(value);
    case 'text':
    default: return typeof value === 'number' ? num(value, 2) : String(value);
  }
}

export const tone = (v) => (!isNum(v) || v === 0 ? 'flat' : v > 0 ? 'pos' : 'neg');

/** Μεταβολή με βέλος (όχι μόνο χρώμα): ▲ +1,23% */
export function change(v, { digits = 2 } = {}) {
  if (!isNum(v)) return html`<span class="chg flat">${DASH}</span>`;
  const t = tone(v);
  const arrow = t === 'pos' ? '▲' : t === 'neg' ? '▼' : '•';
  const label = t === 'pos' ? 'άνοδος' : t === 'neg' ? 'πτώση' : 'αμετάβλητη';
  return html`<span class="chg ${t}"><span class="arr" aria-hidden="true">${arrow}</span><span class="sr-only">${label} </span>${pct(v, { digits, sign: true })}</span>`;
}

export const shortTicker = (t) => String(t || '').replace(/\.AT$/i, '');

/** Σειρά εμφάνισης των τριών καταστάσεων παντού (υπόμνημα, γραφήματα, tooltips): καλό → ουδέτερο → κακό. */
export const ORDER = ['tp', 'to', 'sl'];

/** Οι τρεις καταστάσεις της σφαίρας. */
export const STATES = {
  tp: { key: 'tp', label: 'Στόχος πρώτα', short: 'Στόχος', chip: 'chip-pos', cls: 'oc-tp', glyph: '✓' },
  sl: { key: 'sl', label: 'Stop πρώτα', short: 'Stop', chip: 'chip-neg', cls: 'oc-sl', glyph: '✕' },
  to: { key: 'to', label: 'Λήξη χρόνου', short: 'Λήξη', chip: 'chip-to', cls: 'oc-to', glyph: '◷' },
  open: { key: 'open', label: 'Ανοιχτή', short: 'Ανοιχτή', chip: 'chip-info', cls: 'oc-open', glyph: '…' },
};

/** Ετικέτα έκβασης από 0/1/2, «tp»/«sl»/«to»/«timeout» ή −1/null (ανοιχτή). */
export function labelKey(label) {
  if (label == null) return 'open';
  if (typeof label === 'number') return label === 0 ? 'tp' : label === 1 ? 'sl' : label === 2 ? 'to' : 'open';
  const s = String(label).toLowerCase();
  if (s === '0' || s === 'tp' || s === 'take_profit') return 'tp';
  if (s === '1' || s === 'sl' || s === 'stop') return 'sl';
  if (s === '2' || s === 'to' || s === 'timeout') return 'to';
  return 'open';
}

export const DECISIONS = {
  take: { label: 'ΥΠΟΨΗΦΙΑ ΑΓΟΡΑ', short: 'Αγορά', chip: 'chip-pos', icon: 'check' },
  skip: { label: 'ΑΠΟΡΡΙΨΗ', short: 'Απόρριψη', chip: 'chip-outline', icon: 'minus' },
  blocked: { label: 'ΜΠΛΟΚ ΑΠΟ ΦΥΛΑΚΑ', short: 'Μπλοκ', chip: 'chip-neg', icon: 'ban' },
  none: { label: 'ΧΩΡΙΣ ΣΗΜΑ', short: 'Χωρίς σήμα', chip: 'chip', icon: 'clock' },
};

export function decisionChip(d, { small = false } = {}) {
  const k = DECISIONS[d] ? d : 'none';
  const glyph = k === 'take' ? '▲ ' : k === 'blocked' ? '✕ ' : '';
  return html`<span class="chip ${DECISIONS[k].chip} ${small ? 'chip-sm' : ''}">${glyph}${DECISIONS[k].short}</span>`;
}

export function stateChip(label, { small = true } = {}) {
  const k = labelKey(label);
  const s = STATES[k];
  return html`<span class="chip ${s.chip} ${small ? 'chip-sm' : ''}"><span aria-hidden="true">${s.glyph}</span> ${s.short}</span>`;
}

