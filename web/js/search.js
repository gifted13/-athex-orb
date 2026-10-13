// Τοπική αναζήτηση χωρίς τόνους, με μεταγραφή ελληνικών ↔ λατινικών (π.χ. «opap» ↔ «οπαπ»).
import { esc } from './dom.js';

/** Ένας χαρακτήρας → πεζός χωρίς τόνο (1:1, ώστε να ταιριάζουν οι θέσεις για επισήμανση). */
function foldChar(ch) {
  const base = ch.normalize('NFD')[0] || ch;
  const low = base.toLowerCase();
  return low === 'ς' ? 'σ' : low;
}

/** Κανονικοποίηση όπως η universe.fold() του backend. */
export function fold(text) {
  return Array.from(String(text || ''), foldChar).join('').trim();
}

const GR_NAIVE = { α: 'a', β: 'v', γ: 'g', δ: 'd', ε: 'e', ζ: 'z', η: 'i', θ: 'th', ι: 'i', κ: 'k', λ: 'l', μ: 'm', ν: 'n', ξ: 'x', ο: 'o', π: 'p', ρ: 'r', σ: 's', τ: 't', υ: 'y', φ: 'f', χ: 'ch', ψ: 'ps', ω: 'o' };
const DIGRAPHS = [['ου', 'ou'], ['αυ', 'av'], ['ευ', 'ev'], ['μπ', 'b'], ['ντ', 'd'], ['γκ', 'g'], ['γγ', 'ng']];

function translitNaive(s) {
  return s.replace(/[α-ω]/g, (c) => GR_NAIVE[c] ?? c);
}
function translitPhonetic(s) {
  let out = s;
  for (const [a, b] of DIGRAPHS) out = out.split(a).join(b);
  return translitNaive(out).replace(/ch/g, 'h');
}

/** Όλες οι παραλλαγές ενός κειμένου για αντιστοίχιση. */
export function variants(text) {
  const f = fold(text);
  const set = new Set([f, translitNaive(f), translitPhonetic(f)]);
  return Array.from(set);
}

const TICKER_RE = /^[A-Z0-9][A-Z0-9-]{0,14}(\.AT)?$/;

/** Όπως normalize_ticker(): «opap» → «OPAP.AT», αλλιώς null. */
export function normalizeTicker(rawText) {
  if (!rawText) return null;
  let t = String(rawText).trim().toUpperCase().replace(/\s+/g, '');
  if (!TICKER_RE.test(t)) return null;
  if (!t.endsWith('.AT')) t += '.AT';
  return t;
}

/** Προετοιμασία ευρετηρίου για ένα Item του universe. */
export function indexItem(item) {
  const tick = String(item.ticker || '').replace(/\.AT$/i, '');
  return {
    tick: variants(tick),
    name: variants(item.name || ''),
    extra: (item.aliases || []).flatMap(variants),
    sector: variants(item.sector || ''),
  };
}

/** Βαθμολογία ταιριάσματος (μικρότερο = καλύτερο), ή null αν δεν ταιριάζει. */
export function score(idx, query) {
  const qv = variants(query).filter(Boolean);
  if (!qv.length) return 0;
  let best = null;
  const take = (s) => { if (best == null || s < best) best = s; };
  for (const q of qv) {
    if (idx.tick.some((t) => t === q)) take(0);
    else if (idx.tick.some((t) => t.startsWith(q))) take(1);
    const fields = idx.name.concat(idx.extra);
    if (fields.some((f) => f.startsWith(q) || f.split(/[\s()\-.]+/).some((w) => w && w.startsWith(q)))) take(2);
    else if (fields.some((f) => f.includes(q)) || idx.tick.some((t) => t.includes(q))) take(3);
    else if (idx.sector.some((s) => s.includes(q))) take(4);
  }
  return best;
}

/** Φιλτράρει και ταξινομεί items (σταθερά, με δευτερεύον κλειδί το σύμβολο). */
export function localSearch(items, query, cache = new WeakMap()) {
  const q = fold(query);
  if (!q) return items.slice();
  const out = [];
  for (const it of items) {
    let idx = cache.get(it);
    if (!idx) { idx = indexItem(it); cache.set(it, idx); }
    const s = score(idx, q);
    if (s != null) out.push([s, it]);
  }
  out.sort((a, b) => a[0] - b[0] || String(a[1].ticker).localeCompare(String(b[1].ticker)));
  return out.map((x) => x[1]);
}

/** Επισήμανση του ταιριάσματος σε κείμενο (επιστρέφει ασφαλές HTML string). */
export function highlight(text, query) {
  const s = String(text ?? '');
  const q = fold(query);
  if (!q) return esc(s);
  const folded = Array.from(s, foldChar).join('');
  const chars = Array.from(s);
  const i = folded.indexOf(q);
  if (i < 0) return esc(s);
  const a = chars.slice(0, i).join('');
  const b = chars.slice(i, i + Array.from(q).length).join('');
  const c = chars.slice(i + Array.from(q).length).join('');
  return `${esc(a)}<mark>${esc(b)}</mark>${esc(c)}`;
}
