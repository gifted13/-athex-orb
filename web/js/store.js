// Κατάσταση εφαρμογής: status, σύνολο μετοχών, αγαπημένα (με αισιόδοξες ενημερώσεις), εργασίες.
import { api, jobList } from './api.js';
import { createBus, storage } from './dom.js';
import { toast } from './toast.js';
import { normalizeTicker } from './search.js';
import { shortTicker } from './format.js';

export const bus = createBus();

export const state = {
  status: null,
  universe: [],
  byTicker: new Map(),
  universeLoaded: false,
  universeError: null,
  asOf: null,
  favorites: [],
  favoritesLoaded: false,
  job: null,
};

// ---------------------------------------------------------------------------
// Status / universe
// ---------------------------------------------------------------------------
export async function loadStatus() {
  const s = await api.status();
  state.status = s && typeof s === 'object' ? s : {};
  bus.emit('status', state.status);
  return state.status;
}

export const mode = () => (state.status && state.status.mode) || 'empty';

function setUniverse(items) {
  state.universe = (items || []).filter((i) => i && i.ticker);
  state.byTicker = new Map(state.universe.map((i) => [i.ticker, i]));
  for (const it of state.universe) it.favorite = state.favorites.includes(it.ticker) || (!state.favoritesLoaded && !!it.favorite);
  state.universeLoaded = true;
  state.universeError = null;
  bus.emit('universe');
}

export async function loadUniverse() {
  try {
    const data = await api.universe();
    state.asOf = data && data.as_of ? data.as_of : null;
    setUniverse(data && Array.isArray(data.items) ? data.items : []);
  } catch (e) {
    state.universeError = e;
    state.universeLoaded = true;
    bus.emit('universe');
    throw e;
  }
}

export function upsertItem(item) {
  if (!item || !item.ticker) return;
  const i = state.universe.findIndex((x) => x.ticker === item.ticker);
  if (i >= 0) state.universe[i] = { ...state.universe[i], ...item };
  else state.universe.push(item);
  setUniverse(state.universe);
}

// ---------------------------------------------------------------------------
// Αγαπημένα
// ---------------------------------------------------------------------------
function setFavorites(list, { mirror = true } = {}) {
  const seen = new Set();
  state.favorites = (list || []).filter((t) => typeof t === 'string' && !seen.has(t) && seen.add(t));
  for (const it of state.universe) it.favorite = state.favorites.includes(it.ticker);
  if (mirror) storage.set('favorites', state.favorites);
  bus.emit('favorites', state.favorites);
}

/** Φορτώνει τα αγαπημένα· αν ο διακομιστής δεν απαντά, χρησιμοποιεί σημαίες/τοπικό αντίγραφο. */
export async function loadFavorites() {
  try {
    const r = await api.favorites();
    const list = Array.isArray(r && r.tickers) ? r.tickers : [];
    state.favoritesLoaded = true;
    setFavorites(list);
  } catch {
    const flags = state.universe.filter((i) => i.favorite).map((i) => i.ticker);
    const mirror = storage.get('favorites', []);
    setFavorites(flags.length ? flags : (Array.isArray(mirror) ? mirror : []), { mirror: false });
  }
}

export const isFav = (t) => state.favorites.includes(t);

function applyFav(ticker, on) {
  const has = isFav(ticker);
  if (on && !has) setFavorites([...state.favorites, ticker]);
  else if (!on && has) setFavorites(state.favorites.filter((t) => t !== ticker));
}

const favChains = new Map();

/** Αισιόδοξη εναλλαγή αγαπημένου· επαναφορά + ειδοποίηση σε σφάλμα. Επιστρέφει τη νέα κατάσταση. */
export function setFavorite(ticker, want) {
  const was = isFav(ticker);
  const next = want ?? !was;
  if (next === was) return Promise.resolve(next);
  applyFav(ticker, next);
  const prev = favChains.get(ticker) || Promise.resolve();
  const chain = prev.then(async () => {
    if (isFav(ticker) !== next) return isFav(ticker); // ο χρήστης το άλλαξε ξανά στο μεταξύ
    try {
      if (next) await api.addFavorite(ticker); else await api.removeFavorite(ticker);
      return next;
    } catch (e) {
      if (isFav(ticker) === next) applyFav(ticker, was);
      toast(`Δεν αποθηκεύτηκε η αλλαγή στα αγαπημένα: ${e.message}`, { type: 'error' });
      return was;
    }
  });
  favChains.set(ticker, chain);
  return chain;
}

/** Εναλλαγή με ειδοποίηση και «Αναίρεση». */
export function toggleFavoriteUI(ticker) {
  const item = state.byTicker.get(ticker);
  const name = item ? item.name : shortTicker(ticker);
  const next = !isFav(ticker);
  setFavorite(ticker, next);
  // χωρίς άρθρο: «η ΟΠΑΠ», «η ΟΤΕ» κ.λπ. θα ήταν λάθος για πολλά ονόματα
  if (next) {
    toast(`Προστέθηκε στα αγαπημένα: ${name}`, { type: 'success', action: { label: 'Αναίρεση', onClick: () => setFavorite(ticker, false) } });
  } else {
    toast(`Αφαιρέθηκε από τα αγαπημένα: ${name}`, { type: 'info', action: { label: 'Αναίρεση', onClick: () => setFavorite(ticker, true) } });
  }
  return next;
}

// ---------------------------------------------------------------------------
// Προσθήκη / αφαίρεση μετοχής χρήστη
// ---------------------------------------------------------------------------
export async function addTicker(rawText, { favorite = true } = {}) {
  const ticker = normalizeTicker(rawText);
  if (!ticker) {
    toast('Μη έγκυρο σύμβολο. Χρησιμοποιήστε λατινικούς χαρακτήρες, π.χ. «INTEK» ή «INTEK.AT».', { type: 'error' });
    return null;
  }
  const close = toast(`Προσθήκη ${ticker}… λήψη δεδομένων`, { type: 'info', timeout: 30000 });
  try {
    const r = await api.addStock({ ticker });
    close();
    if (!r || r.ok === false) {
      toast((r && r.error_el) || `Δεν ήταν δυνατή η προσθήκη του ${ticker}.`, { type: 'error' });
      return null;
    }
    if (r.item) upsertItem(r.item);
    else await loadUniverse().catch(() => {});
    if (favorite) await setFavorite(ticker, true);
    toast(`Προστέθηκε στη λίστα${favorite ? ' και στα αγαπημένα' : ''}: ${ticker}${r.note_el ? ` — ${r.note_el}` : ''}`, { type: 'success', timeout: r.note_el ? 6000 : undefined });
    return ticker;
  } catch (e) {
    close();
    toast(e.message, { type: 'error' });
    return null;
  }
}

export async function removeTicker(ticker) {
  try {
    await api.removeStock(ticker);
    applyFav(ticker, false);
    state.universe = state.universe.filter((i) => i.ticker !== ticker);
    setUniverse(state.universe);
    stockCache.delete(ticker);
    toast(`Αφαιρέθηκε από τη λίστα: ${shortTicker(ticker)}`, { type: 'success' });
    return true;
  } catch (e) {
    toast(e.message, { type: 'error' });
    return false;
  }
}

// ---------------------------------------------------------------------------
// Λεπτομέρειες μετοχής (cache + prefetch)
// ---------------------------------------------------------------------------
const stockCache = new Map();
const inflight = new Map();
const STOCK_TTL = 5 * 60 * 1000;

export function cachedStock(ticker) {
  const c = stockCache.get(ticker);
  return c && Date.now() - c.t < STOCK_TTL ? c.data : null;
}

export function fetchStock(ticker) {
  if (inflight.has(ticker)) return inflight.get(ticker);
  const p = api.stock(ticker, 260)
    .then((d) => {
      stockCache.set(ticker, { data: d, t: Date.now() });
      if (stockCache.size > 40) stockCache.delete(stockCache.keys().next().value);
      return d;
    })
    .finally(() => inflight.delete(ticker));
  inflight.set(ticker, p);
  return p;
}

export function prefetchStock(ticker) {
  if (!ticker || cachedStock(ticker) || inflight.has(ticker)) return;
  fetchStock(ticker).catch(() => {});
}

export function clearStockCache() { stockCache.clear(); }

// ---------------------------------------------------------------------------
// Εργασίες (refresh / backtest / demo) με polling κάθε 1s
// ---------------------------------------------------------------------------
const JOB_DONE = {
  refresh: 'Η ενημέρωση ολοκληρώθηκε — τα σήματα είναι ενημερωμένα.',
  backtest: 'Το backtest ολοκληρώθηκε.',
  demo: 'Τα δεδομένα επίδειξης είναι έτοιμα.',
};
export const JOB_LABEL = { refresh: 'Ενημέρωση', backtest: 'Backtest', demo: 'Δεδομένα επίδειξης' };
let pollTimer = null;

export async function startJob(type) {
  if (state.job && (state.job.status === 'running' || state.job.status === 'queued')) {
    toast(`Τρέχει ήδη εργασία: ${JOB_LABEL[state.job.type] || state.job.type}. Περιμένετε να ολοκληρωθεί.`, { type: 'info' });
    return;
  }
  if (type === 'demo' && mode() === 'live') {
    // τα δεδομένα επίδειξης αντικαθιστούν την αποθηκευμένη cache πραγματικών τιμών
    // eslint-disable-next-line no-alert
    if (!window.confirm('Τα δεδομένα επίδειξης θα αντικαταστήσουν τις αποθηκευμένες πραγματικές τιμές (θα ξανακατέβουν με την επόμενη «Ενημέρωση»). Το ημερολόγιο των πραγματικών δεδομένων δεν αλλάζει. Συνέχεια;')) return;
  }
  try {
    const r = await api.startJob(type);
    attachJob({ id: r.job_id, type, status: 'queued', progress: 0, message_el: 'Σε αναμονή…' });
  } catch (e) {
    if (e.status === 409) {
      const jobs = jobList(await api.jobs().catch(() => []));
      const running = jobs.find((j) => j.status === 'running' || j.status === 'queued');
      if (running) attachJob(running);
      toast('Η εργασία εκτελείται ήδη — παρακολουθείτε την πρόοδό της.', { type: 'info' });
    } else {
      toast(e.message, { type: 'error' });
    }
  }
}

export function attachJob(job) {
  if (!job || !job.id) return;
  state.job = { ...job };
  bus.emit('job', state.job);
  clearTimeout(pollTimer);
  pollTimer = setTimeout(pollJob, 1000);
}

async function pollJob() {
  const j = state.job;
  if (!j) return;
  try {
    const r = await api.job(j.id);
    state.job = { ...j, ...r };
    bus.emit('job', state.job);
    if (r.status === 'done') {
      state.job = null;
      bus.emit('job', null);
      toast(JOB_DONE[j.type] || 'Η εργασία ολοκληρώθηκε.', { type: 'success' });
      await refreshAll();
      bus.emit('data-changed', { type: j.type });
      return;
    }
    if (r.status === 'error') {
      state.job = null;
      bus.emit('job', null);
      toast(`Η εργασία απέτυχε: ${r.error || r.message_el || 'άγνωστο σφάλμα'}`, { type: 'error' });
      return;
    }
  } catch (e) {
    if (e.status === 404) {
      state.job = null;
      bus.emit('job', null);
      return;
    }
    // προσωρινό σφάλμα δικτύου: συνεχίζουμε
  }
  pollTimer = setTimeout(pollJob, 1000);
}

export async function refreshAll() {
  clearStockCache();
  await Promise.allSettled([loadStatus(), loadUniverse()]);
  await loadFavorites();
}

export function resumeJobsFromStatus() {
  const jobs = jobList(state.status && state.status.jobs);
  const running = jobs.find((j) => j && (j.status === 'running' || j.status === 'queued'));
  if (running && (!state.job || state.job.id !== running.id)) attachJob(running);
}
