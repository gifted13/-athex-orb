// Επικοινωνία με τον διακομιστή (FastAPI). Όλα τα σφάλματα γίνονται ApiError με ελληνικό μήνυμα.

export class ApiError extends Error {
  constructor(message, { status = 0, network = false, detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.network = network;
    this.detail = detail;
  }
}

const DEFAULT_TIMEOUT = 20000;

function detailMessage(data, status) {
  const d = data && data.detail;
  if (typeof d === 'string' && d) return d;
  if (Array.isArray(d) && d.length) return d.map((x) => x && (x.msg || x.message)).filter(Boolean).join(' · ') || `Μη έγκυρο αίτημα (${status}).`;
  if (data && typeof data.error_el === 'string') return data.error_el;
  if (status === 404) return 'Δεν βρέθηκε.';
  if (status === 409) return 'Η εργασία εκτελείται ήδη.';
  if (status >= 500) return `Σφάλμα διακομιστή (${status}).`;
  return `Το αίτημα απέτυχε (${status}).`;
}

async function request(method, path, { body, signal, timeout = DEFAULT_TIMEOUT } = {}) {
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, timeout);
  const onAbort = () => ctrl.abort();
  if (signal) {
    if (signal.aborted) ctrl.abort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
      cache: 'no-store',
    });
  } catch (err) {
    clearTimeout(timer);
    if (signal && signal.aborted) throw err; // ακυρώθηκε σκόπιμα
    throw new ApiError(timedOut ? 'Ο διακομιστής δεν απάντησε εγκαίρως.' : 'Δεν υπάρχει σύνδεση με τον διακομιστή.', { network: true });
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort);
  }
  clearTimeout(timer);
  let data = null;
  const text = await res.text().catch(() => '');
  if (text) {
    try { data = JSON.parse(text); } catch { data = null; }
  }
  if (!res.ok) throw new ApiError(detailMessage(data, res.status), { status: res.status, detail: data && data.detail });
  return data;
}

const enc = encodeURIComponent;

export const api = {
  status: (o) => request('GET', '/api/status', o),
  universe: (o) => request('GET', '/api/universe', o),
  search: (q, o) => request('GET', `/api/search?q=${enc(q)}`, o),
  addStock: (body) => request('POST', '/api/universe', { body }),
  removeStock: (t) => request('DELETE', `/api/universe/${enc(t)}`),
  favorites: (o) => request('GET', '/api/favorites', o),
  addFavorite: (t) => request('PUT', `/api/favorites/${enc(t)}`),
  removeFavorite: (t) => request('DELETE', `/api/favorites/${enc(t)}`),
  stock: (t, days = 250, o) => request('GET', `/api/stock/${enc(t)}?days=${days}`, o),
  signals: (o) => request('GET', '/api/signals', o),
  backtest: (o) => request('GET', '/api/backtest', { timeout: 45000, ...o }),
  journal: (o) => request('GET', '/api/journal', o),
  settings: (o) => request('GET', '/api/settings', o),
  saveSettings: (body) => request('PUT', '/api/settings', { body }),
  startJob: (type) => request('POST', '/api/jobs', { body: { type } }),
  job: (id, o) => request('GET', `/api/jobs/${enc(id)}`, o),
  jobs: (o) => request('GET', '/api/jobs', o),
};

/** Το /api/jobs μπορεί να επιστρέφει λίστα ή {items|jobs: [...]}. */
export function jobList(data) {
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.items)) return data.items;
  if (data && Array.isArray(data.jobs)) return data.jobs;
  return [];
}
