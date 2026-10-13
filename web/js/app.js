// ATHEX Orb — εκκίνηση, δρομολόγηση (location.hash), γραμμή κορυφής, θέμα, συρτάρι μετοχών.
import { html, setHTML, storage, isDrawerLayout, isTypingTarget, debounce } from './dom.js';
import { icon } from './icons.js';
import * as F from './format.js';
import { state, bus, loadStatus, loadUniverse, loadFavorites, startJob, resumeJobsFromStatus, JOB_LABEL } from './store.js';
import { initSidebar } from './sidebar.js';
import { toast } from './toast.js';
import { applyTheme, effectiveTheme, setThemePref, themeBus, initMotion } from './theme.js';
import * as overview from './views/overview.js';
import * as stock from './views/stock.js';
import * as signals from './views/signals.js';
import * as backtest from './views/backtest.js';
import * as journal from './views/journal.js';
import * as settings from './views/settings.js';

const ROUTES = {
  '': { view: overview, title: 'Αγορά', nav: '' },
  stock: { view: stock, title: 'Μετοχή', nav: '' },
  signals: { view: signals, title: 'Σήματα', nav: 'signals' },
  backtest: { view: backtest, title: 'Backtest', nav: 'backtest' },
  journal: { view: journal, title: 'Ημερολόγιο', nav: 'journal' },
  settings: { view: settings, title: 'Ρυθμίσεις', nav: 'settings' },
};

const NAV = [
  { key: '', href: '#/', label: 'Αγορά', icon: 'market' },
  { key: 'signals', href: '#/signals', label: 'Σήματα', icon: 'signals' },
  { key: 'backtest', href: '#/backtest', label: 'Backtest', icon: 'backtest' },
  { key: 'journal', href: '#/journal', label: 'Ημερολόγιο', icon: 'journal' },
  { key: 'settings', href: '#/settings', label: 'Ρυθμίσεις', icon: 'settings' },
];

const el = {
  view: document.getElementById('view'),
  main: document.getElementById('main'),
  topnav: document.getElementById('topnav'),
  tabbar: document.getElementById('tabbar'),
  sidebar: document.getElementById('sidebar'),
  scrim: document.getElementById('scrim'),
  openSidebar: document.getElementById('open-sidebar'),
  pill: document.getElementById('status-pill'),
  jobBtn: document.getElementById('job-btn'),
  themeBtn: document.getElementById('theme-btn'),
  demoBadge: document.getElementById('demo-badge'),
  topbar: document.querySelector('.topbar'),
  jobStrip: document.getElementById('job-strip'),
};

// ---------------------------------------------------------------------------
// Θέμα
// ---------------------------------------------------------------------------
function renderThemeBtn(t) {
  setHTML(el.themeBtn, icon(t === 'light' ? 'moon' : 'sun'));
  const label = t === 'light' ? 'Εναλλαγή σε σκούρο θέμα' : 'Εναλλαγή σε φωτεινό θέμα';
  el.themeBtn.setAttribute('aria-label', label);
  el.themeBtn.title = label;
}
el.themeBtn.addEventListener('click', () => setThemePref(effectiveTheme() === 'light' ? 'dark' : 'light'));
themeBus.on('theme', renderThemeBtn);
initMotion();
renderThemeBtn(applyTheme());

// ---------------------------------------------------------------------------
// Πλοήγηση
// ---------------------------------------------------------------------------
setHTML(el.topnav, NAV.map((n) => html`<a href="${n.href}" data-nav="${n.key}" title="${n.label}">${icon(n.icon)}<span>${n.label}</span></a>`));
setHTML(el.tabbar, NAV.map((n) => html`<a href="${n.href}" data-nav="${n.key}">${icon(n.icon)}<span>${n.label}</span></a>`));
setHTML(el.openSidebar, icon('search'));

function markNav(key) {
  document.querySelectorAll('[data-nav]').forEach((a) => {
    if (a.dataset.nav === key) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}

export function navigate(hash) {
  if (location.hash === hash) route({ soft: true });
  else location.hash = hash;
}

// ---------------------------------------------------------------------------
// Συρτάρι μετοχών (κινητό/tablet)
// ---------------------------------------------------------------------------
let drawerOpen = false;
let lastFocus = null;
function setInert(on) {
  for (const node of [el.main, el.tabbar, el.topbar]) {
    if (!node) continue;
    if (on) node.setAttribute('inert', ''); else node.removeAttribute('inert');
  }
}
function openDrawer() {
  if (!isDrawerLayout()) { sidebar.focus(); return; }
  drawerOpen = true;
  lastFocus = document.activeElement;
  el.sidebar.classList.add('is-open');
  el.scrim.classList.add('is-open');
  el.openSidebar.setAttribute('aria-expanded', 'true');
  el.sidebar.setAttribute('role', 'dialog');
  el.sidebar.setAttribute('aria-modal', 'true');
  document.body.style.overflow = 'hidden';
  setInert(true);
  setTimeout(() => sidebar.focus(), 60);
}
function closeDrawer() {
  if (!drawerOpen) return;
  drawerOpen = false;
  el.sidebar.classList.remove('is-open');
  el.scrim.classList.remove('is-open');
  el.openSidebar.setAttribute('aria-expanded', 'false');
  el.sidebar.removeAttribute('role');
  el.sidebar.removeAttribute('aria-modal');
  document.body.style.overflow = '';
  setInert(false);
  if (lastFocus && lastFocus.focus && document.contains(lastFocus)) lastFocus.focus({ preventScroll: true });
}
el.openSidebar.addEventListener('click', openDrawer);
el.scrim.addEventListener('click', closeDrawer);
// παγίδα εστίασης: με ανοιχτό συρτάρι το Tab κυκλοφορεί μόνο μέσα του (όχι στο body ή στη σελίδα από πίσω)
el.sidebar.addEventListener('keydown', (e) => {
  if (!drawerOpen || e.key !== 'Tab') return;
  const items = Array.from(el.sidebar.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'))
    .filter((n) => !n.disabled && !n.hidden && n.offsetParent !== null && n.tabIndex >= 0);
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  if (e.shiftKey && (document.activeElement === first || !el.sidebar.contains(document.activeElement))) {
    e.preventDefault(); last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault(); first.focus();
  }
});
window.addEventListener('resize', debounce(() => { if (!isDrawerLayout() && drawerOpen) closeDrawer(); }, 120));

const sidebar = initSidebar(el.sidebar, { navigate, closeDrawer });

// Συντομεύσεις πληκτρολογίου
document.addEventListener('keydown', (e) => {
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === '/' && !isTypingTarget(e.target)) {
    e.preventDefault();
    if (isDrawerLayout()) openDrawer(); else sidebar.focus();
  } else if (e.key === 'Escape' && drawerOpen) {
    closeDrawer();
  }
});

// ---------------------------------------------------------------------------
// Ένδειξη κατάστασης δεδομένων & κουμπί εργασιών
// ---------------------------------------------------------------------------
function renderStatus() {
  const s = state.status;
  const pill = el.pill;
  pill.classList.remove('is-demo', 'is-empty', 'is-offline');
  if (!s) {
    setHTML(pill, html`<span class="dot"></span><span class="pill-text">Σύνδεση…</span>`);
    return;
  }
  const m = s.mode || 'empty';
  const when = s.data_last_date ? F.date(s.data_last_date) : null;
  const title = [
    s.data_updated_at ? `Ενημερώθηκε ${F.ago(s.data_updated_at)}` : null,
    s.model_trained_until ? `Μοντέλο εκπαιδευμένο έως ${F.date(s.model_trained_until)}` : null,
    F.isNum(s.n_with_data) ? `${F.int(s.n_with_data)}/${F.int(s.n_tickers)} μετοχές με δεδομένα` : null,
  ].filter(Boolean).join(' · ');
  pill.title = title;
  if (m === 'empty' || !when) {
    pill.classList.add('is-empty');
    setHTML(pill, html`<span class="dot"></span><span class="pill-text">Χωρίς δεδομένα</span>`);
  } else if (m === 'demo') {
    pill.classList.add('is-demo');
    setHTML(pill, html`<span class="dot"></span><span class="pill-text"><span class="pill-label">Δεδομένα: </span>${when} · <b>ΕΠΙΔΕΙΞΗ</b></span>`);
  } else {
    setHTML(pill, html`<span class="dot"></span><span class="pill-text"><span class="pill-label">Δεδομένα: </span>${when}</span>`);
  }
  el.demoBadge.hidden = m !== 'demo';
}

function renderJob() {
  const j = state.job;
  const b = el.jobBtn;
  const strip = el.jobStrip;
  if (j && (j.status === 'running' || j.status === 'queued')) {
    const p = Math.max(0, Math.min(1, F.toNum(j.progress) ?? 0));
    const msg = j.message_el || 'Σε εξέλιξη…';
    // χωρίς μετρήσιμη πρόοδο ακόμη (π.χ. λήψη από το Yahoo): κινούμενη ένδειξη αντί για ένα «κολλημένο» 0%
    const indeterminate = p < 0.02;
    b.classList.toggle('is-indeterminate', indeterminate);
    if (strip) {
      strip.hidden = false;
      strip.classList.toggle('is-indeterminate', indeterminate);
      strip.querySelector('.js-bar i').style.width = `${(Math.max(p, 0.02) * 100).toFixed(1)}%`;
      strip.querySelector('.js-msg').textContent = `${JOB_LABEL[j.type] || 'Εργασία'}: ${msg}${indeterminate ? '' : ` · ${Math.round(p * 100)}%`}`;
    }
    document.querySelectorAll('[data-job]').forEach((btn) => {
      if (btn.dataset.job === j.type) btn.setAttribute('aria-busy', 'true');
      btn.disabled = true;
    });
    b.classList.add('is-running');
    b.setAttribute('aria-busy', 'true');
    b.setAttribute('aria-label', `${JOB_LABEL[j.type] || 'Εργασία'} σε εξέλιξη: ${msg} ${Math.round(p * 100)}%`);
    b.title = `${JOB_LABEL[j.type] || 'Εργασία'}: ${msg}`;
    const C = 2 * Math.PI * 9;
    setHTML(b, html`<span class="job-fill" style="width:${(p * 100).toFixed(1)}%"></span>
      ${icon('refresh')}
      <svg class="job-ring" viewBox="0 0 22 22" aria-hidden="true"><circle class="bg" cx="11" cy="11" r="9"/><circle class="fg" cx="11" cy="11" r="9" stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="${(C * (1 - Math.max(0.04, p))).toFixed(2)}"/></svg>
      <span class="job-text">${msg}</span><span class="job-pct">${indeterminate ? '' : `${Math.round(p * 100)}%`}</span>`);
  } else {
    if (strip) strip.hidden = true;
    document.querySelectorAll('[data-job]').forEach((btn) => { btn.removeAttribute('aria-busy'); btn.disabled = false; });
    b.classList.remove('is-running', 'is-indeterminate');
    b.removeAttribute('aria-busy');
    b.setAttribute('aria-label', 'Ενημέρωση δεδομένων και σημάτων');
    b.title = 'Λήψη νέων τιμών, επανεκπαίδευση μοντέλου και νέα σήματα';
    setHTML(b, html`${icon('refresh')}<span class="job-label">Ενημέρωση</span>`);
  }
}
el.jobBtn.addEventListener('click', () => {
  if (state.job) {
    toast(`${JOB_LABEL[state.job.type] || 'Εργασία'} σε εξέλιξη: ${state.job.message_el || ''}`, { type: 'info' });
    return;
  }
  startJob('refresh');
});
bus.on('status', renderStatus);
bus.on('job', renderJob);
renderStatus();
renderJob();

// ---------------------------------------------------------------------------
// Πίνακες με οριζόντια κύλιση: σβήσιμο στην άκρη που έχει κι άλλες στήλες
// ---------------------------------------------------------------------------
const scrollRO = new ResizeObserver((entries) => entries.forEach((en) => markScroll(en.target)));
function markScroll(w) {
  const more = w.scrollWidth - w.clientWidth > 2;
  w.classList.toggle('more-right', more && w.scrollLeft + w.clientWidth < w.scrollWidth - 2);
  w.classList.toggle('more-left', more && w.scrollLeft > 2);
}
function watchTables(root) {
  root.querySelectorAll('.table-wrap:not([data-sw])').forEach((w) => {
    w.dataset.sw = '1';
    w.addEventListener('scroll', () => markScroll(w), { passive: true });
    scrollRO.observe(w);
    markScroll(w);
  });
}
new MutationObserver(() => {
  watchTables(el.view);
  if (state.job) renderJob();          // νέα κουμπιά [data-job] σε μια σελίδα: ίδια κατάσταση «απασχολημένο»
}).observe(el.view, { childList: true, subtree: true });

// ---------------------------------------------------------------------------
// Δρομολόγηση
// ---------------------------------------------------------------------------
let cleanups = [];
let routeCtrl = null;
let lastRouteKey = null;

function parseHash() {
  const h = (location.hash || '').replace(/^#\/?/, '');
  const [name, ...rest] = h.split('/');
  let param = rest.length ? rest.join('/') : null;
  try { param = param ? decodeURIComponent(param) : null; } catch { /* αγνοείται */ }
  return { name: name || '', param };
}

function teardown() {
  if (routeCtrl) routeCtrl.abort();
  for (const fn of cleanups.splice(0)) { try { fn(); } catch { /* αγνοείται */ } }
}

async function route({ soft = false } = {}) {
  if (fatalShown) return;
  const r = parseHash();
  const def = ROUTES[r.name];
  if (!def || (r.name === 'stock' && !r.param)) { location.replace('#/'); return; }
  teardown();
  routeCtrl = new AbortController();
  const key = `${r.name}/${r.param || ''}`;
  const same = key === lastRouteKey;
  lastRouteKey = key;
  storage.set('route', location.hash || '#/');
  markNav(def.nav);
  sidebar.setCurrent(r.name === 'stock' ? r.param : null);
  document.title = `${r.name === 'stock' ? F.shortTicker(r.param) : def.title} · ATHEX Orb`;
  if (!soft && !same) window.scrollTo({ top: 0 });
  const ctx = {
    param: r.param,
    signal: routeCtrl.signal,
    onCleanup: (fn) => cleanups.push(fn),
    navigate,
    openDrawer,
    soft: soft || same,
  };
  const page = newPage();
  try {
    await def.view.render(page, ctx);
  } catch (e) {
    if (ctx.signal.aborted) return;
    console.warn(e);
    setHTML(page, html`<div class="card"><div class="empty"><div class="empty-icon" style="color:var(--neg)">${icon('alert')}</div><h2>Κάτι πήγε στραβά</h2><p>${e && e.message ? e.message : 'Άγνωστο σφάλμα.'}</p><div class="actions"><button type="button" class="btn btn-primary" data-retry>${icon('refresh')} Δοκιμή ξανά</button></div></div></div>`);
    page.querySelector('[data-retry]')?.addEventListener('click', () => route({ soft: true }));
  }
}

/** Κάθε σελίδα σχεδιάζεται σε νέο στοιχείο, ώστε οι listeners της να φεύγουν μαζί της. */
function newPage() {
  const page = document.createElement('div');
  page.className = 'view';
  el.view.replaceChildren(page);
  return page;
}

window.addEventListener('hashchange', () => route());
bus.on('data-changed', () => route({ soft: true }));
bus.on('retry-universe', () => {
  if (fatalShown) { boot(); return; }
  loadUniverse().catch((e) => toast(e.message, { type: 'error' }));
});

// ---------------------------------------------------------------------------
// Ο διακομιστής δεν απαντά
// ---------------------------------------------------------------------------
let fatalShown = false;
let retryTimer = null;
function showFatal(err) {
  fatalShown = true;
  teardown();
  markNav(null);
  el.pill.classList.remove('is-demo', 'is-empty');
  el.pill.classList.add('is-offline');
  setHTML(el.pill, html`<span class="dot"></span><span class="pill-text">Εκτός σύνδεσης</span>`);
  if (!state.universe.length) {
    state.universeError = err;
    state.universeLoaded = true;
    bus.emit('universe');
  }
  const origin = location.origin && location.origin.startsWith('http') ? location.origin : 'http://127.0.0.1:8000';
  const page = newPage();
  setHTML(page, html`<div class="fatal"><div class="card">
    <div class="empty-icon" style="width:52px;height:52px;border-radius:16px;display:grid;place-items:center;background:var(--neg-soft);color:var(--neg)">${icon('wifiOff', 'icon-lg')}</div>
    <h1>Ο διακομιστής δεν απαντά</h1>
    <p class="muted">Το ATHEX Orb χρειάζεται τον τοπικό διακομιστή για δεδομένα και προβλέψεις. ${err && err.status ? `(${err.message})` : ''}</p>
    <ol>
      <li>Ανοίξτε ένα τερματικό στον φάκελο του έργου.</li>
      <li>Εκτελέστε: <span class="code">python -m athex_orb serve</span></li>
      <li>Ανοίξτε τη διεύθυνση <span class="code">${origin}</span> — ή πατήστε «Δοκιμή ξανά».</li>
    </ol>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <button type="button" class="btn btn-primary" data-retry>${icon('refresh')} Δοκιμή ξανά</button>
      <span class="subtle" data-countdown>Νέα προσπάθεια αυτόματα σε λίγα δευτερόλεπτα…</span>
    </div>
  </div></div>`);
  page.querySelector('[data-retry]').addEventListener('click', () => boot());
  clearTimeout(retryTimer);
  retryTimer = setTimeout(() => boot({ silent: true }), 5000);
}

// ---------------------------------------------------------------------------
// Εκκίνηση
// ---------------------------------------------------------------------------
let booted = false;
async function boot({ silent = false } = {}) {
  clearTimeout(retryTimer);
  try {
    await loadStatus();
  } catch (e) {
    if (e.network || e.status >= 500 || e.status === 404) {
      if (!silent || !fatalShown) showFatal(e);
      else retryTimer = setTimeout(() => boot({ silent: true }), 5000);
      return;
    }
    toast(e.message, { type: 'error' });
  }
  const wasFatal = fatalShown;
  fatalShown = false;
  el.pill.classList.remove('is-offline');
  renderStatus();
  if (wasFatal) toast('Η σύνδεση με τον διακομιστή αποκαταστάθηκε.', { type: 'success' });
  resumeJobsFromStatus();
  // Τα αγαπημένα πρώτα από το τοπικό αντίγραφο ώστε η λίστα να εμφανιστεί αμέσως.
  const universeP = loadUniverse().catch((e) => toast(`Η λίστα μετοχών δεν φορτώθηκε: ${e.message}`, { type: 'error' }));
  const favP = loadFavorites();
  if (!booted) {
    booted = true;
    if (!location.hash || location.hash === '#') {
      const last = storage.get('route', '#/');
      if (typeof last === 'string' && last.startsWith('#/')) history.replaceState(null, '', last);
    }
  }
  route();
  await Promise.allSettled([universeP, favP]);
}

// Ανανέωση κατάστασης όταν η καρτέλα ξαναγίνεται ορατή.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && !fatalShown && booted) {
    loadStatus().then(resumeJobsFromStatus).catch(() => {});
  }
});

boot();
