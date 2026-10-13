// Επιλογέας μετοχών: αναζήτηση (τοπική + διακομιστή), πλοήγηση με πληκτρολόγιο, αγαπημένα.
import { html, raw, setHTML, debounce, storage, isDrawerLayout } from './dom.js';
import { api } from './api.js';
import { icon } from './icons.js';
import * as F from './format.js';
import { sparkline } from './charts.js';
import { orbDot, normOrb } from './orb.js';
import { localSearch, highlight, normalizeTicker, fold } from './search.js';
import { state, bus, isFav, toggleFavoriteUI, addTicker, prefetchStock } from './store.js';

const SORTS = [
  ['default', 'Προεπιλογή'],
  ['name', 'Όνομα'],
  ['change', 'Μεταβολή ημέρας'],
  ['ptp', 'P(στόχος)'],
  ['ev', 'EV καθαρό'],
];

const cmpDesc = (a, b) => {
  const an = F.isNum(a); const bn = F.isNum(b);
  if (!an && !bn) return 0;
  if (!an) return 1;
  if (!bn) return -1;
  return b - a;
};

function sortItems(list, sort) {
  const s = list.slice();
  switch (sort) {
    case 'name': return s.sort((a, b) => String(a.name || a.ticker).localeCompare(String(b.name || b.ticker), 'el'));
    case 'change': return s.sort((a, b) => cmpDesc(a.change_1d, b.change_1d));
    case 'ptp': return s.sort((a, b) => cmpDesc(a.orb && a.orb.p_tp, b.orb && b.orb.p_tp));
    case 'ev': return s.sort((a, b) => cmpDesc(a.orb && a.orb.ev_net, b.orb && b.orb.ev_net));
    default: return s;
  }
}

export function initSidebar(root, { navigate, closeDrawer }) {
  let query = '';
  let tab = storage.get('sbTab', null);
  let sort = storage.get('sbSort', 'default');
  if (!SORTS.some(([k]) => k === sort)) sort = 'default';
  let active = -1;
  let rows = [];
  let serverItems = [];
  let suggest = null;
  let searchCtrl = null;
  let current = null;
  const idxCache = new WeakMap();

  setHTML(root, html`
    <div class="sidebar-head">
      <h2>Μετοχές</h2>
      <button type="button" class="icon-btn" data-close aria-label="Κλείσιμο λίστας μετοχών">${icon('x')}</button>
    </div>
    <div class="sb-top">
      <div class="search" role="search">
        ${icon('search')}
        <input id="sb-search" type="search" autocomplete="off" autocapitalize="off" spellcheck="false" enterkeyhint="go"
          placeholder="Μετοχή ή σύμβολο, π.χ. ΟΠΑΠ"
          role="combobox" aria-autocomplete="list" aria-expanded="true" aria-controls="sb-list" aria-describedby="sb-help"
          aria-label="Αναζήτηση μετοχής με σύμβολο, όνομα ή κλάδο">
        <kbd class="search-kbd" aria-hidden="true">/</kbd>
        <button type="button" class="search-clear" aria-label="Καθαρισμός αναζήτησης">${icon('x', 'icon-sm')}</button>
      </div>
      <div class="seg sb-tabs" role="group" aria-label="Ποιες μετοχές εμφανίζονται">
        <button type="button" data-tab="fav">${icon('starFill', 'icon-sm')} Αγαπημένα <span class="count" data-count="fav"></span></button>
        <button type="button" data-tab="all">Όλες <span class="count" data-count="all"></span></button>
      </div>
    </div>
    <div class="sb-caption"><span data-caption aria-hidden="true"></span>
      <label class="sort-select" title="Ταξινόμηση">
        <span class="sr-only">Ταξινόμηση λίστας</span>
        ${icon('sort', 'icon-sm')}
        <select data-sort>${SORTS.map(([k, l]) => html`<option value="${k}">${l}</option>`)}</select>
      </label>
    </div>
    <div class="sb-list" id="sb-list" role="listbox" aria-label="Μετοχές"></div>
    <div class="sb-foot" id="sb-help">
      <span><kbd>/</kbd> αναζήτηση</span>
      <span><kbd>↑</kbd><kbd>↓</kbd> επιλογή</span>
      <span><kbd>Enter</kbd> άνοιγμα</span>
      <span><kbd>⇧</kbd><kbd>Enter</kbd> αγαπημένο</span>
    </div>`);

  const input = root.querySelector('#sb-search');
  const searchBox = root.querySelector('.search');
  const listEl = root.querySelector('#sb-list');
  const sortSel = root.querySelector('[data-sort]');
  const captionL = root.querySelector('[data-caption]');
  sortSel.value = sort;

  // ---------------------------------------------------------------------
  function effectiveTab() {
    if (tab === 'fav' || tab === 'all') return tab;
    return state.favorites.length ? 'fav' : 'all';
  }

  function addCandidate(q) {
    const t = (suggest && suggest.ticker) || normalizeTicker(q);
    if (!t || state.byTicker.has(t)) return null;
    if (suggest && suggest.valid === false) return null;
    if (!/^[\x20-\x7E]+$/.test(q)) return null;
    return t;
  }

  function compute() {
    const q = query.trim();
    let list;
    if (q) {
      list = localSearch(state.universe, q, idxCache);
      const have = new Set(list.map((i) => i.ticker));
      for (const it of serverItems) {
        if (!it || !it.ticker || have.has(it.ticker)) continue;
        list.push(state.byTicker.get(it.ticker) || it);
        have.add(it.ticker);
      }
    } else if (effectiveTab() === 'fav') {
      list = sortItems(state.favorites.map((t) => state.byTicker.get(t)).filter(Boolean), sort);
    } else {
      list = sortItems(state.universe, sort);
    }
    rows = list.map((item) => ({ type: 'item', item }));
    const add = q ? addCandidate(q) : null;
    if (add) rows.push({ type: 'add', ticker: add });
  }

  function rowHTML(r, i) {
    const q = query.trim();
    if (r.type === 'add') {
      return html`<div class="stock-row add-row ${i === active ? 'is-active' : ''}" role="option" id="sbo-${i}" aria-selected="${i === active}" data-i="${i}">
        <span class="star" aria-hidden="true">${icon('plus', 'icon-sm')}</span>
        <div class="sr-main"><div class="sr-tick">Προσθήκη «${r.ticker}» στη λίστα</div>
        <div class="sr-name">Λήψη δεδομένων και προσθήκη στα αγαπημένα</div></div>
      </div>`;
    }
    const it = r.item;
    const fav = isFav(it.ticker);
    const short = F.shortTicker(it.ticker);
    const o = normOrb(it.orb);
    const name = it.name || short;
    return html`<div class="stock-row ${i === active ? 'is-active' : ''} ${it.ticker === current ? 'is-current' : ''}" role="option" id="sbo-${i}" aria-selected="${i === active}" data-i="${i}" data-ticker="${it.ticker}" title="${name} · ${it.sector || ''}${o ? ` · ${F.STATES[o.dominant].label} ${F.prob(o[`p_${o.dominant}`])}` : ''}">
      <button type="button" class="star" tabindex="-1" data-star aria-pressed="${fav}" aria-label="${fav ? `Αφαίρεση ${name} από τα αγαπημένα` : `Προσθήκη ${name} στα αγαπημένα`}">${icon(fav ? 'starFill' : 'star', 'icon-sm')}</button>
      <div class="sr-main">
        <div class="sr-tick">${orbDot(it.orb)}<span class="mono">${raw(highlight(short, q))}</span>${it.custom ? html`<span class="tag-custom">ΔΙΚΗ ΣΑΣ</span>` : ''}</div>
        <div class="sr-name">${raw(highlight(name, q))}${it.has_data === false ? ' · χωρίς δεδομένα' : ''}</div>
      </div>
      <span class="sr-spark">${sparkline(it.spark, { w: 54, h: 22 })}</span>
      <div class="sr-px"><span class="price">${F.price(it.last_close, { unit: false })}</span>${F.change(it.change_1d)}</div>
    </div>`;
  }

  function emptyHTML() {
    const q = query.trim();
    if (state.universeError && !state.universe.length) {
      return html`<div class="sb-empty"><div class="empty-icon" style="color:var(--neg)">${icon('alert')}</div><h3>Η λίστα δεν φορτώθηκε</h3><p>${state.universeError.message}</p><button type="button" class="btn btn-sm" data-retry>${icon('refresh')} Δοκιμή ξανά</button></div>`;
    }
    if (q) {
      return html`<div class="sb-empty"><div class="empty-icon" style="color:var(--text-3)">${icon('search')}</div><h3>Καμία μετοχή για «${q}»</h3><p>Δοκιμάστε σύμβολο (π.χ. <span class="mono">OPAP</span>, <span class="mono">PPC</span>), ελληνικό ή λατινικό όνομα (π.χ. ΔΕΗ) ή κλάδο. Για νέα μετοχή γράψτε το σύμβολό της με λατινικούς χαρακτήρες.</p></div>`;
    }
    if (effectiveTab() === 'fav') {
      return html`<div class="sb-empty"><div class="empty-icon">${icon('starFill')}</div><h3>Δεν έχετε αγαπημένες ακόμη</h3><p>Πατήστε το ☆ δίπλα σε μια μετοχή — ή αναζητήστε την παραπάνω — για να την καρφιτσώσετε εδώ.</p><button type="button" class="btn btn-sm" data-show-all>Δείτε όλες τις μετοχές</button></div>`;
    }
    return html`<div class="sb-empty"><h3>Η λίστα είναι άδεια</h3><p>Αναζητήστε ένα σύμβολο για να το προσθέσετε.</p></div>`;
  }

  function skeletonHTML() {
    return Array.from({ length: 9 }, () => html`<div class="sk-row" aria-hidden="true"><div class="sk" style="width:18px;height:18px;border-radius:6px;margin-left:6px"></div><div style="display:grid;gap:6px"><div class="sk sk-line" style="width:46%"></div><div class="sk sk-line" style="width:72%;height:10px"></div></div><div class="sk" style="height:18px"></div><div style="display:grid;gap:6px;justify-items:end"><div class="sk sk-line" style="width:52px"></div><div class="sk sk-line" style="width:40px;height:10px"></div></div></div>`);
  }

  function render() {
    const q = query.trim();
    compute();
    if (active >= rows.length) active = rows.length - 1;
    const tabNow = effectiveTab();
    root.querySelectorAll('[data-tab]').forEach((b) => b.setAttribute('aria-pressed', String(!q && b.dataset.tab === tabNow)));
    root.querySelector('[data-count="fav"]').textContent = state.favorites.length ? F.int(state.favorites.length) : '';
    root.querySelector('[data-count="all"]').textContent = state.universe.length ? F.int(state.universe.length) : '';
    searchBox.classList.toggle('has-value', !!query);
    const nItems = rows.filter((r) => r.type === 'item').length;
    captionL.textContent = q ? (nItems ? `${F.int(nItems)} ${nItems === 1 ? 'αποτέλεσμα' : 'αποτελέσματα'} σε όλες τις μετοχές` : 'Κανένα αποτέλεσμα') : (tabNow === 'fav' ? 'Αγαπημένες μετοχές' : 'Όλες οι μετοχές');
    root.querySelector('.sort-select').hidden = !!q;

    if (!state.universeLoaded) setHTML(listEl, skeletonHTML());
    else if (!rows.length) setHTML(listEl, emptyHTML());
    else setHTML(listEl, rows.map(rowHTML));
    syncActive(false);
  }

  function syncActive(scroll = true) {
    listEl.querySelectorAll('.stock-row.is-active').forEach((el) => { el.classList.remove('is-active'); el.setAttribute('aria-selected', 'false'); });
    const el = active >= 0 ? listEl.querySelector(`#sbo-${active}`) : null;
    if (el) {
      el.classList.add('is-active');
      el.setAttribute('aria-selected', 'true');
      input.setAttribute('aria-activedescendant', el.id);
      if (scroll) el.scrollIntoView({ block: 'nearest' });
      const r = rows[active];
      if (r && r.type === 'item' && scroll) prefetchSoon(r.item.ticker);
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  let prefetchTimer = null;
  function prefetchSoon(ticker) {
    clearTimeout(prefetchTimer);
    prefetchTimer = setTimeout(() => prefetchStock(ticker), 120);
  }

  async function openRow(i) {
    const r = rows[i];
    if (!r) return;
    if (r.type === 'add') {
      const t = await addTicker(r.ticker, { favorite: true });
      if (t) {
        query = '';
        input.value = '';
        serverItems = [];
        suggest = null;
        active = -1;
        render();
        navigate(`#/stock/${t}`);
        if (isDrawerLayout()) closeDrawer();
      }
      return;
    }
    navigate(`#/stock/${r.item.ticker}`);
    if (isDrawerLayout()) closeDrawer();
  }

  function toggleStar(ticker, btn) {
    toggleFavoriteUI(ticker);
    requestAnimationFrame(() => {
      const el = listEl.querySelector(`[data-ticker="${CSS.escape(ticker)}"] .star`) || btn;
      if (el) { el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop'); }
    });
  }

  const serverSearch = debounce(async (q) => {
    if (searchCtrl) searchCtrl.abort();
    searchCtrl = new AbortController();
    try {
      const r = await api.search(q, { signal: searchCtrl.signal });
      if (fold(q) !== fold(query.trim())) return;
      serverItems = Array.isArray(r && r.items) ? r.items : [];
      suggest = r && r.suggest && r.suggest.ticker ? r.suggest : null;
      render();
    } catch {
      /* ακυρώθηκε ή σφάλμα: κρατάμε τα τοπικά αποτελέσματα */
    }
  }, 150);

  // ---------------------------------------------------------------------
  input.addEventListener('input', () => {
    // ένα «/» στην αρχή (η συντόμευση πατήθηκε ενώ το πεδίο είχε ήδη εστίαση) δεν είναι μέρος της αναζήτησης
    if (input.value.startsWith('/')) input.value = input.value.replace(/^\/+/, '');
    query = input.value;
    serverItems = [];
    suggest = null;
    active = query.trim() ? 0 : -1;
    render();
    if (query.trim()) serverSearch(query.trim());
    else { serverSearch.cancel(); if (searchCtrl) searchCtrl.abort(); }
  });

  input.addEventListener('keydown', (e) => {
    if (e.isComposing) return;
    const n = rows.length;
    if (e.key === '/' && !input.value) { e.preventDefault(); input.select(); return; }
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (n) { active = Math.min(n - 1, active + 1); syncActive(); }
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (n) { active = Math.max(0, active - 1); syncActive(); }
        break;
      case 'PageDown':
        e.preventDefault();
        if (n) { active = Math.min(n - 1, Math.max(0, active) + 8); syncActive(); }
        break;
      case 'PageUp':
        e.preventDefault();
        if (n) { active = Math.max(0, active - 8); syncActive(); }
        break;
      case 'Enter': {
        e.preventDefault();
        if (!n) return;
        const i = active < 0 ? 0 : active;
        const r = rows[i];
        if (e.shiftKey && r && r.type === 'item') toggleStar(r.item.ticker);
        else openRow(i);
        break;
      }
      case 'Escape':
        if (query) {
          e.preventDefault();
          query = '';
          input.value = '';
          serverItems = [];
          suggest = null;
          active = -1;
          render();
        } else if (isDrawerLayout()) {
          e.preventDefault();
          closeDrawer();
        } else {
          input.blur();
        }
        break;
      default:
    }
  });

  root.querySelector('.search-clear').addEventListener('click', () => {
    query = '';
    input.value = '';
    serverItems = [];
    suggest = null;
    active = -1;
    render();
    input.focus();
  });

  root.querySelectorAll('[data-tab]').forEach((b) => b.addEventListener('click', () => {
    tab = b.dataset.tab;
    storage.set('sbTab', tab);
    if (query) { query = ''; input.value = ''; serverItems = []; suggest = null; }
    active = -1;
    render();
    listEl.scrollTop = 0;
  }));

  sortSel.addEventListener('change', () => {
    sort = sortSel.value;
    storage.set('sbSort', sort);
    render();
  });

  root.querySelector('[data-close]').addEventListener('click', () => closeDrawer());

  listEl.addEventListener('click', (e) => {
    if (e.target.closest('[data-retry]')) {
      bus.emit('retry-universe');
      return;
    }
    if (e.target.closest('[data-show-all]')) {
      tab = 'all';
      storage.set('sbTab', tab);
      render();
      return;
    }
    const row = e.target.closest('.stock-row');
    if (!row) return;
    const i = +row.dataset.i;
    const star = e.target.closest('[data-star]');
    if (star && row.dataset.ticker) {
      e.stopPropagation();
      toggleStar(row.dataset.ticker, star);
      return;
    }
    active = i;
    syncActive(false);
    openRow(i);
  });

  listEl.addEventListener('pointerover', (e) => {
    if (e.pointerType === 'touch') return;
    const row = e.target.closest('.stock-row');
    if (!row) return;
    const i = +row.dataset.i;
    if (i !== active) { active = i; syncActive(false); }
    if (row.dataset.ticker) prefetchSoon(row.dataset.ticker);
  });

  bus.on('universe', render);
  bus.on('favorites', render);
  render();

  return {
    focus() { input.focus(); input.select(); },
    setCurrent(ticker) {
      current = ticker;
      listEl.querySelectorAll('.stock-row.is-current').forEach((el) => el.classList.remove('is-current'));
      if (ticker) {
        const el = listEl.querySelector(`[data-ticker="${CSS.escape(ticker)}"]`);
        if (el) el.classList.add('is-current');
      }
    },
    render,
  };
}
