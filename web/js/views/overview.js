// Αγορά: κλίμα, βασικοί δείκτες, αγαπημένα και πίνακας όλων των μετοχών.
import { html, setHTML, storage } from '../dom.js';
import { icon } from '../icons.js';
import * as F from '../format.js';
import { sparkline } from '../charts.js';
import { orbHTML, normOrb, orbDot } from '../orb.js';
import { state, bus, prefetchStock } from '../store.js';
import { disclaimer, onboardingCard, wireCommon, starButton, kpi, skeletonPage } from './common.js';

// hide-md: κρύβεται σε μεσαία πλάτη (με την πλαϊνή λίστα ο πίνακας δεν χωρά) — τα βασικά μένουν πάντα ορατά
const COLS = [
  { key: 'name', label: 'Μετοχή' },
  { key: 'sector', label: 'Κλάδος', cls: 'hide-sm hide-md' },
  { key: 'price', label: 'Τιμή', cls: 'r' },
  { key: 'change_1d', label: 'Ημέρα', cls: 'r hide-sm' },
  { key: 'change_5d', label: '5 ημ.', cls: 'r hide-sm hide-md' },
  { key: null, label: '60 ημ.', cls: 'hide-sm hide-md' },
  { key: 'p_tp', label: 'P(στόχος)', cls: 'r', title: 'Πιθανότητα να χτυπήσει πρώτα ο στόχος (η λωρίδα: στόχος · λήξη · stop)' },
  { key: 'ev', label: 'EV καθαρό', cls: 'r hide-sm' },
  { key: 'decision', label: 'Απόφαση', cls: 'hide-sm' },
];

const DEC_ORDER = { take: 0, skip: 1, blocked: 2, none: 3 };

function sortVal(it, key) {
  switch (key) {
    case 'name': return String(it.name || it.ticker);
    case 'sector': return String(it.sector || '');
    case 'price': return it.last_close;
    case 'change_1d': return it.change_1d;
    case 'change_5d': return it.change_5d;
    case 'p_tp': return it.orb ? it.orb.p_tp : null;
    case 'ev': return it.orb ? it.orb.ev_net : null;
    case 'decision': return DEC_ORDER[it.decision] ?? 9;
    default: return null;
  }
}

function sortList(list, { key, dir }) {
  if (!key) return list;
  const mul = dir === 'asc' ? 1 : -1;
  return list.slice().sort((a, b) => {
    const va = sortVal(a, key); const vb = sortVal(b, key);
    if (typeof va === 'string' || typeof vb === 'string') return mul * String(va ?? '').localeCompare(String(vb ?? ''), 'el');
    const an = F.isNum(va); const bn = F.isNum(vb);
    if (!an && !bn) return 0;
    if (!an) return 1;
    if (!bn) return -1;
    return mul * (va - vb);
  });
}

/** Λωρίδα τριών πιθανοτήτων (στόχος / λήξη / stop) με αριθμό την P(στόχος). */
export function probStrip(orb) {
  const o = normOrb(orb);
  if (!o) return html`<span class="subtle">—</span>`;
  const title = `Στόχος πρώτα ${F.prob(o.p_tp)} · Λήξη ${F.prob(o.p_to)} · Stop πρώτα ${F.prob(o.p_sl)}`;
  return html`<span class="pstrip-wrap" title="${title}"><span class="pstrip" aria-hidden="true"><i style="width:${(o.p_tp * 100).toFixed(1)}%;background:var(--pos-line)"></i><i style="width:${(o.p_to * 100).toFixed(1)}%;background:var(--to-line)"></i><i style="width:${(o.p_sl * 100).toFixed(1)}%;background:var(--neg-line)"></i></span><b class="num">${F.prob(o.p_tp)}</b><span class="sr-only"> πιθανότητα στόχου πρώτα</span></span>`;
}

function favCard(it) {
  const o = normOrb(it.orb);
  const name = it.name || F.shortTicker(it.ticker);
  return html`<article class="card fav-card">
    <div class="fc-top">
      <div class="fc-names"><a class="fc-link" href="#/stock/${encodeURIComponent(it.ticker)}" data-prefetch="${it.ticker}"><span class="fc-t mono">${F.shortTicker(it.ticker)}</span></a><div class="fc-n">${name}</div></div>
      ${starButton(it.ticker, name)}
    </div>
    <div class="fc-mid">
      <div style="min-width:0"><div class="fc-price">${F.price(it.last_close)}</div>${F.change(it.change_1d)}</div>
      ${orbHTML(it.orb, { size: 54, mini: true, caption: true })}
    </div>
    <div class="fc-spark">${sparkline(it.spark, { w: 300, h: 40, area: true })}</div>
    <div class="fc-foot">
      <span>P(στόχος) <b>${o ? F.prob(o.p_tp) : '—'}</b></span>
      <span>EV <b class="${F.tone(o && o.ev_net)}">${o && F.isNum(o.ev_net) ? F.pct(o.ev_net, { digits: 2, sign: true }) : '—'}</b></span>
      ${F.decisionChip(it.decision, { small: true })}
    </div>
  </article>`;
}

function tableRow(it) {
  const name = it.name || F.shortTicker(it.ticker);
  const o = normOrb(it.orb);
  return html`<tr class="clickable" data-href="#/stock/${encodeURIComponent(it.ticker)}" data-prefetch="${it.ticker}">
    <td style="width:44px;padding-right:0">${starButton(it.ticker, name)}</td>
    <td><div class="cell-stock">${orbDot(it.orb)}<a class="names" href="#/stock/${encodeURIComponent(it.ticker)}"><span class="t mono">${F.shortTicker(it.ticker)}${it.custom ? html` <span class="tag-custom" style="font-family:var(--font-sans)">ΔΙΚΗ ΣΑΣ</span>` : ''}</span><span class="n">${name}</span></a></div></td>
    <td class="hide-sm hide-md muted">${it.sector || '—'}</td>
    <td class="r"><span class="num" style="font-weight:550">${F.price(it.last_close)}</span><span class="show-sm">${F.change(it.change_1d)}</span></td>
    <td class="r hide-sm">${F.change(it.change_1d)}</td>
    <td class="r hide-sm hide-md">${F.change(it.change_5d)}</td>
    <td class="hide-sm hide-md" style="width:90px">${sparkline(it.spark, { w: 80, h: 24 })}</td>
    <td class="r">${probStrip(it.orb)}</td>
    <td class="r hide-sm"><span class="num ${F.tone(o && o.ev_net)}" style="font-weight:600">${o && F.isNum(o.ev_net) ? F.pct(o.ev_net, { digits: 2, sign: true }) : '—'}</span></td>
    <td class="hide-sm">${F.decisionChip(it.decision, { small: true })}</td>
  </tr>`;
}

export async function render(page, ctx) {
  let tsort = storage.get('ovSort', { key: null, dir: 'desc' });
  if (!tsort || typeof tsort !== 'object') tsort = { key: null, dir: 'desc' };

  const draw = () => {
    const focusKey = document.activeElement && page.contains(document.activeElement) ? (document.activeElement.dataset.fav || document.activeElement.dataset.sortKey) : null;
    const s = state.status || {};
    const m = s.mode || 'empty';
    if (!state.universeLoaded) { setHTML(page, skeletonPage({ kpis: 4, rows: 8 })); return; }
    const items = state.universe;
    const withData = items.filter((i) => i.has_data !== false && F.isNum(i.last_close));
    const takes = items.filter((i) => i.decision === 'take').sort((a, b) => ((b.orb && b.orb.ev_net) || 0) - ((a.orb && a.orb.ev_net) || 0));
    const up5 = withData.filter((i) => F.isNum(i.change_5d) && i.change_5d > 0).length;
    const dn5 = withData.filter((i) => F.isNum(i.change_5d) && i.change_5d < 0).length;
    const ch1 = withData.map((i) => i.change_1d).filter(F.isNum);
    const avg1 = ch1.length ? ch1.reduce((a, b) => a + b, 0) / ch1.length : null;
    const regimeText = s.regime_el || s.market_regime_el || null;
    const horizon = s.horizon_days || 10;
    const lastDate = s.data_last_date || state.asOf;

    const head = html`<div class="page-head">
      <div>
        <div class="eyebrow">Χρηματιστήριο Αθηνών</div>
        <h1>Αγορά</h1>
        <p class="lede">Πιθανότητες για τις επόμενες ${horizon} συνεδριάσεις${lastDate ? html` · δεδομένα έως <b>${F.date(lastDate)}</b>` : ''}</p>
      </div>
      <div class="spacer"></div>
      ${withData.length ? html`<div class="regime">
        ${regimeText ? html`<span class="chip chip-info">${icon('compass', 'icon-sm')} Καθεστώς: ${regimeText}</span>` : ''}
        <span class="chip chip-outline" title="Μετοχές με θετική/αρνητική απόδοση 5 ημερών">
          <span class="breadth" aria-hidden="true"><i style="width:${(up5 / Math.max(1, withData.length) * 100).toFixed(1)}%;background:var(--pos-line)"></i><i style="width:${(dn5 / Math.max(1, withData.length) * 100).toFixed(1)}%;background:var(--neg-line)"></i></span>
          ${F.int(up5)} ▲ · ${F.int(dn5)} ▼ (5 ημ.)
        </span>
        ${avg1 != null ? html`<span class="chip chip-outline">Μέση μεταβολή ημέρας ${F.change(avg1)}</span>` : ''}
      </div>` : ''}
    </div>`;

    const demo = m === 'demo' ? html`<div class="notice notice-warn" role="note">${icon('alert')}<div><strong>Λειτουργία επίδειξης.</strong> Οι τιμές είναι συνθετικές και δεν αντιστοιχούν σε πραγματικές συναλλαγές. Πατήστε «Ενημέρωση» για πραγματικά δεδομένα.</div></div>` : '';

    if (m === 'empty' || !withData.length) {
      setHTML(page, html`${head}${demo}${onboardingCard()}${disclaimer()}`);
      return;
    }

    const kpis = html`<div class="kpis">
      ${kpi({ label: 'Μετοχές με δεδομένα', iconName: 'layers', value: html`${F.int(s.n_with_data ?? withData.length)}<small> / ${F.int(s.n_tickers ?? items.length)}</small>`, sub: 'στη λίστα παρακολούθησης' })}
      ${kpi({ label: 'Υποψήφιες αγορές σήμερα', iconName: 'signals', value: F.int(takes.length), valueClass: takes.length ? 'pos' : '', sub: takes.length ? html`${takes.slice(0, 3).map((t, i) => html`${i ? ' · ' : ''}<a href="#/stock/${encodeURIComponent(t.ticker)}" class="mono">${F.shortTicker(t.ticker)}</a>`)}${takes.length > 3 ? ` +${takes.length - 3}` : ''}` : 'Καμία σήμερα — κανένα σήμα δεν πέρασε τα φίλτρα' })}
      ${kpi({ label: 'Τελευταία δεδομένα', iconName: 'database', value: F.date(lastDate), sub: s.data_updated_at ? `Ενημερώθηκε ${F.ago(s.data_updated_at)}` : '' })}
      ${kpi({ label: 'Μοντέλο εκπαιδευμένο έως', iconName: 'cpu', value: F.date(s.model_trained_until), sub: html`<span title="ATR = μέσο ημερήσιο εύρος τιμής (14 συνεδριάσεων). Ο στόχος μπαίνει ${F.num(s.k_tp ?? 2, 1)} φορές το ATR πάνω από την είσοδο, το stop ${F.num(s.k_sl ?? 1, 1)} φορά κάτω.">Στόχος ${F.num(s.k_tp ?? 2, 1)} × ATR · Stop ${F.num(s.k_sl ?? 1, 1)} × ATR <span class="subtle">(ATR = μέσο ημερήσιο εύρος)</span></span>` })}
    </div>`;

    const favItems = state.favorites.map((t) => state.byTicker.get(t)).filter(Boolean);
    const favSection = html`<section aria-labelledby="fav-title">
      <div class="section-title"><h2 id="fav-title">★ Αγαπημένα</h2><span class="sub">${favItems.length ? `${F.int(favItems.length)} μετοχές` : ''}</span><span class="spacer"></span>
        <button type="button" class="btn btn-sm btn-ghost" data-open-search>${icon('plus')} Προσθήκη</button></div>
      ${favItems.length ? html`<div class="fav-grid">${favItems.map(favCard)}</div>` : html`<div class="card"><div class="empty" style="padding:30px 20px">
          <div class="empty-icon" style="color:#f5b301">${icon('starFill')}</div>
          <h3>Καρφιτσώστε τις μετοχές που σας ενδιαφέρουν</h3>
          <p>Πατήστε το ☆ σε οποιαδήποτε μετοχή — στη λίστα, στον πίνακα παρακάτω ή στη σελίδα της — και θα εμφανίζεται εδώ με τη σφαίρα της.</p>
          <div class="actions"><button type="button" class="btn" data-open-search>${icon('search')} Αναζήτηση μετοχής</button></div></div></div>`}
    </section>`;

    const sorted = sortList(items, tsort);
    const th = (c) => {
      if (!c.key) return html`<th class="${c.cls || ''}" scope="col">${c.label}</th>`;
      const on = tsort.key === c.key;
      const ariaSort = on ? (tsort.dir === 'asc' ? 'ascending' : 'descending') : null;
      return html`<th class="${c.cls || ''}" scope="col" ${ariaSort ? html`aria-sort="${ariaSort}"` : ''}><button type="button" class="th-sort" data-sort-key="${c.key}" ${c.title ? html`title="${c.title}"` : ''}>${c.label}${icon(on ? (tsort.dir === 'asc' ? 'chevronDown' : 'chevronDown') : 'sort', 'icon-sm')}</button></th>`;
    };
    const table = html`<section aria-labelledby="all-title">
      <div class="section-title"><h2 id="all-title">Όλες οι μετοχές</h2><span class="sub">${F.int(items.length)} μετοχές · πατήστε μια γραμμή για λεπτομέρειες</span></div>
      <div class="card"><div class="table-wrap"><table class="table">
        <thead><tr><th scope="col" style="width:44px;padding-right:0"><span class="sr-only">Αγαπημένο</span></th>${COLS.map(th)}</tr></thead>
        <tbody>${sorted.map(tableRow)}</tbody>
      </table></div></div>
    </section>`;

    setHTML(page, html`${head}${demo}${kpis}${favSection}${table}${disclaimer()}`);
    page.querySelectorAll('th[aria-sort="ascending"] .th-sort .icon-sm').forEach((i) => { i.style.transform = 'rotate(180deg)'; });
    if (focusKey) {
      const target = page.querySelector(`[data-fav="${CSS.escape(focusKey)}"], [data-sort-key="${CSS.escape(focusKey)}"]`);
      if (target) target.focus({ preventScroll: true });
    }
  };

  wireCommon(page);
  page.addEventListener('click', (e) => {
    const sortBtn = e.target.closest('[data-sort-key]');
    if (sortBtn) {
      const key = sortBtn.dataset.sortKey;
      if (tsort.key === key) {
        if (tsort.dir === 'desc') tsort = { key, dir: 'asc' };
        else tsort = { key: null, dir: 'desc' };
      } else {
        tsort = { key, dir: key === 'name' || key === 'sector' || key === 'decision' ? 'asc' : 'desc' };
      }
      storage.set('ovSort', tsort);
      draw();
      return;
    }
    if (e.target.closest('[data-open-search]')) { ctx.openDrawer(); return; }
    if (e.target.closest('a, button')) return;
    const tr = e.target.closest('tr[data-href]');
    if (tr) ctx.navigate(tr.dataset.href);
    const card = e.target.closest('.fav-card');
    if (card) { const a = card.querySelector('.fc-link'); if (a) ctx.navigate(a.getAttribute('href')); }
  });
  page.addEventListener('pointerover', (e) => {
    const p = e.target.closest('[data-prefetch]');
    if (p) prefetchStock(p.dataset.prefetch);
  });

  const offs = [bus.on('universe', draw), bus.on('favorites', draw), bus.on('status', draw)];
  ctx.onCleanup(() => offs.forEach((f) => f()));
  draw();
}

