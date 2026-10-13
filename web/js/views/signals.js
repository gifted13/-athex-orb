// Σήματα: οι σημερινές αποφάσεις (αγορές πρώτα), με φίλτρα.
import { html, setHTML, storage } from '../dom.js';
import { icon } from '../icons.js';
import * as F from '../format.js';
import { api } from '../api.js';
import { state, bus, isFav, prefetchStock } from '../store.js';
import { disclaimer, wireCommon, errorCard, emptyState, skeletonPage, kpi, noDataCard, starButton } from './common.js';
import { orbDot } from '../orb.js';

function sortSignals(items) {
  return items.slice().sort((a, b) => {
    const ta = a.decision === 'take' ? 0 : 1;
    const tb = b.decision === 'take' ? 0 : 1;
    if (ta !== tb) return ta - tb;
    const ea = F.isNum(a.ev_net) ? a.ev_net : -Infinity;
    const eb = F.isNum(b.ev_net) ? b.ev_net : -Infinity;
    return eb - ea;
  });
}

function row(s) {
  const name = s.name || (state.byTicker.get(s.ticker) || {}).name || F.shortTicker(s.ticker);
  const item = state.byTicker.get(s.ticker);
  const orb = item ? item.orb : { p_tp: s.p_tp, p_sl: s.p_sl, p_to: s.p_to };
  const tpR = F.isNum(s.tp_price) && F.isNum(s.entry_ref) ? s.tp_price / s.entry_ref - 1 : null;
  const slR = F.isNum(s.sl_price) && F.isNum(s.entry_ref) ? s.sl_price / s.entry_ref - 1 : null;
  // στο κινητό οι απορρίψεις/μπλοκ γίνονται συμπαγείς κάρτες (μετοχή, P(στόχος), EV, αιτία)
  const compact = s.decision !== 'take';
  const noSize = !F.isNum(s.size_eur);
  return html`<tr class="clickable ${compact ? 'row-compact' : ''}" data-href="#/stock/${encodeURIComponent(s.ticker)}" data-prefetch="${s.ticker}">
    <td class="card-head-cell"><div class="cell-stock">${starButton(s.ticker, name)}${orbDot(orb)}<a class="names" href="#/stock/${encodeURIComponent(s.ticker)}"><span class="t mono">${F.shortTicker(s.ticker)}</span><span class="n">${name}</span></a></div><span class="reason-inline">${s.reason_el || ''}</span><span class="show-sm">${F.decisionChip(s.decision, { small: true })}</span></td>
    <td class="hide-sm">${F.decisionChip(s.decision, { small: true })}</td>
    <td class="r" data-label="P(στόχος)"><span class="num pos" style="font-weight:600">${F.prob(s.p_tp)}</span></td>
    <td class="r hide-md cm-hide" data-label="P(stop)"><span class="num neg" style="font-weight:600">${F.prob(s.p_sl)}</span></td>
    <td class="r" data-label="EV καθαρό"><span class="num ${F.tone(s.ev_net)}" style="font-weight:650">${F.pct(s.ev_net, { digits: 2, sign: true })}</span></td>
    <td class="r hide-md cm-hide" data-label="Είσοδος"><span class="num">${F.price(s.entry_ref)}</span></td>
    <td class="r cm-hide" data-label="Στόχος"><span class="num">${F.price(s.tp_price)}</span>${tpR != null ? html` <span class="subtle num" style="font-size:11.5px">${F.pct(tpR, { digits: 1, sign: true })}</span>` : ''}</td>
    <td class="r cm-hide" data-label="Stop"><span class="num">${F.price(s.sl_price)}</span>${slR != null ? html` <span class="subtle num" style="font-size:11.5px">${F.pct(slR, { digits: 1, sign: true })}</span>` : ''}</td>
    <td class="r ${noSize ? 'sm-hide' : ''}" data-label="Ποσό"><span class="num">${noSize ? '—' : F.eur(s.size_eur, { digits: 0 })}</span></td>
    <td class="span-2 muted reason-cell hide-lg" data-label="Αιτία" title="${s.reason_el || ''}"><span class="clamp-2">${s.reason_el || '—'}</span></td>
  </tr>`;
}

export async function render(page, ctx) {
  let filter = storage.get('sigFilter', 'all') === 'take' ? 'take' : 'all';
  let onlyFav = !!storage.get('sigFav', false);
  let data = null;

  const draw = () => {
    const s = state.status || {};
    if (!data) return;
    const all = sortSignals((Array.isArray(data.items) ? data.items : []).filter((x) => x && x.ticker));
    const head = html`<div class="page-head">
      <div><div class="eyebrow">Εικονικές συναλλαγές</div><h1>Σήματα</h1>
        <p class="lede">Οι αποφάσεις για την επόμενη συνεδρίαση${data.as_of ? html` · σήμα της <b>${F.date(data.as_of)}</b>` : ''}. Καμία εντολή δεν εκτελείται.</p></div>
    </div>`;
    if (!all.length) {
      setHTML(page, html`${head}${s.mode === 'empty' ? noDataCard() : html`<div class="card">${emptyState({ iconName: 'signals', title: 'Δεν υπάρχουν σήματα', text: 'Δεν υπάρχουν αποφάσεις για την τελευταία συνεδρίαση. Κάντε μια ενημέρωση για νέους υπολογισμούς.', actions: html`<button type="button" class="btn btn-primary" data-job="refresh">${icon('refresh')} Ενημέρωση</button>` })}</div>`}${disclaimer()}`);
      return;
    }
    const takes = all.filter((x) => x.decision === 'take');
    const blocked = all.filter((x) => x.decision === 'blocked').length;
    const skips = all.filter((x) => x.decision === 'skip').length;
    const avgEv = takes.length ? takes.reduce((a, b) => a + (F.isNum(b.ev_net) ? b.ev_net : 0), 0) / takes.length : null;
    const totalSize = takes.reduce((a, b) => a + (F.isNum(b.size_eur) ? b.size_eur : 0), 0);
    let list = filter === 'take' ? takes : all;
    if (onlyFav) list = list.filter((x) => isFav(x.ticker));

    const kpis = html`<div class="kpis">
      ${kpi({ label: 'Υποψήφιες αγορές', iconName: 'signals', value: F.int(takes.length), valueClass: takes.length ? 'pos' : '', sub: `από ${F.int(all.length)} αξιολογήσεις` })}
      ${kpi({ label: 'Χωρίς αγορά', iconName: 'shield', value: html`${F.int(skips)}<small> απορρίψεις</small> · ${F.int(blocked)}<small> μπλοκ</small>`, sub: 'απόρριψη: χαμηλό EV ή όριο χαρτοφυλακίου · μπλοκ: έλεγχος Φύλακα' })}
      ${kpi({ label: 'Μέσο EV αγορών', iconName: 'sigma', value: avgEv == null ? '—' : F.pct(avgEv, { digits: 2, sign: true }), valueClass: F.tone(avgEv), sub: 'καθαρό, μετά τα κόστη' })}
      ${kpi({ label: 'Συνολικό ποσό', iconName: 'wallet', value: totalSize ? F.eur(totalSize, { digits: 0 }) : '—', sub: 'αν εκτελεστούν όλες (εικονικά)' })}
    </div>`;

    const filters = html`<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <div class="seg" role="group" aria-label="Φίλτρο απόφασης">
        <button type="button" data-filter="take" aria-pressed="${filter === 'take'}">Μόνο αγορές <span class="count">${F.int(takes.length)}</span></button>
        <button type="button" data-filter="all" aria-pressed="${filter === 'all'}">Όλες <span class="count">${F.int(all.length)}</span></button>
      </div>
      <label class="toggle"><input type="checkbox" data-onlyfav ${onlyFav ? 'checked' : ''}> Μόνο αγαπημένα</label>
    </div>`;

    const table = list.length ? html`<div class="card"><div class="table-wrap"><table class="table table-cards">
      <thead><tr><th scope="col">Μετοχή</th><th scope="col">Απόφαση</th><th scope="col" class="r">P(στόχος)</th><th scope="col" class="r hide-md">P(stop)</th><th scope="col" class="r">EV καθαρό</th><th scope="col" class="r hide-md">Είσοδος</th><th scope="col" class="r">Στόχος</th><th scope="col" class="r">Stop</th><th scope="col" class="r">Ποσό</th><th scope="col" class="hide-lg">Αιτία</th></tr></thead>
      <tbody>${list.map((x, i) => html`${filter === 'all' && x.decision !== 'take' && (i === 0 || list[i - 1].decision === 'take') ? html`<tr class="group-row"><td colspan="10">Απορρίψεις & μπλοκ · ${F.int(list.length - list.filter((y) => y.decision === 'take').length)}</td></tr>` : ''}${row(x)}`)}</tbody></table></div></div>`
      : html`<div class="card">${emptyState({ iconName: onlyFav ? 'star' : 'signals', title: onlyFav ? 'Κανένα σήμα στα αγαπημένα' : 'Καμία υποψήφια αγορά σήμερα', text: onlyFav ? 'Καμία από τις αγαπημένες σας μετοχές δεν ταιριάζει στο φίλτρο. Απενεργοποιήστε το «Μόνο αγαπημένα» για να δείτε όλες.' : 'Κανένα σήμα δεν πέρασε τους ελέγχους και το ελάχιστο EV. Αυτό είναι φυσιολογικό — το φίλτρο είναι σκόπιμα αυστηρό.' })}</div>`;

    setHTML(page, html`${head}${kpis}${filters}${table}${disclaimer()}`);
  };

  wireCommon(page, { onRetry: () => load() });
  page.addEventListener('click', (e) => {
    const f = e.target.closest('[data-filter]');
    if (f) { filter = f.dataset.filter; storage.set('sigFilter', filter); draw(); return; }
    if (e.target.closest('a, button, input, label')) return;
    const tr = e.target.closest('tr[data-href]');
    if (tr) ctx.navigate(tr.dataset.href);
  });
  page.addEventListener('change', (e) => {
    if (e.target.matches('[data-onlyfav]')) { onlyFav = e.target.checked; storage.set('sigFav', onlyFav); draw(); }
  });
  page.addEventListener('pointerover', (e) => {
    const p = e.target.closest('[data-prefetch]');
    if (p) prefetchStock(p.dataset.prefetch);
  });
  const offs = [bus.on('favorites', () => draw())];
  ctx.onCleanup(() => offs.forEach((f) => f()));

  async function load() {
    setHTML(page, skeletonPage({ kpis: 4, rows: 8 }));
    try {
      data = await api.signals({ signal: ctx.signal });
      if (ctx.signal.aborted) return;
      data = data && typeof data === 'object' ? data : { items: [] };
      draw();
    } catch (e) {
      if (ctx.signal.aborted) return;
      setHTML(page, html`<div class="page-head"><div><h1>Σήματα</h1></div></div>${errorCard(e)}`);
    }
  }
  await load();
}
