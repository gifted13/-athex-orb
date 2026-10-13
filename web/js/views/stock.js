// Σελίδα μετοχής: απόφαση, Σφαίρα με τους έξι πράκτορες, γράφημα τιμής με σχέδιο, ιστορικό, εκβάσεις.
import { html, setHTML, storage, isMobile } from '../dom.js';
import { icon, AGENT_ICONS } from '../icons.js';
import * as F from '../format.js';
import { priceChart, orbAreaChart } from '../charts.js';
import { orbHTML, orbLegend, normOrb, tint } from '../orb.js';
import { state, bus, isFav, toggleFavoriteUI, cachedStock, fetchStock, addTicker, removeTicker } from '../store.js';
import { normalizeTicker } from '../search.js';
import { disclaimer, wireCommon, errorCard, emptyState, noDataCard } from './common.js';

const STATUS = {
  ok: { chip: 'chip-pos', text: 'Εντάξει', icon: 'check' },
  warn: { chip: 'chip-warn', text: 'Προσοχή', icon: 'alert' },
  block: { chip: 'chip-neg', text: 'Μπλοκ', icon: 'ban' },
  info: { chip: 'chip-info', text: 'Πληροφορία', icon: 'info' },
};
const AGENT_ORDER = ['scout', 'market', 'judge', 'quant', 'guard', 'executor'];
// δύο στήλες με παρόμοιο ύψος: αριστερά δεδομένα, αγορά, μοντέλο και σχέδιο· δεξιά οικονομικά και έλεγχοι
const LEFT = new Set(['scout', 'market', 'judge', 'executor']);

function addBusinessDays(d, n) {
  const x = new Date(d.getTime());
  while (n > 0) {
    x.setDate(x.getDate() + 1);
    const wd = x.getDay();
    if (wd !== 0 && wd !== 6) n--;
  }
  return x;
}

function metricsGrid(metrics) {
  const list = (Array.isArray(metrics) ? metrics : []).filter((m) => m && m.label_el).slice(0, 4);
  if (!list.length) return '';
  return html`<div class="metrics">${list.map((m) => {
    const v = F.metric(m.value, m.fmt);
    const t = m.fmt === 'pct' && F.isNum(m.value) && /EV|απόδοση|vs/i.test(m.label_el) ? F.tone(m.value) : '';
    return html`<div class="metric" ${m.hint_el ? html`title="${m.hint_el}"` : ''}><div class="m-label">${m.label_el}</div><div class="m-value ${t}">${v}</div></div>`;
  })}</div>`;
}

function checksList(checks) {
  const list = (Array.isArray(checks) ? checks : []).filter(Boolean);
  if (!list.length) return '';
  return html`<ul class="checks" aria-label="Έλεγχοι κινδύνου">${list.map((c) => {
    const cls = c.passed ? 'ck-ok' : c.blocking === false ? 'ck-soft' : 'ck-fail';
    const ic = c.passed ? 'check' : c.blocking === false ? 'alert' : 'x';
    const sr = c.passed ? 'Πέρασε' : c.blocking === false ? 'Προειδοποίηση' : 'Απέτυχε';
    return html`<li><span class="ck ${cls}">${icon(ic)}<span class="sr-only">${sr}: </span></span><span class="ck-label">${c.label_el || c.id}</span>${c.detail_el ? html`<span class="ck-detail">${c.detail_el}</span>` : ''}</li>`;
  })}</ul>`;
}

function driversList(drivers, base) {
  const list = (Array.isArray(drivers) ? drivers : []).filter((d) => d && F.isNum(d.contribution)).slice(0, 6);
  if (!list.length) return html`<p class="subtle" style="font-size:12.5px">Δεν υπάρχουν διαθέσιμοι παράγοντες για αυτό το μοντέλο.</p>`;
  const b = base && F.isNum(base.contribution) ? base : null;
  const max = Math.max(...list.map((d) => Math.abs(d.contribution)), b ? Math.abs(b.contribution) : 0, 1e-9);
  const bar = (v) => {
    const w = (Math.abs(v) / max) * 50;
    return v >= 0 ? `left:50%;width:${w.toFixed(1)}%;background:var(--pos-line)` : `right:50%;width:${w.toFixed(1)}%;background:var(--neg-line)`;
  };
  const row = (label, title, v, cls = '') => html`<div class="dr ${cls}" role="listitem"><span class="dr-label" title="${title}">${label}</span><span class="dr-bar" aria-hidden="true"><i style="${bar(v)}"></i></span><span class="dr-val ${v >= 0 ? 'pos' : 'neg'}">${F.num(v, 2, { sign: true })}</span></div>`;
  return html`<div class="drivers" role="list" aria-label="Κύριοι παράγοντες της πρόβλεψης">
    <div class="drivers-legend" aria-hidden="true"><span>← υπέρ stop</span><span>υπέρ στόχου →</span></div>
    ${b ? row(b.label_el || 'Βάση', 'Πού ξεκινά το μοντέλο για μια μέση περίπτωση (ιστορικές συχνότητες), πριν από τους παράγοντες', b.contribution, 'dr-base') : ''}
    ${list.map((d) => row(d.short_el || d.label_el || d.feature, d.label_el || d.feature, d.contribution))}
  </div>`;
}

function planBlock(plan, asOf) {
  if (!plan || !F.isNum(plan.entry_ref)) return '';
  const e = plan.entry_ref;
  const tpR = F.isNum(plan.tp_price) ? plan.tp_price / e - 1 : null;
  const slR = F.isNum(plan.sl_price) ? plan.sl_price / e - 1 : null;
  const rr = tpR != null && slR != null && slR < 0 ? tpR / -slR : null;
  const h = F.isNum(plan.horizon_days) ? plan.horizon_days : null;
  const base = F.parseDate(asOf);
  const until = base && h ? addBusinessDays(base, h) : null;
  return html`<div class="plan" aria-label="Σχέδιο θέσης">
    <div class="pl"><span>${icon('play', 'icon-sm')} Είσοδος</span><b>${F.price(e)}<small>αναφορά</small></b></div>
    <div class="pl"><span><span class="sw" style="background:var(--pos-line)"></span> Στόχος</span><b>${F.price(plan.tp_price)}<small class="pos">${tpR != null ? F.pct(tpR, { digits: 1, sign: true }) : ''}</small></b></div>
    <div class="pl"><span><span class="sw" style="background:var(--neg-line)"></span> Stop</span><b>${F.price(plan.sl_price)}<small class="neg">${slR != null ? F.pct(slR, { digits: 1, sign: true }) : ''}</small></b></div>
    <div class="pl"><span>${icon('clock', 'icon-sm')} Λήξη</span><b>${h ? `${F.int(h)} συνεδριάσεις` : '—'}<small>${until ? `~${F.date(until, 'short')}` : ''}</small></b></div>
    <div class="pl"><span>${icon('wallet', 'icon-sm')} Ποσό</span><b>${F.isNum(plan.size_eur) ? F.eur(plan.size_eur, { digits: 0 }) : '—'}<small>${F.isNum(plan.shares) ? `${F.num(plan.shares, plan.shares % 1 ? 2 : 0)} τεμ.` : ''}</small></b></div>
    ${rr != null ? html`<div class="pl"><span>${icon('scale', 'icon-sm')} Απόδοση / κίνδυνος</span><b>${F.num(rr, 1)} : 1</b></div>` : ''}
  </div>`;
}

function agentCard(a, d) {
  const id = AGENT_ORDER.includes(a.id) ? a.id : 'other';
  const st = STATUS[a.status] || STATUS.info;
  let extra = '';
  let metrics = metricsGrid(a.metrics);
  if (a.id === 'guard') extra = checksList(a.checks);
  if (a.id === 'judge') extra = driversList(a.drivers, a.drivers_base);
  if (a.id === 'executor' && d.plan) { extra = planBlock(d.plan, d.as_of); metrics = ''; }
  const notes = (Array.isArray(a.notes_el) ? a.notes_el : []).filter(Boolean);
  return html`<article class="card agent agent--${STATUS[a.status] ? a.status : 'info'}" data-agent="${id}" aria-labelledby="ag-${id}">
    <div class="agent-head">
      <span class="agent-icon" aria-hidden="true">${icon(AGENT_ICONS[a.id] || 'info')}</span>
      <div style="min-width:0"><h3 class="agent-name" id="ag-${id}">${a.name_el || a.id}</h3><div class="agent-role">${a.role_el || ''}</div></div>
      <span class="chip chip-sm ${st.chip}">${icon(st.icon, 'icon-sm')} ${st.text}</span>
    </div>
    ${a.headline_el ? html`<p class="agent-headline">${a.headline_el}</p>` : ''}
    ${a.id === 'judge' ? html`${metrics}${extra}` : html`${extra}${metrics}`}
    ${notes.length ? html`<details class="notes"><summary>Σημειώσεις (${notes.length}) ${icon('chevronDown', 'icon-sm')}</summary><ul>${notes.map((n) => html`<li>${n}</li>`)}</ul></details>` : ''}
  </article>`;
}

function decisionBanner(d) {
  const k = F.DECISIONS[d.decision] ? d.decision : 'none';
  const dec = F.DECISIONS[k];
  const reason = d.decision_reason_el || (k === 'none' ? 'Δεν υπάρχει πρόβλεψη για την τελευταία συνεδρίαση.' : '');
  return html`<div class="decision decision--${k}" role="status">
    <div class="d-icon" aria-hidden="true">${icon(dec.icon)}</div>
    <div style="min-width:0"><div class="d-title">${dec.label}</div>${reason ? html`<div class="d-reason">${reason}</div>` : ''}</div>
    <div class="d-side">${k === 'take' ? 'Εικονική συναλλαγή · δεν εκτελείται εντολή' : 'Απόφαση για την επόμενη συνεδρίαση'}${d.as_of ? html`<br>Σήμα της ${F.date(d.as_of)}` : ''}</div>
  </div>`;
}

function outcomesBlock(outcomes) {
  const list = (Array.isArray(outcomes) ? outcomes : []).filter(Boolean).slice().sort((a, b) => String(a.signal_date).localeCompare(String(b.signal_date)));
  if (!list.length) return emptyState({ iconName: 'clock', title: 'Χωρίς εκβάσεις ακόμη', text: 'Μόλις επιλυθούν τα πρώτα σήματα αυτής της μετοχής (στόχος, stop ή λήξη), θα εμφανιστούν εδώ.' });
  const counts = { tp: 0, sl: 0, to: 0, open: 0 };
  let sum = 0; let nRet = 0;
  for (const o of list) {
    counts[F.labelKey(o.label)]++;
    if (F.isNum(o.gross_ret)) { sum += o.gross_ret; nRet++; }
  }
  const resolved = counts.tp + counts.sl + counts.to;
  const recent = list.slice().reverse().slice(0, 5);
  return html`<div>
    <div class="outcome-strip" role="img" aria-label="Τελευταίες ${list.length} εκβάσεις: ${counts.tp} στόχος, ${counts.sl} stop, ${counts.to} λήξη">
      ${list.map((o) => { const k = F.labelKey(o.label); return html`<i class="${F.STATES[k].cls}" title="${F.date(o.signal_date)} · ${F.STATES[k].label} · ${F.pct(o.gross_ret, { digits: 1, sign: true })}">${F.STATES[k].glyph}</i>`; })}
    </div>
    <div class="outcomes-sum">
      <span><span class="sw" style="background:var(--pos-line)"></span> Στόχος <b>${F.int(counts.tp)}</b></span>
      <span><span class="sw" style="background:var(--to-line)"></span> Λήξη <b>${F.int(counts.to)}</b></span>
      <span><span class="sw" style="background:var(--neg-line)"></span> Stop <b>${F.int(counts.sl)}</b></span>
      <span>Ποσοστό στόχου <b>${resolved ? F.prob(counts.tp / resolved) : '—'}</b></span>
      <span>Μέση μικτή απόδοση <b class="${F.tone(nRet ? sum / nRet : null)}">${nRet ? F.pct(sum / nRet, { digits: 2, sign: true }) : '—'}</b></span>
    </div>
    <table class="table" style="margin-top:12px">
      <thead><tr><th scope="col">Σήμα</th><th scope="col">Έκβαση</th><th scope="col" class="r">Μικτή απόδοση</th></tr></thead>
      <tbody>${recent.map((o) => html`<tr><td>${F.date(o.signal_date)}</td><td>${F.stateChip(o.label)}</td><td class="r"><span class="num ${F.tone(o.gross_ret)}" style="font-weight:600">${F.pct(o.gross_ret, { digits: 2, sign: true })}</span></td></tr>`)}</tbody>
    </table>
  </div>`;
}

function skeleton() {
  return html`<div aria-busy="true" aria-label="Φόρτωση μετοχής…">
    <div style="display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap"><div style="flex:1;display:grid;gap:10px"><div class="sk" style="width:240px;height:30px;max-width:80%"></div><div class="sk" style="width:180px;height:14px"></div></div><div class="sk" style="width:130px;height:36px"></div></div>
    <div class="sk" style="height:66px;border-radius:16px;margin-top:20px"></div>
    <div class="stage" style="margin-top:20px"><div class="stage-grid">
      <div class="card orb-card" style="min-height:330px;align-items:center"><div class="sk" style="width:220px;height:220px;border-radius:50%;margin:auto"></div></div>
      ${[AGENT_ORDER.filter((a) => LEFT.has(a)), AGENT_ORDER.filter((a) => !LEFT.has(a))].map((col, ci) => html`<div class="agents-col agents-col--${ci ? 'r' : 'l'}">${col.map((a) => html`<div class="card agent" data-agent="${a}"><div style="display:flex;gap:10px"><div class="sk" style="width:34px;height:34px;border-radius:10px"></div><div style="flex:1;display:grid;gap:6px"><div class="sk sk-line" style="width:50%"></div><div class="sk sk-line" style="width:70%;height:10px"></div></div></div><div class="sk sk-line" style="width:90%"></div><div class="sk sk-line" style="width:60%"></div></div>`)}</div>`)}
    </div></div>
  </div>`;
}

export async function render(page, ctx) {
  const ticker = ctx.param;
  let data = ctx.soft ? null : cachedStock(ticker);
  let mode = storage.get('chartMode', null);
  if (mode !== 'line' && mode !== 'candles') mode = isMobile() ? 'line' : 'candles';
  let range = storage.get('chartRange', '6m') === '12m' ? '12m' : '6m';
  let priceCleanup = null;
  let areaCleanup = null;
  ctx.onCleanup(() => { if (priceCleanup) priceCleanup(); if (areaCleanup) areaCleanup(); });

  const mountPrice = () => {
    if (priceCleanup) priceCleanup();
    const host = page.querySelector('[data-chart="price"]');
    if (!host || !data) return;
    const candles = Array.isArray(data.candles) ? data.candles : [];
    const n = range === '6m' ? 126 : 260;
    priceCleanup = priceChart(host, { candles: candles.slice(-n), plan: data.plan, mode, title: `Τιμή ${F.shortTicker(ticker)}` });
    page.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    page.querySelectorAll('[data-range]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.range === range)));
  };

  const syncFav = () => {
    const b = page.querySelector('[data-stock-fav]');
    if (!b) return;
    const fav = isFav(ticker);
    const label = fav ? 'Στα αγαπημένα' : 'Προσθήκη στα αγαπημένα';
    b.setAttribute('aria-pressed', String(fav));
    b.setAttribute('aria-label', label);          // στο κινητό το κείμενο κρύβεται (μόνο το αστέρι)
    b.title = fav ? 'Αφαίρεση από τα αγαπημένα' : 'Προσθήκη στα αγαπημένα';
    setHTML(b, html`${icon(fav ? 'starFill' : 'star')}<span>${label}</span>`);
  };

  const draw = (d) => {
    if (areaCleanup) { areaCleanup(); areaCleanup = null; }
    const item = state.byTicker.get(ticker) || {};
    const name = d.name || item.name || F.shortTicker(ticker);
    const sector = d.sector || item.sector || '';
    const candles = Array.isArray(d.candles) ? d.candles : [];
    const last = candles[candles.length - 1];
    const prev = candles[candles.length - 2];
    const lastClose = last && F.isNum(last.c) ? last.c : item.last_close;
    const chg = last && prev && F.isNum(prev.c) && prev.c ? last.c / prev.c - 1 : item.change_1d;
    const o = normOrb(d.orb);
    const agents = (Array.isArray(d.agents) ? d.agents : []).filter((a) => a && typeof a === 'object');
    agents.sort((a, b) => (AGENT_ORDER.indexOf(a.id) + 99) % 99 - (AGENT_ORDER.indexOf(b.id) + 99) % 99);
    const hasData = d.has_data !== false && candles.length > 0;
    const horizon = (d.plan && d.plan.horizon_days) || (state.status && state.status.horizon_days) || 10;
    document.title = `${F.shortTicker(ticker)} · ${name} · ATHEX Orb`;

    const head = html`<a class="back-link" href="#/">${icon('chevronLeft', 'icon-sm')} Αγορά</a>
    <div class="stock-head">
      <div class="title">
        <h1>${name}</h1>
        <div class="meta"><span class="ticker-badge">${d.ticker || ticker}</span>${sector ? html`<span class="chip chip-outline">${sector}</span>` : ''}${d.custom || item.custom ? html`<span class="chip chip-sm" style="color:var(--violet);background:color-mix(in srgb, var(--violet) 14%, transparent)">Δική σας</span>` : ''}${state.status && state.status.mode === 'demo' ? html`<span class="chip chip-warn chip-sm">ΕΠΙΔΕΙΞΗ</span>` : ''}</div>
      </div>
      ${F.isNum(lastClose) ? html`<div class="quote">
        <div class="price">${F.price(lastClose)}</div>
        <div>${F.change(chg)} <span class="subtle" style="font-size:12px">ημέρα</span></div>
        ${last ? html`<div class="asof">Κλείσιμο ${F.date(last.d)}</div>` : ''}
      </div>` : ''}
      <div class="head-actions">
        <button type="button" class="btn fav-btn" data-stock-fav aria-pressed="false"></button>
        ${d.custom || item.custom ? html`<button type="button" class="btn btn-ghost btn-danger" data-remove title="Αφαίρεση από τη λίστα">${icon('trash')}<span class="sr-only">Αφαίρεση από τη λίστα</span></button>` : ''}
      </div>
    </div>`;

    const base = F.parseDate(d.as_of);
    const until = base ? F.date(addBusinessDays(base, horizon), 'short') : '';
    const stage = html`<section class="stage" aria-label="Η Σφαίρα και οι έξι πράκτορες">
      <div class="stage-grid">
        <div class="card orb-card" style="--orb-tint:${tint(o)}">
          <div class="eyebrow">Η Σφαίρα</div>
          <p class="orb-q">Τι θα συμβεί πρώτα μέσα σε ${horizon} συνεδριάσεις;</p>
          <div class="orb-wrap">${orbHTML(d.orb, { size: isMobile() ? 220 : 244 })}${orbLegend(d.orb, { plan: d.plan, until, horizon })}</div>
        </div>
        <div class="agents-col agents-col--l">${agents.filter((a) => LEFT.has(a.id)).map((a) => agentCard(a, d))}</div>
        <div class="agents-col agents-col--r">${agents.filter((a) => !LEFT.has(a.id)).map((a) => agentCard(a, d))}</div>
      </div>
    </section>`;

    const plan = d.plan && F.isNum(d.plan.entry_ref) ? d.plan : null;
    const charts = hasData ? html`
      <section class="card" aria-labelledby="price-title">
        <div class="card-head">
          <h2 id="price-title">Τιμή & σχέδιο θέσης</h2>
          <span class="spacer"></span>
          <div class="seg" role="group" aria-label="Τύπος γραφήματος"><button type="button" data-mode="candles">Κεριά</button><button type="button" data-mode="line">Γραμμή</button></div>
          <div class="seg" role="group" aria-label="Περίοδος"><button type="button" data-range="6m">6Μ</button><button type="button" data-range="12m">12Μ</button></div>
        </div>
        <div class="card-body">
          ${plan ? html`<div class="legend" style="margin-bottom:10px">
            <span><span class="sw-line" style="background:var(--pos-line)"></span>Στόχος <b>${F.price(plan.tp_price)}</b></span>
            <span><span class="sw-line" style="background:var(--neg-line)"></span>Stop <b>${F.price(plan.sl_price)}</b></span>
            <span><span class="sw-line" style="background:var(--text-2)"></span>Είσοδος <b>${F.price(plan.entry_ref)}</b></span>
            <span class="subtle">Σκιασμένη ζώνη: ορίζοντας ${F.int(plan.horizon_days || horizon)} συνεδριάσεων</span>
          </div>` : ''}
          <div class="chart" data-chart="price"></div>
        </div>
      </section>
      <div class="bt-grid">
        <section class="card" aria-labelledby="hist-title">
          <div class="card-head"><h2 id="hist-title">Ιστορικό σφαίρας</h2><span class="sub">Πώς άλλαζαν οι πιθανότητες</span></div>
          <div class="card-body">
            <div class="legend" style="margin-bottom:10px"><span><span class="sw" style="background:var(--pos-line)"></span>Στόχος πρώτα</span><span><span class="sw" style="background:var(--to-line)"></span>Λήξη χρόνου</span><span><span class="sw" style="background:var(--neg-line)"></span>Stop πρώτα</span></div>
            <div class="chart" data-chart="area"></div>
          </div>
        </section>
        <section class="card" aria-labelledby="out-title">
          <div class="card-head"><h2 id="out-title">Πρόσφατες εκβάσεις</h2><span class="sub">Τελευταία επιλυμένα σήματα</span></div>
          <div class="card-body">${outcomesBlock(d.outcomes)}</div>
        </section>
      </div>` : html`<div class="card">${emptyState({
        iconName: 'database', title: 'Δεν υπάρχουν ακόμη δεδομένα τιμών',
        text: 'Η μετοχή δεν έχει (ακόμη) ιστορικό τιμών. Μια ενημέρωση θα προσπαθήσει να κατεβάσει δεδομένα — αν το σύμβολο δεν υπάρχει στο Yahoo Finance, θα παραλειφθεί.',
        actions: html`<button type="button" class="btn btn-primary" data-job="refresh">${icon('refresh')} Ενημέρωση</button>`,
      })}</div>`;

    const pageNotes = (Array.isArray(d.notes_el) ? d.notes_el : []).filter((n) => typeof n === 'string' && n);
    const notesBox = pageNotes.length ? html`<div class="notice notice-info" role="note">${icon('info')}<div>${pageNotes.map((n, i) => html`${i ? html`<br>` : ''}${n}`)}</div></div>` : '';
    if (state.status && state.status.mode === 'empty') {
      // καθόλου δεδομένα ακόμη: όχι «άδειοι» πράκτορες — οι δύο τρόποι να ξεκινήσει κανείς
      setHTML(page, html`${head}${noDataCard('Δεν υπάρχουν ακόμη δεδομένα τιμών. Κατεβάστε πραγματικά δεδομένα (Yahoo Finance) ή δοκιμάστε την εφαρμογή με δεδομένα επίδειξης — μετά θα δείτε εδώ τη Σφαίρα και τους έξι πράκτορες για τη μετοχή.')}${disclaimer()}`);
      syncFav();
      return;
    }
    setHTML(page, html`${head}${decisionBanner(d)}${notesBox}${stage}${charts}${disclaimer()}`);
    syncFav();
    if (hasData) {
      mountPrice();
      const areaHost = page.querySelector('[data-chart="area"]');
      const n = range === '6m' ? 126 : 260;
      if (areaHost) areaCleanup = orbAreaChart(areaHost, (Array.isArray(d.orb_history) ? d.orb_history : []).slice(-n));
    }
  };

  const notFound = (err) => {
    const t = normalizeTicker(ticker);
    setHTML(page, html`<a class="back-link" href="#/">${icon('chevronLeft', 'icon-sm')} Αγορά</a><div class="card">${emptyState({
      iconName: 'search', tone: 'warn', title: `Η μετοχή «${F.shortTicker(ticker)}» δεν βρέθηκε`,
      text: err && err.message && err.status !== 404 ? err.message : 'Δεν υπάρχει στη λίστα παρακολούθησης. Μπορείτε να την προσθέσετε αν το σύμβολο υπάρχει στο Χρηματιστήριο Αθηνών.',
      actions: html`${t ? html`<button type="button" class="btn btn-primary" data-add="${t}">${icon('plus')} Προσθήκη «${t}»</button>` : ''}<a class="btn" href="#/">Επιστροφή στην αγορά</a>`,
    })}</div>`);
  };

  wireCommon(page, { onRetry: () => load(true) });
  page.addEventListener('click', async (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.hasAttribute('data-stock-fav')) {
      toggleFavoriteUI(ticker);
      b.querySelector('.icon')?.classList.add('pop');
      return;
    }
    if (b.dataset.mode && b.dataset.mode !== mode) { mode = b.dataset.mode; storage.set('chartMode', mode); mountPrice(); return; }
    if (b.dataset.range && b.dataset.range !== range) {
      range = b.dataset.range;
      storage.set('chartRange', range);
      mountPrice();
      const areaHost = page.querySelector('[data-chart="area"]');
      if (areaHost && data) { if (areaCleanup) areaCleanup(); areaCleanup = orbAreaChart(areaHost, (data.orb_history || []).slice(range === '6m' ? -126 : -260)); }
      return;
    }
    if (b.dataset.add) {
      const t = await addTicker(b.dataset.add, { favorite: true });
      if (t) ctx.navigate(`#/stock/${t}`);
      return;
    }
    if (b.hasAttribute('data-remove')) {
      // eslint-disable-next-line no-alert
      if (window.confirm(`Αφαίρεση από τη λίστα παρακολούθησης: ${F.shortTicker(ticker)};`)) {
        if (await removeTicker(ticker)) ctx.navigate('#/');
      }
    }
  });

  const offs = [bus.on('favorites', syncFav)];
  ctx.onCleanup(() => offs.forEach((f) => f()));

  async function load(force = false) {
    if (!data || force) {
      if (!data) setHTML(page, skeleton());
      try {
        const fresh = await fetchStock(ticker);
        if (ctx.signal.aborted) return;
        if (!fresh || typeof fresh !== 'object') throw new Error('Κενή απάντηση από τον διακομιστή.');
        data = fresh;
      } catch (e) {
        if (ctx.signal.aborted) return;
        if (e.status === 404) { notFound(e); return; }
        setHTML(page, errorCard(e));
        return;
      }
    }
    draw(data);
  }
  await load();
}
