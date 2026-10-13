// Backtest: ειλικρινής ετυμηγορία, καμπύλες κεφαλαίου, drawdown, μετρικές, βαθμονόμηση, συναλλαγές.
import { html, setHTML } from '../dom.js';
import { icon } from '../icons.js';
import * as F from '../format.js';
import { api } from '../api.js';
import { state, bus, startJob } from '../store.js';
import { equityChart, drawdownChart, calibrationChart, EQUITY_SERIES } from '../charts.js';
import { disclaimer, wireCommon, errorCard, emptyState, skeletonPage, noDataCard } from './common.js';

const METRICS = [
  ['total_return', 'Συνολική απόδοση', 'pct', 'high'],
  ['cagr', 'Ετήσια απόδοση (CAGR)', 'pct', 'high'],
  ['max_drawdown', 'Μέγιστη πτώση', 'pct', 'high'],
  ['sharpe', 'Δείκτης Sharpe', 'num', 'high'],
  ['volatility', 'Μεταβλητότητα (ετήσια)', 'pct', 'low'],
  ['n_trades', 'Συναλλαγές', 'int', null],
  ['win_rate', 'Ποσοστό κερδοφόρων', 'pct', 'high'],
  ['avg_win', 'Μέσο κέρδος', 'pct', null],
  ['avg_loss', 'Μέση ζημία', 'pct', null],
  ['profit_factor', 'Συντελεστής κέρδους', 'num', 'high'],
  ['avg_bars_held', 'Μέση διάρκεια (συνεδριάσεις)', 'num1', null],
  ['total_costs_eur', 'Συνολικά κόστη', 'eur', null],
  ['exposure', 'Χρόνος σε θέση', 'pct', null],
];
const PAGE = 20;
const MODEL_LABELS = { logreg: 'Λογιστική παλινδρόμηση', hgb: 'Gradient boosting' };
const COLS = [
  ['strategy', 'Στρατηγική', 'Στρατ.'],
  ['buy_hold', 'Δείκτης (αγορά & διακράτηση)', 'Δείκτης'],
  ['buy_hold_universe', 'Ισοσταθμική λίστα', 'Ισοσταθμ.'],
  ['random_median', 'Τυχαίες (διάμεσος)', 'Τυχαίες'],
];

function fmtMetric(v, kind) {
  if (!F.isNum(v)) return '—';
  switch (kind) {
    case 'pct': return F.pct(v, { digits: 1 });
    case 'num': return F.num(v, 2);
    case 'num1': return F.num(v, 1);
    case 'int': return F.int(v);
    case 'eur': return F.eur(v, { digits: 0 });
    default: return String(v);
  }
}

/** Τα κριτήρια της ετυμηγορίας ένα‑ένα (✓/✕), ώστε να φαίνεται τι λείπει για «πλεονέκτημα». */
function criteriaList(v, sk) {
  if (!v) return '';
  const s = v.strategy_return;
  const pv = v.p_value;
  const yes = (c) => (c == null ? null : !!c);
  const fmt = (x) => F.pct(x, { digits: 1, sign: true });
  const items = [
    [yes(F.isNum(pv) ? pv <= 0.05 : null), F.isNum(pv) ? html`Ξεπερνά τις τυχαίες στρατηγικές <span class="subtle">(p = ${F.num(pv, 3)}, απαιτείται ≤ 0,05)</span>` : 'Ξεπερνά τις τυχαίες στρατηγικές'],
    [yes(F.isNum(s) && F.isNum(v.buy_hold_return) ? s > v.buy_hold_return : null), html`Νικά τον δείκτη <span class="subtle">(${fmt(s)} έναντι ${fmt(v.buy_hold_return)})</span>`],
    [yes(F.isNum(s) && F.isNum(v.buy_hold_universe_return) ? s > v.buy_hold_universe_return : null), html`Νικά το ισοσταθμικό χαρτοφυλάκιο της λίστας <span class="subtle">(${fmt(v.buy_hold_universe_return)})</span>`],
    [yes(sk && sk.n ? sk.ok : null), sk && F.isNum(sk.logloss_model) ? html`Πιθανότητες καλύτερες από τις ιστορικές συχνότητες <span class="subtle">(log-loss ${F.num(sk.logloss_model, 3)} έναντι ${F.num(sk.logloss_base, 3)})</span>` : 'Πιθανότητες καλύτερες από τις ιστορικές συχνότητες'],
    [yes(v.profitable), 'Κερδοφόρα μετά από όλα τα κόστη'],
    [yes(F.isNum(v.n_trades) && F.isNum(v.n_random) ? v.n_trades >= 30 && v.n_random >= (v.min_random_runs || 100) : null), html`Αρκετά δείγματα <span class="subtle">(${F.int(v.n_trades)} συναλλαγές ≥ 30 · ${F.int(v.n_random)} τυχαίες ≥ ${F.int(v.min_random_runs || 100)})</span>`],
  ];
  return html`<ul class="v-criteria" aria-label="Κριτήρια ετυμηγορίας">${items.map(([ok, text]) => html`<li class="${ok == null ? 'is-na' : ok ? 'is-ok' : 'is-fail'}"><span class="ck ${ok == null ? 'ck-soft' : ok ? 'ck-ok' : 'ck-fail'}" aria-hidden="true">${icon(ok == null ? 'minus' : ok ? 'check' : 'x')}</span><span class="sr-only">${ok == null ? 'Δεν κρίθηκε: ' : ok ? 'Ναι: ' : 'Όχι: '}</span><span>${text}</span></li>`)}</ul>`;
}

function verdictCard(v, sk) {
  const level = v && ['good', 'weak', 'none'].includes(v.level) ? v.level : (v && v.edge ? 'good' : 'none');
  const title = level === 'good' ? 'Υπάρχει ένδειξη πλεονεκτήματος' : level === 'weak' ? 'Ασθενές, μη αποδεδειγμένο πλεονέκτημα' : 'Δεν υπάρχει πλεονέκτημα';
  const ic = level === 'good' ? 'check' : level === 'weak' ? 'alert' : 'ban';
  const pctl = v && F.isNum(v.percentile) ? Math.max(0, Math.min(1, v.percentile)) : null;
  const color = level === 'good' ? 'var(--pos-line)' : level === 'weak' ? 'var(--warn)' : 'var(--neg-line)';
  return html`<section class="card verdict verdict--${level}" aria-labelledby="verdict-title">
    <div class="v-badge" aria-hidden="true">${icon(ic)}</div>
    <div style="min-width:0;flex:1">
      <div class="eyebrow">Ετυμηγορία</div>
      <h2 class="v-title" id="verdict-title">${title}</h2>
      <p class="v-text">${(v && v.text_el) || 'Η στρατηγική συγκρίνεται με την αγορά & διακράτηση και με τυχαίες εισόδους υπό τους ίδιους κανόνες.'}</p>
      ${criteriaList(v, sk)}
    </div>
    ${pctl != null ? html`<div class="v-gauge">
      <div class="lbl">Η στρατηγική ξεπέρασε το</div>
      <div class="big">${F.prob(pctl)}</div>
      <div class="lbl">των τυχαίων δοκιμών · απαιτείται ≥ 95%</div>
      <div class="pctl" role="img" aria-label="Η στρατηγική ξεπέρασε το ${F.prob(pctl)} των τυχαίων δοκιμών, όριο 95%"><i style="width:${(pctl * 100).toFixed(1)}%;background:${color}"></i><span class="mark" title="Όριο 95%"></span></div>
      <div class="lbl" style="margin-top:6px;display:flex;justify-content:space-between"><span>0%</span><span>όριο 95%</span></div>
    </div>` : ''}
  </section>`;
}

function metricsTable(metrics) {
  const m = metrics || {};
  const cols = COLS.filter(([k]) => k !== 'buy_hold_universe' || m[k]);
  return html`<div class="table-wrap"><table class="table metrics-table table-cards">
    <thead><tr><th scope="col">Μετρική</th>${cols.map(([k, l]) => html`<th scope="col" class="r ${k === 'strategy' ? 'col-strat' : ''}">${l}</th>`)}</tr></thead>
    <tbody>${METRICS.map(([key, label, kind, better]) => {
      const vals = cols.map(([k]) => (m[k] ? m[k][key] : null));
      let best = -1;
      if (better) {
        const nums = vals.map((v) => (F.isNum(v) ? v : null));
        const cand = nums.map((v, i) => [v, i]).filter(([v]) => v != null);
        if (cand.length > 1) best = cand.sort((a, b) => (better === 'high' ? b[0] - a[0] : a[0] - b[0]))[0][1];
      }
      return html`<tr><th scope="row" class="m-name">${label}</th>${vals.map((v, i) => {
        const tone = (key === 'total_return' || key === 'cagr') ? F.tone(v) : key === 'max_drawdown' && F.isNum(v) && v < 0 ? 'neg' : '';
        return html`<td class="r num ${i === 0 ? 'col-strat' : ''} ${i === best ? 'best' : ''}" data-label="${cols[i][2]}"><span class="${tone}">${fmtMetric(v, kind)}</span>${i === best ? html` <span class="sr-only">(καλύτερο)</span><span aria-hidden="true" style="color:var(--accent);font-size:10px">●</span>` : ''}</td>`;
      })}</tr>`;
    })}</tbody>
  </table></div>`;
}

function splitBlock(by) {
  const tp = F.isNum(by && by.tp) ? by.tp : 0;
  const sl = F.isNum(by && by.sl) ? by.sl : 0;
  const to = F.isNum(by && by.to) ? by.to : 0;
  const n = tp + sl + to;
  if (!n) return html`<p class="subtle">Δεν υπάρχουν κλεισμένες συναλλαγές.</p>`;
  const seg = (v, color, label) => (v ? html`<i style="width:${(v / n * 100).toFixed(2)}%;background:${color}" title="${label}: ${F.int(v)}">${v / n > 0.09 ? F.prob(v / n) : ''}</i>` : '');
  return html`<div class="split-bar" role="img" aria-label="Στόχος ${F.int(tp)}, Λήξη ${F.int(to)}, Stop ${F.int(sl)}">${seg(tp, 'var(--pill-pos)', 'Στόχος')}${seg(to, 'var(--pill-to)', 'Λήξη')}${seg(sl, 'var(--pill-neg)', 'Stop')}</div>
    <div class="split-legend">
      <div class="row"><span><span class="sw" style="background:var(--pos-line)"></span>Στόχος πρώτα</span><b>${F.int(tp)} · ${F.prob(tp / n)}</b></div>
      <div class="row"><span><span class="sw" style="background:var(--to-line)"></span>Λήξη χρόνου</span><b>${F.int(to)} · ${F.prob(to / n)}</b></div>
      <div class="row"><span><span class="sw" style="background:var(--neg-line)"></span>Stop πρώτα</span><b>${F.int(sl)} · ${F.prob(sl / n)}</b></div>
    </div>`;
}

function tradesTable(trades, page) {
  const list = trades.slice().reverse();
  const pages = Math.max(1, Math.ceil(list.length / PAGE));
  const p = Math.min(Math.max(0, page), pages - 1);
  const slice = list.slice(p * PAGE, p * PAGE + PAGE);
  return html`<div class="table-wrap"><table class="table table-cards">
    <thead><tr><th scope="col">Μετοχή</th><th scope="col">Είσοδος</th><th scope="col">Έξοδος</th><th scope="col">Έκβαση</th><th scope="col" class="r hide-lg">Ποσό</th><th scope="col" class="r hide-lg">P(στόχος)</th><th scope="col" class="r">Καθαρή απόδοση</th><th scope="col" class="r">Καθαρό αποτέλεσμα</th></tr></thead>
    <tbody>${slice.map((t) => html`<tr>
      <td class="card-head-cell"><a href="#/stock/${encodeURIComponent(t.ticker)}" class="mono" style="font-weight:600;color:var(--text)">${F.shortTicker(t.ticker)}</a><span class="show-sm">${F.stateChip(t.label)}</span></td>
      <td data-label="Είσοδος"><span class="num">${F.date(t.entry_date)}</span><br><span class="subtle num" style="font-size:12px">${F.price(t.entry_price)}</span></td>
      <td data-label="Έξοδος"><span class="num">${F.date(t.exit_date)}</span><br><span class="subtle num" style="font-size:12px">${F.price(t.exit_price)}</span></td>
      <td class="hide-sm">${F.stateChip(t.label)}</td>
      <td class="r hide-lg" data-label="Ποσό"><span class="num">${F.eur(t.size_eur, { digits: 0 })}</span></td>
      <td class="r hide-lg" data-label="P(στόχος)"><span class="num">${F.prob(t.p_tp)}</span></td>
      <td class="r" data-label="Καθαρή απόδοση"><span class="num ${F.tone(t.net_ret)}" style="font-weight:600">${F.pct(t.net_ret, { digits: 2, sign: true })}</span></td>
      <td class="r" data-label="Καθαρό αποτέλεσμα"><span class="num ${F.tone(t.net_pnl_eur)}">${F.eur(t.net_pnl_eur, { sign: true })}</span></td>
    </tr>`)}</tbody>
  </table></div>
  <div class="pager"><span>Συναλλαγές ${F.int(p * PAGE + 1)}–${F.int(Math.min(list.length, p * PAGE + PAGE))} από ${F.int(list.length)} (πιο πρόσφατες πρώτα)</span>
    <span class="btns"><button type="button" class="btn btn-sm" data-page="${p - 1}" ${p <= 0 ? 'disabled' : ''}>${icon('chevronLeft', 'icon-sm')} Προηγούμενες</button><button type="button" class="btn btn-sm" data-page="${p + 1}" ${p >= pages - 1 ? 'disabled' : ''}>Επόμενες ${icon('chevronRight', 'icon-sm')}</button></span></div>`;
}

export async function render(page, ctx) {
  let data = null;
  let tradePage = 0;
  const cleanups = [];
  const clearCharts = () => { while (cleanups.length) cleanups.pop()(); };
  ctx.onCleanup(clearCharts);

  const canRun = () => !!(state.status && state.status.mode && state.status.mode !== 'empty');
  const runBtn = (cls = 'btn btn-primary') => (canRun()
    ? html`<button type="button" class="${cls}" data-run>${icon('play')} Τρέξε backtest</button>`
    : html`<button type="button" class="${cls}" data-run disabled title="Χρειάζονται πρώτα δεδομένα και εκπαιδευμένο μοντέλο">${icon('play')} Τρέξε backtest</button>`);
  const head = (d) => html`<div class="page-head">
    <div><div class="eyebrow">Έλεγχος στο παρελθόν</div><h1>Backtest</h1>
      <p class="lede">${d && d.period ? html`Περίοδος ${F.date(d.period.start)} – ${F.date(d.period.end)}` : 'Προσομοίωση με τους ίδιους κανόνες, κόστη και ελέγχους'}${d && d.generated_at ? html` · υπολογίστηκε ${F.ago(d.generated_at)}` : ''}</p></div>
    <div class="spacer"></div>
    ${canRun() ? html`<div class="actions">${runBtn()}</div>` : ''}
  </div>`;

  const draw = () => {
    clearCharts();
    const d = data;
    if (!d || d.available === false) {
      setHTML(page, html`${head(null)}${state.status && state.status.mode === 'empty' ? noDataCard() : html`<div class="card">${emptyState({
        iconName: 'backtest', title: 'Δεν υπάρχει ακόμη backtest',
        text: 'Το backtest προσομοιώνει τη στρατηγική σε όλο το ιστορικό, με κόστη, και τη συγκρίνει με την αγορά & διακράτηση και με 200 στρατηγικές τυχαίων εισόδων.',
        actions: runBtn(),
      })}</div>`}${disclaimer()}`);
      return;
    }
    const M = d.metrics || {};
    const trades = Array.isArray(d.trades) ? d.trades : [];
    const params = d.params && typeof d.params === 'object' ? d.params : {};
    const capital = F.isNum(params.capital_eur) ? params.capital_eur : (Array.isArray(d.equity) && d.equity[0] && F.isNum(d.equity[0].strategy) ? d.equity[0].strategy : null);
    const last = Array.isArray(d.equity) && d.equity.length ? d.equity[d.equity.length - 1] : null;
    const PARAM_LABELS = { horizon_days: ['Ορίζοντας', (v) => `${F.int(v)} ημ.`], k_tp: ['Στόχος', (v) => `${F.num(v, 1)} × ATR`], k_sl: ['Stop', (v) => `${F.num(v, 1)} × ATR`], capital_eur: ['Κεφάλαιο', (v) => F.eur(v, { digits: 0 })], position_eur_max: ['Μέγ. θέση', (v) => F.eur(v, { digits: 0 })], max_positions: ['Μέγ. θέσεις', (v) => F.int(v)], random_runs: ['Τυχαίες δοκιμές', (v) => F.int(v)], model_kind: ['Μοντέλο', (v) => MODEL_LABELS[v] || String(v)] };
    const allNotes = (Array.isArray(d.notes_el) ? d.notes_el : []).filter((n) => typeof n === 'string' && n);
    const demoNote = d.mode === 'demo' ? (allNotes.find((n) => n.startsWith('ΕΠΙΔΕΙΞΗ')) || 'ΕΠΙΔΕΙΞΗ: συνθετικά δεδομένα — τα αποτελέσματα δεν λένε τίποτα για την πραγματική αγορά.') : null;
    const notes = allNotes.filter((n) => n !== demoNote);
    if (d.random && typeof d.random.method_el === 'string' && d.random.method_el) notes.push(d.random.method_el);
    const notices = html`${d.stale ? html`<div class="notice notice-warn" role="note">${icon('alert')}<div><strong>${d.stale_reason_el || 'Το backtest δεν είναι ενημερωμένο.'}</strong> Πατήστε «Τρέξε backtest» για αποτελέσματα με τις τρέχουσες ρυθμίσεις.</div></div>` : ''}${demoNote ? html`<div class="notice notice-warn" role="note">${icon('alert')}<div><strong>Λειτουργία επίδειξης.</strong> ${demoNote.replace(/^ΕΠΙΔΕΙΞΗ:\s*/, '').replace(/^./, (c) => c.toUpperCase())}</div></div>` : ''}`;
    const paramChips = Object.entries(params).filter(([k, v]) => PARAM_LABELS[k] && v != null).map(([k, v]) => html`<span class="chip chip-outline">${PARAM_LABELS[k][0]}: <b style="color:var(--text);font-weight:600">${PARAM_LABELS[k][1](v)}</b></span>`);

    setHTML(page, html`${head(d)}
      ${verdictCard(d.verdict, d.skill)}
      ${notices}
      ${paramChips.length ? html`<div class="params" aria-label="Παράμετροι">${paramChips}<span class="chip chip-outline" title="ATR = μέσο ημερήσιο εύρος τιμής (14 συνεδριάσεων)">ATR = μέσο ημερήσιο εύρος</span></div>` : ''}
      <section class="card" aria-labelledby="eq-title">
        <div class="card-head"><h2 id="eq-title">Καμπύλη κεφαλαίου</h2><span class="sub">μετά από όλα τα κόστη</span></div>
        <div class="card-body">
          <div class="legend" style="margin-bottom:12px">
            ${EQUITY_SERIES.map((s) => html`<span><span class="sw-line" style="background:${s.color};${s.dash ? 'opacity:.85' : ''}"></span>${s.label}${last && F.isNum(last[s.key]) ? html` <b>${F.eur(last[s.key], { digits: 0 })}</b>` : ''}</span>`)}
            <span><span class="sw" style="background:var(--chart-rand);opacity:.35"></span>Ζώνη τυχαίων 5–95%</span>
          </div>
          <div class="chart" data-chart="equity"></div>
        </div>
      </section>
      <section class="card" aria-labelledby="dd-title">
        <div class="card-head"><h2 id="dd-title">Πτώση από την κορυφή</h2><span class="sub">της στρατηγικής, από το υψηλότερο σημείο της</span></div>
        <div class="card-body"><div class="chart" data-chart="dd"></div></div>
      </section>
      <div class="bt-grid">
        <section class="card" aria-labelledby="m-title">
          <div class="card-head"><h2 id="m-title">Σύγκριση μετρικών</h2><span class="sub">● καλύτερη τιμή · τυχαίες: διάμεσος κάθε μετρικής</span></div>
          <div class="card-body" style="padding-left:0;padding-right:0">${metricsTable(M)}</div>
        </section>
        <div style="display:grid;gap:16px;align-content:start">
          <section class="card" aria-labelledby="cal-title">
            <div class="card-head"><h2 id="cal-title">Βαθμονόμηση</h2><span class="sub">είναι αξιόπιστες οι πιθανότητες;</span></div>
            <div class="card-body"><div class="chart" data-chart="cal"></div>
              <div class="legend" style="margin-top:10px"><span><span class="sw" style="background:var(--accent-strong);border-radius:50%;opacity:.7"></span>Ομάδες προβλέψεων (μέγεθος = πλήθος)</span><span><span class="sw-line" style="background:var(--text-3)"></span>Τέλεια βαθμονόμηση</span></div></div>
          </section>
          <section class="card" aria-labelledby="split-title">
            <div class="card-head"><h2 id="split-title">Εκβάσεις συναλλαγών</h2></div>
            <div class="card-body">${splitBlock(d.by_label)}</div>
          </section>
        </div>
      </div>
      <section class="card" aria-labelledby="tr-title">
        <div class="card-head"><h2 id="tr-title">Συναλλαγές</h2><span class="sub">${trades.length ? `τελευταίες ${F.int(trades.length)}` : ''}</span></div>
        <div class="card-body" style="padding:8px 0 0" data-trades>${trades.length ? tradesTable(trades, tradePage) : html`<p class="subtle" style="padding:0 20px 20px">Καμία συναλλαγή.</p>`}</div>
      </section>
      ${notes.length ? html`<section class="card" aria-labelledby="bt-notes-title">
        <div class="card-head"><h2 id="bt-notes-title">Πώς έγινε ο έλεγχος</h2><span class="sub">κανόνες της προσομοίωσης</span></div>
        <div class="card-body"><ul class="bt-notes">${notes.map((n) => html`<li>${n}</li>`)}</ul></div>
      </section>` : ''}
      ${disclaimer()}`);

    const eq = page.querySelector('[data-chart="equity"]');
    if (eq) cleanups.push(equityChart(eq, d.equity, { capital }));
    const dd = page.querySelector('[data-chart="dd"]');
    if (dd) cleanups.push(drawdownChart(dd, d.drawdown));
    const cal = page.querySelector('[data-chart="cal"]');
    if (cal) cleanups.push(calibrationChart(cal, d.calibration));
  };

  wireCommon(page, { onRetry: () => load() });
  page.addEventListener('click', (e) => {
    if (e.target.closest('[data-run]')) { startJob('backtest'); return; }
    const pg = e.target.closest('[data-page]');
    if (pg && !pg.disabled && data) {
      tradePage = +pg.dataset.page;
      const host = page.querySelector('[data-trades]');
      setHTML(host, tradesTable(data.trades || [], tradePage));
      host.closest('section').scrollIntoView({ block: 'start', behavior: 'smooth' });
    }
  });

  async function load() {
    clearCharts();
    setHTML(page, skeletonPage({ kpis: 0, rows: 10 }));
    try {
      const d = await api.backtest({ signal: ctx.signal });
      if (ctx.signal.aborted) return;
      data = d && typeof d === 'object' ? d : { available: false };
      draw();
    } catch (e) {
      if (ctx.signal.aborted) return;
      if (e.status === 404) { data = { available: false }; draw(); return; }
      setHTML(page, html`${head(null)}${errorCard(e)}`);
    }
  }
  const off = bus.on('job', (j) => {
    const btns = page.querySelectorAll('[data-run]');
    btns.forEach((b) => { b.disabled = !canRun() || !!(j && j.type === 'backtest'); });
  });
  ctx.onCleanup(off);
  await load();
}
