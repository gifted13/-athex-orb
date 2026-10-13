// Ημερολόγιο εικονικών συναλλαγών: αγορές, απορρίψεις και μπλοκ, με την έκβασή τους.
import { html, setHTML, storage } from '../dom.js';
import { icon } from '../icons.js';
import * as F from '../format.js';
import { api } from '../api.js';
import { state, isFav, bus } from '../store.js';
import { disclaimer, wireCommon, errorCard, emptyState, skeletonPage, kpi, noDataCard } from './common.js';

const STATUS = {
  pending: { label: 'Σε αναμονή', chip: 'chip', glyph: '…' },
  open: { label: 'Ανοιχτή', chip: 'chip-info', glyph: '◔' },
  tp: { label: 'Στόχος', chip: 'chip-pos', glyph: '✓' },
  sl: { label: 'Stop', chip: 'chip-neg', glyph: '✕' },
  timeout: { label: 'Λήξη', chip: 'chip-to', glyph: '◷' },
  void: { label: 'Άκυρη', chip: 'chip-outline', glyph: '–' },
};
const RESOLVED = new Set(['tp', 'sl', 'timeout']);
const PAGE = 25;

function statusChip(s) {
  const st = STATUS[s] || { label: s || '—', chip: 'chip-outline', glyph: '' };
  return html`<span class="chip chip-sm status-chip ${st.chip}"><span aria-hidden="true">${st.glyph}</span> ${st.label}</span>`;
}

/** Σύνοψη από τις εγγραφές (χρησιμοποιείται όταν λείπουν πεδία από το summary του API). */
function computeSummary(items) {
  const group = (dec) => {
    const sub = items.filter((i) => i.decision === dec);
    const res = sub.filter((i) => RESOLVED.has(i.status));
    const withRet = res.filter((i) => F.isNum(i.net_ret));
    const wins = withRet.filter((i) => i.net_ret > 0).length;
    return {
      n: sub.length,
      resolved: res.length,
      tp: res.filter((i) => i.status === 'tp').length,
      sl: res.filter((i) => i.status === 'sl').length,
      timeout: res.filter((i) => i.status === 'timeout').length,
      hit_rate: withRet.length ? wins / withRet.length : null,
      avg_net_ret: withRet.length ? withRet.reduce((a, b) => a + b.net_ret, 0) / withRet.length : null,
    };
  };
  return {
    total: items.length,
    take: group('take'),
    skip: group('skip'),
    pending: items.filter((i) => i.status === 'pending').length,
    open: items.filter((i) => i.status === 'open').length,
  };
}

function pick(server, local) {
  if (!server || typeof server !== 'object') return local;
  const out = { ...local };
  for (const k of Object.keys(local)) if (server[k] != null) out[k] = server[k];
  return out;
}

export async function render(page, ctx) {
  let data = null;
  let filter = storage.get('jrFilter', 'all');
  if (!['all', 'take', 'skip'].includes(filter)) filter = 'all';
  let onlyFav = !!storage.get('jrFav', false);
  let pageNo = 0;

  const draw = () => {
    const items = (Array.isArray(data && data.items) ? data.items : []).filter(Boolean)
      .slice().sort((a, b) => String(b.signal_date || '').localeCompare(String(a.signal_date || '')) || String(a.ticker).localeCompare(String(b.ticker)));
    const local = computeSummary(items);
    const ss = (data && data.summary) || {};
    const sum = { ...local, ...Object.fromEntries(Object.entries(ss).filter(([, v]) => v != null && typeof v !== 'object')) };
    sum.take = pick(ss.take, local.take);
    sum.skip = pick(ss.skip, local.skip);

    const head = html`<div class="page-head"><div><div class="eyebrow">Εικονικές συναλλαγές</div><h1>Ημερολόγιο</h1>
      <p class="lede">Κάθε απόφαση καταγράφεται — και οι απορρίψεις — ώστε να φαίνεται αν το φίλτρο προσθέτει πραγματική αξία.</p></div></div>`;
    if (!items.length) {
      setHTML(page, html`${head}${state.status && state.status.mode === 'empty' ? noDataCard() : html`<div class="card">${emptyState({ iconName: 'journal', title: 'Το ημερολόγιο είναι άδειο', text: 'Μετά από κάθε ενημέρωση, οι αποφάσεις της ημέρας (λήψεις και απορρίψεις) καταγράφονται εδώ και επιλύονται αυτόματα όταν υπάρξουν νέες τιμές.' })}</div>`}${disclaimer()}`);
      return;
    }
    const t = sum.take || {};
    const s = sum.skip || {};
    const better = F.isNum(t.hit_rate) && F.isNum(s.hit_rate) ? t.hit_rate - s.hit_rate : null;

    const kpis = html`<div class="kpis">
      ${kpi({ label: 'Επιτυχία αγορών', iconName: 'check', value: t.hit_rate == null ? '—' : F.prob(t.hit_rate), valueClass: '', sub: `${F.int(t.resolved || 0)} κρίθηκαν από ${F.int(t.n || 0)} · κερδοφόρες μετά τα κόστη` })}
      ${kpi({ label: 'Επιτυχία απορρίψεων', iconName: 'minus', value: s.hit_rate == null ? '—' : F.prob(s.hit_rate), sub: `αν είχαν αγοραστεί · ${F.int(s.resolved || 0)} κρίθηκαν` })}
      ${kpi({ label: 'Μέση καθαρή απόδοση αγορών', iconName: 'sigma', value: t.avg_net_ret == null ? '—' : F.pct(t.avg_net_ret, { digits: 2, sign: true }), valueClass: F.tone(t.avg_net_ret), sub: s.avg_net_ret == null ? '' : `απορρίψεις: ${F.pct(s.avg_net_ret, { digits: 2, sign: true })}` })}
      ${kpi({ label: 'Σε εξέλιξη', iconName: 'clock', value: html`${F.int(sum.open || 0)}<small> ανοιχτές</small> · ${F.int(sum.pending || 0)}<small> σε αναμονή</small>`, sub: `${F.int(sum.total || items.length)} εγγραφές συνολικά` })}
    </div>`;
    const insight = better == null ? '' : html`<div class="notice ${better > 0.02 ? 'notice-info' : 'notice-warn'}" role="note">${icon(better > 0.02 ? 'info' : 'alert')}<div>${better > 0.02
      ? html`Μέχρι στιγμής οι αγορές πετυχαίνουν <strong>${F.pct(better, { digits: 0, sign: true })}</strong> συχνότερα από τις απορρίψεις. Το δείγμα είναι μικρό — κρίνετε με προσοχή.`
      : html`<strong>Οι αγορές δεν ξεπερνούν τις απορρίψεις</strong> (διαφορά ${F.pct(better, { digits: 0, sign: true })}). Μέχρι να αλλάξει αυτό, το φίλτρο δεν αποδεικνύεται χρήσιμο.`}</div></div>`;

    const nResolved = (t.resolved || 0) + (s.resolved || 0);
    const horizon = (state.status && state.status.horizon_days) || 10;
    const waiting = !nResolved && (sum.pending || sum.open) ? html`<div class="notice notice-info" role="note">${icon('clock')}<div><strong>Καμία απόφαση δεν έχει κριθεί ακόμη.</strong> Κάθε εγγραφή επιλύεται αυτόματα (στόχος, stop ή λήξη) όταν μια επόμενη «Ενημέρωση» φέρει τις τιμές των επόμενων έως ${F.int(horizon)} συνεδριάσεων. Έτσι το ημερολόγιο δείχνει με τον καιρό αν οι αγορές πετυχαίνουν συχνότερα από τις απορρίψεις.</div></div>` : '';

    let list = items;
    if (filter !== 'all') list = list.filter((i) => i.decision === filter);
    if (onlyFav) list = list.filter((i) => isFav(i.ticker));
    const pages = Math.max(1, Math.ceil(list.length / PAGE));
    pageNo = Math.min(pageNo, pages - 1);
    const slice = list.slice(pageNo * PAGE, pageNo * PAGE + PAGE);

    const filters = html`<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <div class="seg" role="group" aria-label="Φίλτρο απόφασης">
        <button type="button" data-filter="all" aria-pressed="${filter === 'all'}">Όλες <span class="count">${F.int(items.length)}</span></button>
        <button type="button" data-filter="take" aria-pressed="${filter === 'take'}">Αγορές <span class="count">${F.int(t.n || 0)}</span></button>
        <button type="button" data-filter="skip" aria-pressed="${filter === 'skip'}">Απορρίψεις &amp; μπλοκ <span class="count">${F.int(s.n || 0)}</span></button>
      </div>
      <label class="toggle"><input type="checkbox" data-onlyfav ${onlyFav ? 'checked' : ''}> Μόνο αγαπημένα</label>
    </div>`;

    const table = slice.length ? html`<div class="card"><div class="table-wrap"><table class="table table-cards">
      <thead><tr><th scope="col">Σήμα</th><th scope="col">Μετοχή</th><th scope="col">Απόφαση</th><th scope="col">Κατάσταση</th><th scope="col" class="r hide-lg">P(στόχος)</th><th scope="col" class="r">EV καθαρό</th><th scope="col" class="r hide-md">Είσοδος</th><th scope="col" class="r hide-md">Έξοδος</th><th scope="col" class="r">Καθαρή απόδοση</th><th scope="col" class="hide-lg">Αιτία</th></tr></thead>
      <tbody>${slice.map((i) => {
        const name = i.name || (state.byTicker.get(i.ticker) || {}).name || '';
        return html`<tr>
          <td class="hide-sm"><span class="num">${F.date(i.signal_date)}</span></td>
          <td class="card-head-cell"><div class="cell-stock"><a class="names" href="#/stock/${encodeURIComponent(i.ticker)}"><span class="t mono">${F.shortTicker(i.ticker)}</span><span class="n">${name}</span></a></div><span class="reason-inline">${i.reason_el || ''}</span><span class="show-sm subtle num" style="font-size:12px">${F.date(i.signal_date)}</span></td>
          <td data-label="Απόφαση">${F.decisionChip(i.decision === 'take' ? 'take' : i.blocked ? 'blocked' : 'skip', { small: true })}</td>
          <td data-label="Κατάσταση">${statusChip(i.status)}</td>
          <td class="r hide-lg" data-label="P(στόχος)"><span class="num">${F.prob(i.p_tp)}</span></td>
          <td class="r" data-label="EV καθαρό"><span class="num ${F.tone(i.ev_net)}">${F.pct(i.ev_net, { digits: 2, sign: true })}</span></td>
          <td class="r hide-md" data-label="Είσοδος"><span class="num">${F.price(i.entry_price ?? i.entry_ref)}</span>${i.entry_date ? html`<br><span class="subtle num" style="font-size:11.5px">${F.date(i.entry_date, 'short')}</span>` : ''}</td>
          <td class="r hide-md" data-label="Έξοδος"><span class="num">${F.price(i.exit_price)}</span>${i.exit_date ? html`<br><span class="subtle num" style="font-size:11.5px">${F.date(i.exit_date, 'short')}</span>` : ''}</td>
          <td class="r" data-label="Καθαρή απόδοση"><span class="num ${F.tone(i.net_ret)}" style="font-weight:600">${F.pct(i.net_ret, { digits: 2, sign: true })}</span></td>
          <td class="span-2 muted reason-cell hide-lg" data-label="Αιτία" title="${i.reason_el || ''}"><span class="clamp-2">${i.reason_el || '—'}</span></td>
        </tr>`;
      })}</tbody></table></div>
      ${pages > 1 ? html`<div class="pager"><span>Σελίδα ${F.int(pageNo + 1)} από ${F.int(pages)} · ${F.int(list.length)} εγγραφές</span><span class="btns"><button type="button" class="btn btn-sm" data-page="${pageNo - 1}" ${pageNo <= 0 ? 'disabled' : ''}>${icon('chevronLeft', 'icon-sm')} Νεότερες</button><button type="button" class="btn btn-sm" data-page="${pageNo + 1}" ${pageNo >= pages - 1 ? 'disabled' : ''}>Παλαιότερες ${icon('chevronRight', 'icon-sm')}</button></span></div>` : ''}
    </div>` : html`<div class="card">${emptyState({ iconName: 'journal', title: 'Καμία εγγραφή με αυτά τα φίλτρα', text: 'Αλλάξτε το φίλτρο ή απενεργοποιήστε το «Μόνο αγαπημένα».' })}</div>`;

    setHTML(page, html`${head}${kpis}${insight}${waiting}${filters}${table}${disclaimer()}`);
  };

  wireCommon(page, { onRetry: () => load() });
  page.addEventListener('click', (e) => {
    const f = e.target.closest('[data-filter]');
    if (f) { filter = f.dataset.filter; pageNo = 0; storage.set('jrFilter', filter); draw(); return; }
    const pg = e.target.closest('[data-page]');
    if (pg && !pg.disabled) { pageNo = +pg.dataset.page; draw(); }
  });
  page.addEventListener('change', (e) => {
    if (e.target.matches('[data-onlyfav]')) { onlyFav = e.target.checked; pageNo = 0; storage.set('jrFav', onlyFav); draw(); }
  });
  const off = bus.on('favorites', () => { if (onlyFav && data) draw(); });
  ctx.onCleanup(off);

  async function load() {
    setHTML(page, skeletonPage({ kpis: 4, rows: 10 }));
    try {
      data = await api.journal({ signal: ctx.signal });
      if (ctx.signal.aborted) return;
      draw();
    } catch (e) {
      if (ctx.signal.aborted) return;
      setHTML(page, html`<div class="page-head"><div><h1>Ημερολόγιο</h1></div></div>${errorCard(e)}`);
    }
  }
  await load();
}
