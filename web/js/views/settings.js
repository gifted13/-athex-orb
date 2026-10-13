// Ρυθμίσεις: κεφάλαιο, όρια εξόδου, φύλακας, κόστη, μοντέλο — με επεξήγηση για κάθε πεδίο.
import { html, setHTML } from '../dom.js';
import { icon } from '../icons.js';
import * as F from '../format.js';
import { api } from '../api.js';
import { state, startJob, loadStatus, refreshAll } from '../store.js';
import { toast } from '../toast.js';
import { themePref, setThemePref, motionPref, setMotionPref } from '../theme.js';
import { disclaimer, wireCommon, errorCard, skeletonPage } from './common.js';

// type: eur | int | num | pct (αποθηκεύεται ως κλάσμα, εμφανίζεται ως %) | select
const SECTIONS = [
  {
    id: 'portfolio', title: 'Χαρτοφυλάκιο', icon: 'wallet',
    lede: 'Πόσα χρήματα «παίζει» η προσομοίωση και πώς τα μοιράζει.',
    fields: [
      { key: 'capital_eur', label: 'Κεφάλαιο', type: 'eur', min: 100, help: 'Το εικονικό κεφάλαιο για το backtest και τις εικονικές συναλλαγές.' },
      { key: 'position_eur_max', label: 'Μέγιστο ποσό ανά θέση', type: 'eur', min: 50, help: 'Ανώτατο ποσό σε κάθε μετοχή. Σε μετοχές με χαμηλό τζίρο η θέση μικραίνει αυτόματα.' },
      { key: 'max_positions', label: 'Μέγιστες ταυτόχρονες θέσεις', type: 'int', min: 1, max: 50, help: 'Πόσες θέσεις μπορούν να είναι ανοιχτές ταυτόχρονα.' },
      { key: 'max_per_sector', label: 'Μέγιστες θέσεις ανά κλάδο', type: 'int', min: 1, max: 50, help: 'Αποφεύγει τη συγκέντρωση σε έναν κλάδο (π.χ. μόνο τράπεζες).' },
    ],
  },
  {
    id: 'barriers', title: 'Στόχος, stop & χρόνος', icon: 'target', retrain: true,
    lede: 'Τα τρία σενάρια της Σφαίρας. ATR = μέσο ημερήσιο εύρος τιμής των τελευταίων 14 συνεδριάσεων· «στόχος 2 × ATR» σημαίνει ότι ο στόχος μπαίνει δύο «συνηθισμένες ημερήσιες κινήσεις» πάνω από την είσοδο. Αλλαγές εδώ απαιτούν νέα εκπαίδευση.',
    fields: [
      { key: 'k_tp', label: 'Στόχος (× ATR)', type: 'num', step: 0.1, min: 0.2, max: 10, help: 'Ο στόχος τοποθετείται k × ATR πάνω από την τιμή εισόδου.' },
      { key: 'k_sl', label: 'Stop (× ATR)', type: 'num', step: 0.1, min: 0.2, max: 10, help: 'Το stop τοποθετείται k × ATR κάτω από την τιμή εισόδου.' },
      { key: 'horizon_days', label: 'Ορίζοντας (συνεδριάσεις)', type: 'int', min: 1, max: 60, help: 'Αν δεν χτυπήσει ούτε στόχος ούτε stop, η θέση κλείνει μετά από τόσες συνεδριάσεις.' },
    ],
  },
  {
    id: 'guard', title: 'Φύλακας', icon: 'shield',
    lede: 'Έλεγχοι που πρέπει να περάσει μια μετοχή πριν γίνει υποψήφια.',
    fields: [
      { key: 'min_adv_eur', label: 'Ελάχιστος μέσος τζίρος 20 ημερών', type: 'eur', min: 0, help: 'Μετοχές με μικρότερο ημερήσιο τζίρο μπλοκάρονται (κίνδυνος ρευστότητας).' },
      { key: 'min_ev_net', label: 'Ελάχιστο καθαρό EV', type: 'pct', digits: 2, help: 'Η αναμενόμενη απόδοση μετά τα κόστη πρέπει να ξεπερνά αυτό το όριο.' },
      { key: 'max_atr_pct', label: 'Μέγιστη μεταβλητότητα (ATR / τιμή)', type: 'pct', digits: 1, help: 'Αποκλείει μετοχές με υπερβολικές ημερήσιες διακυμάνσεις.' },
    ],
  },
  {
    id: 'costs', title: 'Κόστη συναλλαγών', icon: 'coins',
    lede: 'Πληρώνονται σε κάθε συναλλαγή, σε backtest και σε EV.',
    fields: [
      { key: 'commission_pct', label: 'Προμήθεια ανά πλευρά', type: 'pct', digits: 3, help: 'Ποσοστό της αξίας της εντολής, σε αγορά και σε πώληση.' },
      { key: 'commission_min_eur', label: 'Ελάχιστη προμήθεια', type: 'eur', min: 0, digits: 2, help: 'Ελάχιστη χρέωση ανά εντολή, όσο μικρή κι αν είναι.' },
      { key: 'exchange_fees_pct', label: 'Χρεώσεις χρηματιστηρίου', type: 'pct', digits: 3, help: 'Χρεώσεις ΧΑ/αποθετηρίου ανά πλευρά (κατά προσέγγιση).' },
      { key: 'sell_tax_pct', label: 'Φόρος πώλησης', type: 'pct', digits: 2, help: 'Ελέγξτε την ισχύουσα νομοθεσία.' },
    ],
  },
  {
    id: 'model', title: 'Μοντέλο', icon: 'cpu', retrain: true,
    lede: 'Ο αλγόριθμος που υπολογίζει τις τρεις πιθανότητες.',
    fields: [
      { key: 'kind', label: 'Τύπος μοντέλου', type: 'select', options: [['logreg', 'Λογιστική παλινδρόμηση (εξηγήσιμο)'], ['hgb', 'Gradient boosting (ευέλικτο)']], help: 'Η λογιστική παλινδρόμηση δείχνει και τους παράγοντες κάθε πρόβλεψης.' },
    ],
  },
];

function toDisplay(f, v) {
  if (v == null) return '';
  if (f.type === 'select') return String(v);
  if (f.type === 'pct') return F.num(v * 100, f.digits ?? 2).replace(/−/g, '-').replace(/\./g, '');
  if (f.type === 'int') return String(Math.round(v));
  if (f.type === 'eur') return F.num(v, f.digits ?? 0).replace(/\./g, '');
  return F.num(v, 2).replace(/\./g, '');
}

/** Όριο πεδίου στη μορφή του πεδίου: «100 €», «0,2 × ATR», «50», «60 ημ.». */
function boundText(f, v) {
  const digits = Number.isInteger(v) ? 0 : (Number.isInteger(v * 10) ? 1 : 2);
  const n = F.num(v, f.type === 'int' ? 0 : digits);
  const unit = unitOf(f);
  return unit ? `${n} ${unit}` : n;
}

function parseInput(f, text) {
  if (f.type === 'select') return { ok: true, value: text };
  const t = String(text).trim().replace(/\s/g, '').replace(/−/g, '-').replace(/€|%/g, '');
  if (t === '') return { ok: false, error: 'Το πεδίο είναι υποχρεωτικό.' };
  // ελληνική μορφή: 1.234,5 → 1234.5 · δεχόμαστε και 1234.5
  const norm = t.includes(',') ? t.replace(/\./g, '').replace(',', '.') : t;
  if (!/^-?\d+(\.\d+)?$/.test(norm)) return { ok: false, error: 'Μη έγκυρος αριθμός.' };
  let v = parseFloat(norm);
  if (f.type === 'int') {
    if (!Number.isInteger(v)) return { ok: false, error: 'Χρειάζεται ακέραιος αριθμός.' };
  }
  if (f.min != null && v < f.min) return { ok: false, error: `Ελάχιστο: ${boundText(f, f.min)}` };
  if (f.max != null && v > f.max) return { ok: false, error: `Μέγιστο: ${boundText(f, f.max)}` };
  if (f.type === 'pct') {
    if (v < -100 || v > 100) return { ok: false, error: 'Το ποσοστό πρέπει να είναι μεταξύ −100% και 100%.' };
    v = +(v / 100).toPrecision(10);
  }
  return { ok: true, value: v };
}

const unitOf = (f) => (f.type === 'eur' ? '€' : f.type === 'pct' ? '%' : f.key.startsWith('k_') ? '× ATR' : f.key === 'horizon_days' ? 'ημ.' : '');

function fieldHTML(sec, f, value) {
  const id = `f-${sec.id}-${f.key}`;
  if (f.type === 'select') {
    return html`<div class="field" data-field="${sec.id}.${f.key}">
      <label for="${id}">${f.label}</label>
      <div class="input"><select id="${id}" data-sec="${sec.id}" data-key="${f.key}" aria-describedby="${id}-help">${f.options.map(([v, l]) => html`<option value="${v}" ${String(value) === v ? 'selected' : ''}>${l}</option>`)}</select>${icon('chevronDown')}</div>
      <div class="help" id="${id}-help">${f.help}</div>
    </div>`;
  }
  const unit = unitOf(f);
  return html`<div class="field" data-field="${sec.id}.${f.key}">
    <label for="${id}">${f.label}</label>
    <div class="input ${unit ? 'has-unit' : ''}"><input id="${id}" type="text" inputmode="${f.type === 'int' ? 'numeric' : 'decimal'}" autocomplete="off" data-sec="${sec.id}" data-key="${f.key}" value="${toDisplay(f, value)}" aria-describedby="${id}-help ${id}-err" style="${unit.length > 2 ? 'padding-right:64px' : ''}">${unit ? html`<span class="unit">${unit}</span>` : ''}</div>
    <div class="err" id="${id}-err" aria-live="polite"></div>
    <div class="help" id="${id}-help">${f.help}</div>
  </div>`;
}

export async function render(page, ctx) {
  let loaded = null;
  // ο διακομιστής ξέρει αν οι αποθηκευμένες ρυθμίσεις διαφέρουν από αυτές του εκπαιδευμένου μοντέλου
  let retrainNote = !!(state.status && state.status.retrain_needed);

  const appearance = () => html`<section class="card form-section" aria-labelledby="ap-title">
    <h2 id="ap-title">${icon('sun')} Εμφάνιση</h2>
    <p>Αποθηκεύεται μόνο σε αυτό το πρόγραμμα περιήγησης.</p>
    <div class="field"><span style="font-size:13px;font-weight:550">Θέμα</span>
      <div class="seg" role="group" aria-label="Θέμα">${[['dark', 'Σκούρο'], ['light', 'Φωτεινό'], ['system', 'Σύστημα']].map(([k, l]) => html`<button type="button" data-theme-pref="${k}" aria-pressed="${themePref() === k}">${l}</button>`)}</div></div>
    <div class="field"><span style="font-size:13px;font-weight:550">Κίνηση</span>
      <div class="seg" role="group" aria-label="Κίνηση">${[['auto', 'Αυτόματα'], ['reduce', 'Μειωμένη']].map(([k, l]) => html`<button type="button" data-motion-pref="${k}" aria-pressed="${motionPref() === k}">${l}</button>`)}</div>
      <div class="help">Η «Μειωμένη» σταματά τη «ανάσα» της σφαίρας και τις μεταβάσεις.</div></div>
  </section>`;

  const howItWorks = () => html`<section class="card form-section" aria-labelledby="how-title">
    <h2 id="how-title">${icon('info')} Πώς λειτουργεί</h2>
    <ol class="how">
      <li><span class="how-icon">${icon('scout')}</span><span><b>Ανιχνευτής</b> — κατεβάζει και ελέγχει τις ημερήσιες τιμές κάθε μετοχής.</span></li>
      <li><span class="how-icon">${icon('compass')}</span><span><b>Χαρτογράφος</b> — διαβάζει το κλίμα της αγοράς και του κλάδου.</span></li>
      <li><span class="how-icon">${icon('scale')}</span><span><b>Κριτής</b> — το μοντέλο δίνει τρεις πιθανότητες: στόχος πρώτα, stop πρώτα ή λήξη χρόνου (η Σφαίρα).</span></li>
      <li><span class="how-icon">${icon('sigma')}</span><span><b>Ποσοτικός</b> — μετατρέπει τις πιθανότητες σε αναμενόμενη αξία μετά από όλα τα κόστη.</span></li>
      <li><span class="how-icon">${icon('shield')}</span><span><b>Φύλακας</b> — μπλοκάρει ό,τι είναι αρνητικό, ακριβό ή επικίνδυνο.</span></li>
      <li><span class="how-icon">${icon('target')}</span><span><b>Εκτελεστής</b> — καταγράφει εικονικές θέσεις. <b>Ποτέ</b> δεν στέλνει πραγματικές εντολές.</span></li>
    </ol>
    <p class="subtle" style="font-size:12.5px;margin-top:0">Χωρίς «κρυφή ματιά» στο μέλλον: κάθε πρόβλεψη χρησιμοποιεί μόνο δεδομένα που υπήρχαν εκείνη τη μέρα. Το backtest συγκρίνει πάντα με τυχαίες εισόδους.</p>
  </section>`;

  const demoCard = () => html`<section class="card form-section" aria-labelledby="demo-title">
    <h2 id="demo-title">${icon('sparkle')} Δεδομένα επίδειξης</h2>
    <p>Συνθετικές τιμές για να δοκιμάσετε την εφαρμογή χωρίς σύνδεση. Επισημαίνονται παντού ως «ΕΠΙΔΕΙΞΗ».</p>
    <div><button type="button" class="btn" data-job="demo">${icon('sparkle')} Δημιουργία δεδομένων επίδειξης</button></div>
  </section>`;

  const advanced = () => html`<section class="card form-section" aria-labelledby="adv-title">
    <h2 id="adv-title">${icon('database')} Για προχωρημένους</h2>
    <p>Οι αλλαγές σας αποθηκεύονται στο αρχείο <span class="code">data/settings.json</span> και υπερισχύουν των προεπιλογών του <span class="code">config.toml</span> (εκεί ορίζονται και η πηγή δεδομένων, τα χρόνια ιστορικού και το πλήθος των τυχαίων δοκιμών του backtest).</p>
  </section>`;

  const shortcuts = () => html`<section class="card form-section hide-sm" aria-labelledby="kb-title">
    <h2 id="kb-title">${icon('keyboard')} Συντομεύσεις</h2>
    <div class="shortcuts">
      <div><span>Αναζήτηση μετοχής</span><kbd>/</kbd></div>
      <div><span>Μετακίνηση στα αποτελέσματα</span><span><kbd>↑</kbd> <kbd>↓</kbd></span></div>
      <div><span>Άνοιγμα μετοχής</span><kbd>Enter</kbd></div>
      <div><span>Προσθήκη/αφαίρεση από αγαπημένα</span><span><kbd>⇧</kbd> <kbd>Enter</kbd></span></div>
      <div><span>Καθαρισμός αναζήτησης</span><kbd>Esc</kbd></div>
      <div><span>Μετακίνηση στα γραφήματα</span><span><kbd>←</kbd> <kbd>→</kbd></span></div>
    </div>
  </section>`;

  const draw = () => {
    const s = loaded || {};
    const forms = SECTIONS.filter((sec) => s[sec.id] && typeof s[sec.id] === 'object')
      .map((sec) => {
        const fields = sec.fields.filter((f) => s[sec.id][f.key] !== undefined);
        if (!fields.length) return '';
        return html`<section class="card form-section" aria-labelledby="sec-${sec.id}">
          <h2 id="sec-${sec.id}">${icon(sec.icon)} ${sec.title}${sec.retrain ? html` <span class="chip chip-sm chip-outline" title="Αλλαγές εδώ απαιτούν νέα ενημέρωση">επανεκπαίδευση</span>` : ''}</h2>
          <p>${sec.lede}</p>
          ${fields.map((f) => fieldHTML(sec, f, s[sec.id][f.key]))}
        </section>`;
      });
    setHTML(page, html`<div class="page-head"><div><div class="eyebrow">Προτιμήσεις</div><h1>Ρυθμίσεις</h1>
        <p class="lede">Κεφάλαιο, στόχος και stop, έλεγχοι κινδύνου και κόστη. Κάθε πεδίο εξηγεί τι κάνει.</p></div></div>
      <form class="settings-grid" novalidate data-form>
        <div style="display:grid;gap:16px;align-content:start">${forms.slice(0, 3)}</div>
        <div style="display:grid;gap:16px;align-content:start">${forms.slice(3)}${appearance()}${demoCard()}${howItWorks()}${shortcuts()}${advanced()}</div>
        <div class="save-bar ${retrainNote ? '' : 'is-idle'}" style="grid-column:1/-1" data-savebar>
          <span class="msg" data-msg aria-live="polite">${retrainNote ? html`<span class="chip chip-warn chip-sm">${icon('alert', 'icon-sm')} Απαιτείται νέα ενημέρωση</span> για να εφαρμοστούν οι αλλαγές στο μοντέλο.` : 'Καμία αλλαγή.'}</span>
          ${retrainNote ? html`<button type="button" class="btn" data-job="refresh">${icon('refresh')} Ενημέρωση τώρα</button>` : ''}
          <button type="button" class="btn btn-ghost" data-reset disabled>Αναίρεση αλλαγών</button>
          <button type="submit" class="btn btn-primary" data-save disabled>${icon('check')} Αποθήκευση</button>
        </div>
      </form>
      ${disclaimer()}`);
    updateDirty();
  };

  function fieldDef(secId, key) {
    const sec = SECTIONS.find((s) => s.id === secId);
    return sec && sec.fields.find((f) => f.key === key);
  }

  function collect() {
    const changes = {};
    let errors = 0;
    page.querySelectorAll('[data-sec][data-key]').forEach((inp) => {
      const f = fieldDef(inp.dataset.sec, inp.dataset.key);
      const box = inp.closest('.field');
      const err = box.querySelector('.err');
      const r = parseInput(f, inp.value);
      box.classList.toggle('is-invalid', !r.ok);
      inp.setAttribute('aria-invalid', String(!r.ok));
      if (err) err.textContent = r.ok ? '' : r.error;
      if (!r.ok) { errors++; box.classList.remove('is-dirty'); return; }
      const orig = loaded[inp.dataset.sec][inp.dataset.key];
      const changed = f.type === 'select' ? String(orig) !== String(r.value) : Math.abs(Number(orig) - r.value) > 1e-12;
      box.classList.toggle('is-dirty', changed);
      if (changed) (changes[inp.dataset.sec] ||= {})[inp.dataset.key] = r.value;
    });
    return { changes, errors };
  }

  function updateDirty() {
    if (!loaded) return;
    const { changes, errors } = collect();
    const n = Object.values(changes).reduce((a, o) => a + Object.keys(o).length, 0);
    const save = page.querySelector('[data-save]');
    const reset = page.querySelector('[data-reset]');
    const msg = page.querySelector('[data-msg]');
    if (save) save.disabled = n === 0 || errors > 0;
    if (reset) reset.disabled = n === 0;
    const bar = page.querySelector('[data-savebar]');
    if (bar) bar.classList.toggle('is-idle', !n && !errors && !retrainNote);
    if (msg && (n || errors)) {
      msg.textContent = errors ? `Διορθώστε ${errors === 1 ? '1 πεδίο' : `${errors} πεδία`} πριν την αποθήκευση.` : `${n === 1 ? '1 αλλαγή' : `${n} αλλαγές`} προς αποθήκευση.`;
    }
  }

  async function save() {
    const { changes, errors } = collect();
    if (errors || !Object.keys(changes).length) return;
    const btn = page.querySelector('[data-save]');
    if (btn) { btn.disabled = true; setHTML(btn, html`${icon('refresh')} Αποθήκευση…`); }
    try {
      const r = await api.saveSettings(changes);
      const needs = !!(r && (r.retrain_needed ?? r.needs_retrain ?? r.needs_refresh ?? r.retrain));
      for (const [sec, vals] of Object.entries(changes)) Object.assign(loaded[sec], vals);
      if (r && r.settings && typeof r.settings === 'object') loaded = r.settings;
      retrainNote = needs;          // σε σύγκριση με το εκπαιδευμένο μοντέλο (η επαναφορά μιας αλλαγής το καθαρίζει)
      // αν ο διακομιστής ξαναϋπολόγισε τα σήματα (φύλακας/κόστη/χαρτοφυλάκιο), ανανεώνονται λίστα και caches
      (r && r.signals_updated ? refreshAll() : loadStatus()).catch(() => {});
      toast(needs ? 'Οι ρυθμίσεις αποθηκεύτηκαν — απαιτείται νέα ενημέρωση για το μοντέλο.' : 'Οι ρυθμίσεις αποθηκεύτηκαν.', { type: 'success', action: needs ? { label: 'Ενημέρωση', onClick: () => startJob('refresh') } : null });
      draw();
    } catch (e) {
      toast(`Η αποθήκευση απέτυχε: ${e.message}`, { type: 'error' });
      if (btn) { setHTML(btn, html`${icon('check')} Αποθήκευση`); updateDirty(); }
    }
  }

  wireCommon(page, { onRetry: () => load() });
  page.addEventListener('input', (e) => { if (e.target.matches('[data-sec]')) updateDirty(); });
  page.addEventListener('change', (e) => { if (e.target.matches('[data-sec]')) updateDirty(); });
  page.addEventListener('submit', (e) => { e.preventDefault(); save(); });
  page.addEventListener('click', (e) => {
    if (e.target.closest('[data-reset]')) { draw(); return; }
    const tp = e.target.closest('[data-theme-pref]');
    if (tp) { setThemePref(tp.dataset.themePref); page.querySelectorAll('[data-theme-pref]').forEach((b) => b.setAttribute('aria-pressed', String(b === tp))); return; }
    const mp = e.target.closest('[data-motion-pref]');
    if (mp) { setMotionPref(mp.dataset.motionPref); page.querySelectorAll('[data-motion-pref]').forEach((b) => b.setAttribute('aria-pressed', String(b === mp))); }
  });

  async function load() {
    setHTML(page, skeletonPage({ kpis: 0, rows: 12 }));
    try {
      const s = await api.settings({ signal: ctx.signal });
      if (ctx.signal.aborted) return;
      loaded = s && typeof s === 'object' ? JSON.parse(JSON.stringify(s)) : {};
      draw();
    } catch (e) {
      if (ctx.signal.aborted) return;
      setHTML(page, html`<div class="page-head"><div><h1>Ρυθμίσεις</h1></div></div>${errorCard(e)}`);
    }
  }
  await load();
}
