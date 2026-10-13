// Κοινά κομμάτια για τις σελίδες: αποποίηση, κενές καταστάσεις, onboarding, skeletons.
import { html } from '../dom.js';
import { icon } from '../icons.js';
import { state, startJob, isFav, toggleFavoriteUI } from '../store.js';

export const DEFAULT_DISCLAIMER = 'Εργαλείο έρευνας — όχι επενδυτική συμβουλή. Οι αποδόσεις του παρελθόντος δεν εγγυώνται μελλοντικές. Δεν εκτελεί εντολές.';

export function disclaimer() {
  const text = (state.status && state.status.disclaimer_el) || DEFAULT_DISCLAIMER;
  return html`<footer class="disclaimer" role="contentinfo">${icon('info')}<span>${text}</span></footer>`;
}

export function emptyState({ iconName = 'info', title, text, actions = '', tone = '' }) {
  const color = tone === 'neg' ? 'color:var(--neg);background:var(--neg-soft)' : tone === 'warn' ? 'color:var(--warn);background:var(--warn-soft)' : '';
  return html`<div class="empty"><div class="empty-icon" style="${color}">${icon(iconName)}</div><h2>${title}</h2>${text ? html`<p>${text}</p>` : ''}${actions ? html`<div class="actions">${actions}</div>` : ''}</div>`;
}

export function errorCard(err, { retry = true } = {}) {
  return html`<div class="card">${emptyState({
    iconName: 'alert', tone: 'neg', title: 'Δεν ήταν δυνατή η φόρτωση',
    text: err && err.message ? err.message : 'Άγνωστο σφάλμα.',
    actions: retry ? html`<button type="button" class="btn btn-primary" data-retry>${icon('refresh')} Δοκιμή ξανά</button>` : '',
  })}</div>`;
}

/** Κάρτα έναρξης όταν δεν υπάρχουν δεδομένα (mode == "empty"). */
export function onboardingCard() {
  return html`<section class="card onboard" aria-labelledby="onboard-title">
    <div aria-hidden="true">${onboardOrb()}</div>
    <div>
      <div class="eyebrow">Καλώς ήρθατε</div>
      <h2 id="onboard-title">Ας φέρουμε τα πρώτα δεδομένα</h2>
      <p>Το ATHEX Orb χρειάζεται ιστορικές τιμές για να εκπαιδεύσει το μοντέλο και να υπολογίσει τις πιθανότητες
        «Στόχος πρώτα / Stop πρώτα / Λήξη χρόνου» για κάθε μετοχή.</p>
      <div class="actions">
        <button type="button" class="btn btn-primary btn-lg" data-job="refresh">${icon('database')} Λήψη πραγματικών δεδομένων</button>
        <button type="button" class="btn btn-lg" data-job="demo">${icon('sparkle')} Δοκιμή με δεδομένα επίδειξης</button>
      </div>
      <div class="steps">
        <span>• Τα πραγματικά δεδομένα κατεβαίνουν από το Yahoo Finance (χρειάζεται σύνδεση στο διαδίκτυο).</span>
        <span>• Τα δεδομένα επίδειξης είναι συνθετικά — ιδανικά για να γνωρίσετε την εφαρμογή.</span>
      </div>
    </div>
  </section>`;
}

function onboardOrb() {
  // στατική μικρή σφαίρα (χωρίς πιθανότητες)
  return html`<div class="orb" style="--orb-size:150px;--orb-tint:rgba(56,189,248,1)">
    <div class="orb-glow"></div>
    <svg viewBox="-6 -6 212 212" aria-hidden="true">
      <defs>
        <radialGradient id="ob-core" cx="50%" cy="38%" r="68%"><stop offset="0" stop-color="#1d2a47"/><stop offset=".55" stop-color="#0d1427"/><stop offset="1" stop-color="#03060d"/></radialGradient>
        <radialGradient id="ob-glow" cx="50%" cy="100%" r="80%"><stop offset="0" stop-color="#38bdf8" stop-opacity=".9"/><stop offset=".45" stop-color="#38bdf8" stop-opacity=".3"/><stop offset="1" stop-color="#38bdf8" stop-opacity="0"/></radialGradient>
        <radialGradient id="ob-spec" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#fff" stop-opacity=".42"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
      </defs>
      <circle class="orb-track" cx="100" cy="100" r="92" stroke-width="7"/>
      <g class="orb-arcs"><circle cx="100" cy="100" r="92" fill="none" stroke="var(--accent-strong)" stroke-width="7" stroke-linecap="round" pathLength="100" stroke-dasharray="22 78" transform="rotate(-90 100 100)"/></g>
      <circle cx="100" cy="100" r="77" fill="url(#ob-core)"/>
      <circle cx="100" cy="100" r="77" fill="url(#ob-glow)"/>
      <ellipse cx="76" cy="64" rx="32" ry="18" fill="url(#ob-spec)" transform="rotate(-28 76 64)"/>
      <text class="orb-value" x="100" y="112" text-anchor="middle" font-size="36">?</text>
    </svg></div>`;
}

/** Συνδέει κουμπιά [data-job] και [data-retry] μέσα σε ένα στοιχείο. */
export function wireCommon(root, { onRetry } = {}) {
  root.addEventListener('click', (e) => {
    const jobBtn = e.target.closest('[data-job]');
    if (jobBtn) { startJob(jobBtn.dataset.job); return; }
    const retry = e.target.closest('[data-retry]');
    if (retry && onRetry) { onRetry(); return; }
    const star = e.target.closest('[data-fav]');
    if (star) {
      e.preventDefault();
      e.stopPropagation();
      toggleFavoriteUI(star.dataset.fav);
    }
  });
}

export function starButton(ticker, name, { cls = 'star' } = {}) {
  const fav = isFav(ticker);
  return html`<button type="button" class="${cls}" data-fav="${ticker}" aria-pressed="${fav}" aria-label="${fav ? `Αφαίρεση ${name} από τα αγαπημένα` : `Προσθήκη ${name} στα αγαπημένα`}" title="${fav ? 'Αφαίρεση από τα αγαπημένα' : 'Προσθήκη στα αγαπημένα'}">${icon(fav ? 'starFill' : 'star', 'icon-sm')}</button>`;
}

export function skeletonPage({ kpis = 4, rows = 6 } = {}) {
  return html`<div aria-busy="true" aria-label="Φόρτωση…">
    <div style="display:grid;gap:10px"><div class="sk" style="width:120px;height:12px"></div><div class="sk" style="width:260px;height:28px"></div></div>
    ${kpis ? html`<div class="kpis" style="margin-top:22px">${Array.from({ length: kpis }, () => html`<div class="card kpi"><div class="sk sk-line" style="width:60%"></div><div class="sk" style="height:26px;width:50%;margin-top:6px"></div><div class="sk sk-line" style="width:70%;height:10px;margin-top:4px"></div></div>`)}</div>` : ''}
    <div class="card" style="margin-top:20px;padding:18px 20px;display:grid;gap:14px">${Array.from({ length: rows }, (_, i) => html`<div class="sk sk-line" style="width:${[92, 78, 86, 70, 90, 64, 82, 74][i % 8]}%"></div>`)}</div>
  </div>`;
}

export function kpi({ label, value, sub = '', iconName = null, valueClass = '' }) {
  return html`<div class="card kpi"><div class="kpi-label">${iconName ? icon(iconName) : ''}${label}</div><div class="kpi-value ${valueClass}">${value}</div>${sub ? html`<div class="kpi-sub">${sub}</div>` : ''}</div>`;
}

/** Τυπικό «Ο διακομιστής δεν έχει ακόμα δεδομένα» για σελίδες εκτός της Αγοράς. */
export function noDataCard(text = 'Δεν υπάρχουν ακόμη δεδομένα. Κάντε μια ενημέρωση ή δοκιμάστε τα δεδομένα επίδειξης.') {
  return html`<div class="card">${emptyState({
    iconName: 'database', title: 'Χωρίς δεδομένα', text,
    actions: html`<button type="button" class="btn btn-primary" data-job="refresh">${icon('database')} Λήψη δεδομένων</button><button type="button" class="btn" data-job="demo">${icon('sparkle')} Δεδομένα επίδειξης</button>`,
  })}</div>`;
}
