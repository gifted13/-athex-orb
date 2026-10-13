// Η Σφαίρα: τρία τόξα (Στόχος / Λήξη / Stop) γύρω από μια φωτεινή σφαίρα.
import { html, raw, uid } from './dom.js';
import { prob, pct, price, toNum, STATES, isNum, ORDER } from './format.js';

const RGB = { tp: [16, 185, 129], sl: [244, 63, 94], to: [131, 146, 168] };
const LINE = { tp: 'var(--pos-line)', sl: 'var(--neg-line)', to: 'var(--to-line)' };
const TEXT = { tp: 'var(--pos)', sl: 'var(--neg)', to: 'var(--to)' };

/** Κανονικοποιεί το αντικείμενο orb του API (null-safe, άθροισμα = 1, κυρίαρχη κατάσταση). */
export function normOrb(o) {
  if (!o || typeof o !== 'object') return null;
  let a = toNum(o.p_tp);
  let b = toNum(o.p_sl);
  let c = toNum(o.p_to);
  if (a == null && b == null && c == null) return null;
  a = Math.max(0, a ?? 0);
  b = Math.max(0, b ?? 0);
  c = Math.max(0, c ?? Math.max(0, 1 - a - b));
  const s = a + b + c;
  if (s <= 0) return null;
  a /= s; b /= s; c /= s;
  let dom = o.dominant;
  if (!['tp', 'sl', 'to'].includes(dom)) dom = a >= b && a >= c ? 'tp' : b >= c ? 'sl' : 'to';
  const be = toNum(o.breakeven_p_tp);
  return { p_tp: a, p_sl: b, p_to: c, dominant: dom, ev_net: toNum(o.ev_net), breakeven: be != null && be >= 0 && be <= 1 ? be : null };
}

/** Απόχρωση της σφαίρας: μείξη των τριών χρωμάτων, με έμφαση στην κυρίαρχη κατάσταση. */
export function tint(o, alpha = 1) {
  if (!o) return `rgba(100, 116, 139, ${alpha})`;
  const w = { tp: o.p_tp ** 3, sl: o.p_sl ** 3, to: o.p_to ** 3 };
  const tot = w.tp + w.sl + w.to || 1;
  const mix = [0, 1, 2].map((k) => Math.round((RGB.tp[k] * w.tp + RGB.sl[k] * w.sl + RGB.to[k] * w.to) / tot));
  return `rgba(${mix[0]}, ${mix[1]}, ${mix[2]}, ${alpha})`;
}

export const domProb = (o) => (o ? o[`p_${o.dominant}`] : null);

export function orbAria(o) {
  if (!o) return 'Σφαίρα πιθανοτήτων: δεν υπάρχει πρόβλεψη';
  const be = o.breakeven != null ? ` Νεκρό σημείο για τον στόχο: ${pct(o.breakeven, { digits: 1 })}.` : '';
  return `Σφαίρα πιθανοτήτων: ${ORDER.map((k) => `${STATES[k].label} ${prob(o[`p_${k}`])}`).join(', ')}. Κυρίαρχο: ${STATES[o.dominant].label}.${be}`;
}

/**
 * Επιστρέφει HTML για τη σφαίρα.
 * size: πλάτος σε px · mini: χωρίς υποδιαιρέσεις/ετικέτα (κάρτες, λίστες) ·
 * caption (μόνο mini): η κυρίαρχη κατάσταση γραμμένη κάτω από τη σφαίρα (ΣΤΟΧΟΣ / ΛΗΞΗ / STOP) ·
 * Στη μεγάλη σφαίρα, αν το API δίνει νεκρό σημείο (breakeven_p_tp), μια εγκοπή στο πράσινο τόξο δείχνει
 * πού πρέπει να φτάσει η P(στόχος) για να καλύψει τα κόστη.
 */
export function orbHTML(orbIn, { size = 260, mini = false, caption = false } = {}) {
  const o = normOrb(orbIn);
  const id = uid('orb');
  const t = tint(o);
  const C = 100;
  const R = 92;
  const sw = mini ? 10 : 6.5;
  const sphereR = mini ? 76 : 77;
  const gap = mini ? 5.5 : 2.4;

  let arcs = '';
  if (o) {
    let start = 0;
    for (const k of ORDER) {
      const p = o[`p_${k}`] * 100;
      const len = p - gap;
      if (len > 0.05) {
        arcs += `<circle class="orb-arc orb-arc--${k}${o.dominant === k ? ' is-dom' : ''}" cx="${C}" cy="${C}" r="${R}" stroke-width="${sw}" pathLength="100" stroke-dasharray="${len.toFixed(2)} ${(100 - len).toFixed(2)}" stroke-dashoffset="${(-(start + gap / 2)).toFixed(2)}" transform="rotate(-90 ${C} ${C})"/>`;
      }
      start += p;
    }
  }

  let beMark = '';
  if (o && !mini && o.breakeven != null) {
    const a = o.breakeven * Math.PI * 2 - Math.PI / 2;
    const pt = (r) => `${(C + r * Math.cos(a)).toFixed(2)},${(C + r * Math.sin(a)).toFixed(2)}`;
    const [x0, y0] = pt(R - 9).split(',');
    const [x1, y1] = pt(R + 9).split(',');
    beMark = `<g class="orb-be"><title>Νεκρό σημείο: P(στόχος) ${pct(o.breakeven, { digits: 1 })}</title><line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" class="orb-be-halo"/><line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" class="orb-be-line"/></g>`;
  }

  let ticks = '';
  if (!mini) {
    for (let i = 0; i < 60; i++) {
      const a = (i / 60) * Math.PI * 2 - Math.PI / 2;
      const r0 = i % 5 === 0 ? 98.5 : 99.5;
      const r1 = 102;
      ticks += `<line class="orb-tick" x1="${(C + r0 * Math.cos(a)).toFixed(2)}" y1="${(C + r0 * Math.sin(a)).toFixed(2)}" x2="${(C + r1 * Math.cos(a)).toFixed(2)}" y2="${(C + r1 * Math.sin(a)).toFixed(2)}"${i % 5 === 0 ? ' style="stroke-width:1.1"' : ''}/>`;
    }
  }

  const dp = domProb(o);
  const value = o ? Math.round(dp * 100) : null;
  const label = o ? STATES[o.dominant].label : 'Χωρίς πρόβλεψη';
  const labelColor = o ? tint(o, 1).replace(/rgba\((\d+), (\d+), (\d+), [\d.]+\)/, (m, r, g, b) => `rgb(${Math.min(255, +r + 90)}, ${Math.min(255, +g + 90)}, ${Math.min(255, +b + 90)})`) : '#cbd5e1';

  const text = mini
    ? `<text class="orb-value" x="${C}" y="${C + 16}" text-anchor="middle" font-size="46">${value == null ? '—' : value}<tspan font-size="26">${value == null ? '' : '%'}</tspan></text>`
    : `${o ? `<text class="orb-caption" x="${C}" y="${C - 30}" text-anchor="middle" font-size="8.5" letter-spacing=".9">ΠΙΘΑΝΟΤΗΤΑ</text>` : ''}
       <text class="orb-value" x="${C}" y="${C + 14}" text-anchor="middle" font-size="44">${value == null ? '—' : value}<tspan font-size="22" dx="1">${value == null ? '' : '%'}</tspan></text>
       <text class="orb-label" x="${C}" y="${C + 36}" text-anchor="middle" font-size="9.5" style="fill:${labelColor}">${label}</text>`;

  const svg = `<svg viewBox="-6 -6 212 212" aria-hidden="true" focusable="false">
    <defs>
      <radialGradient id="${id}-core" cx="50%" cy="38%" r="68%">
        <stop offset="0" stop-color="#1d2a47"/><stop offset=".55" stop-color="#0d1427"/><stop offset="1" stop-color="#03060d"/>
      </radialGradient>
      <radialGradient id="${id}-glow" cx="50%" cy="100%" r="80%">
        <stop offset="0" stop-color="${t}" stop-opacity=".95"/><stop offset=".42" stop-color="${t}" stop-opacity=".38"/><stop offset="1" stop-color="${t}" stop-opacity="0"/>
      </radialGradient>
      <radialGradient id="${id}-rim" cx="50%" cy="50%" r="50%">
        <stop offset=".74" stop-color="${t}" stop-opacity="0"/><stop offset=".93" stop-color="${t}" stop-opacity=".32"/><stop offset="1" stop-color="${t}" stop-opacity=".85"/>
      </radialGradient>
      <radialGradient id="${id}-spec" cx="50%" cy="50%" r="50%">
        <stop offset="0" stop-color="#fff" stop-opacity=".42"/><stop offset="1" stop-color="#fff" stop-opacity="0"/>
      </radialGradient>
      <clipPath id="${id}-clip"><circle cx="${C}" cy="${C}" r="${sphereR}"/></clipPath>
    </defs>
    ${ticks}
    <circle class="orb-track" cx="${C}" cy="${C}" r="${R}" stroke-width="${sw}"/>
    <g class="orb-arcs">${arcs}</g>${beMark}
    <g clip-path="url(#${id}-clip)">
      <circle cx="${C}" cy="${C}" r="${sphereR}" fill="url(#${id}-core)"/>
      <circle cx="${C}" cy="${C}" r="${sphereR}" fill="url(#${id}-glow)"/>
      ${mini ? '' : `<g>
        <ellipse class="orb-meridian" cx="${C}" cy="${C}" rx="${sphereR}" ry="${sphereR}"/>
        <ellipse class="orb-meridian" cx="${C}" cy="${C}" rx="${sphereR}" ry="${sphereR}"/>
        <ellipse class="orb-meridian" cx="${C}" cy="${C}" rx="${sphereR}" ry="${sphereR}"/>
      </g>
      <ellipse class="orb-lat" cx="${C}" cy="${C}" rx="${sphereR}" ry="${sphereR * 0.2}"/>
      <ellipse class="orb-lat" cx="${C}" cy="${C - sphereR * 0.5}" rx="${sphereR * 0.866}" ry="${sphereR * 0.15}"/>
      <ellipse class="orb-lat" cx="${C}" cy="${C + sphereR * 0.5}" rx="${sphereR * 0.866}" ry="${sphereR * 0.15}"/>`}
      <circle cx="${C}" cy="${C}" r="${sphereR}" fill="url(#${id}-rim)"/>
      <ellipse cx="${C - 24}" cy="${C - 36}" rx="${sphereR * 0.42}" ry="${sphereR * 0.24}" fill="url(#${id}-spec)" transform="rotate(-28 ${C - 24} ${C - 36})"/>
    </g>
    <circle cx="${C}" cy="${C}" r="${sphereR}" fill="none" stroke="rgba(255,255,255,.09)" stroke-width=".8"/>
    ${text}
  </svg>`;

  const node = html`<div class="orb ${mini ? 'orb--mini' : ''}" style="--orb-size:${size}px; --orb-tint:${t}" role="img" aria-label="${orbAria(o)}">
    <div class="orb-glow" aria-hidden="true"></div>${raw(svg)}</div>`;
  if (!(mini && caption)) return node;
  return html`<div class="orb-mini-wrap">${node}<span class="orb-mini-cap" style="color:${o ? TEXT[o.dominant] : 'var(--text-3)'}" aria-hidden="true">${o ? STATES[o.dominant].short : '—'}</span></div>`;
}

/** Μικρή κουκκίδα-donut για λίστες (ίδια σειρά με τη Σφαίρα: στόχος, λήξη, stop). */
export function orbDot(orbIn) {
  const o = normOrb(orbIn);
  if (!o) return html`<span class="orb-dot is-null" title="Χωρίς πρόβλεψη" aria-hidden="true"></span>`;
  const a = (o.p_tp * 100).toFixed(1);
  const b = ((o.p_tp + o.p_to) * 100).toFixed(1);
  return html`<span class="orb-dot" style="--a:${a};--b:${b}" title="${orbAria(o)}" aria-hidden="true"></span>`;
}

/**
 * Υπόμνημα: τι σημαίνει κάθε πιθανότητα για τη συγκεκριμένη μετοχή, το νεκρό σημείο και το καθαρό EV.
 * opts: { plan: {entry_ref, tp_price, sl_price, horizon_days}, until: «26 Οκτ», horizon }
 */
export function orbLegend(orbIn, { plan = null, until = '', horizon = null } = {}) {
  const o = normOrb(orbIn);
  const e = plan && isNum(plan.entry_ref) ? plan.entry_ref : null;
  const rel = (v) => (e && isNum(v) ? pct(v / e - 1, { digits: 1, sign: true }) : '');
  const h = horizon || (plan && plan.horizon_days) || null;
  const sub = {
    tp: e && isNum(plan.tp_price) ? html`η τιμή φτάνει <b>${price(plan.tp_price)}</b> (${rel(plan.tp_price)}) πριν από το stop` : 'η τιμή φτάνει τον στόχο πριν από το stop',
    to: html`κανένα από τα δύο${until ? html` έως <b>~${until}</b>` : ''}${h ? ` (${h} συνεδριάσεις)` : ''}· κλείσιμο στη λήξη`,
    sl: e && isNum(plan.sl_price) ? html`πέφτει πρώτα στο stop <b>${price(plan.sl_price)}</b> (${rel(plan.sl_price)})` : 'η τιμή πέφτει πρώτα στο stop',
  };
  const rows = ORDER.map((k) => {
    const v = o ? o[`p_${k}`] : null;
    const be = k === 'tp' && o && o.breakeven != null ? o.breakeven : null;
    return html`<div class="ol-row ${o && o.dominant === k ? 'is-dom' : ''}">
      <span class="ol-label"><span class="sw" style="background:${LINE[k]}"></span>${STATES[k].label}</span>
      <span class="ol-val">${v == null ? '—' : prob(v)}</span>
      <span class="ol-bar"><i style="width:${v == null ? 0 : (v * 100).toFixed(1)}%;background:${LINE[k]}"></i>${be != null ? html`<span class="ol-be" style="left:${(be * 100).toFixed(1)}%" title="Νεκρό σημείο ${pct(be, { digits: 1 })}"></span>` : ''}</span>
      <span class="ol-sub">${sub[k]}</span>
    </div>`;
  });
  let beRow = '';
  if (o && o.breakeven != null) {
    const short = o.p_tp < o.breakeven;
    beRow = html`<div class="orb-be-row ${short ? 'is-short' : 'is-ok'}"><span class="be-key" aria-hidden="true"></span><span>Νεκρό σημείο <b>${pct(o.breakeven, { digits: 1 })}</b>: τόση P(στόχος) χρειάζεται για να καλυφθούν τα κόστη — η Σφαίρα δίνει <b>${pct(o.p_tp, { digits: 1 })}</b>${short ? ', δηλαδή λιγότερο' : ', δηλαδή αρκετή'}.</span></div>`;
  }
  const ev = o ? o.ev_net : null;
  return html`<div class="orb-legend">${rows}${beRow}
    <div class="orb-ev"><span>EV καθαρό (μετά τα κόστη)</span><b class="${isNum(ev) ? (ev > 0 ? 'pos' : ev < 0 ? 'neg' : '') : 'subtle'}">${isNum(ev) ? pct(ev, { digits: 2, sign: true }) : '—'}</b></div>
  </div>`;
}
