// Γραφήματα σε καθαρό SVG (χωρίς βιβλιοθήκες): sparkline, τιμή (κεριά/γραμμή) με σχέδιο TP/SL,
// στοιβαγμένη περιοχή πιθανοτήτων, καμπύλες κεφαλαίου με ζώνη p05–p95, drawdown, βαθμονόμηση.
import { html, raw, uid, esc } from './dom.js';
import * as F from './format.js';

const isNum = F.isNum;
const r1 = (v) => Math.round(v * 10) / 10;

// ---------------------------------------------------------------------------
// Κλίμακες & βοηθητικά
// ---------------------------------------------------------------------------
export function linear(d0, d1, r0, r1_) {
  const k = d1 === d0 ? 0 : (r1_ - r0) / (d1 - d0);
  const f = (v) => r0 + (v - d0) * k;
  f.invert = (p) => (k ? d0 + (p - r0) / k : d0);
  return f;
}

/** «Ωραίες» υποδιαιρέσεις (1, 2, 5 × 10^k). */
export function niceTicks(min, max, count = 5) {
  if (!isNum(min) || !isNum(max)) return [];
  if (min === max) { const d = Math.abs(min) * 0.05 || 1; min -= d; max += d; }
  const span = max - min;
  const step0 = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(step0));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function extent(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!isNum(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo === Infinity ? [0, 1] : [lo, hi];
}

/** Path από σημεία· τα null σπάνε τη γραμμή. */
function linePath(pts) {
  let d = '';
  let pen = false;
  for (const p of pts) {
    if (!p || !isNum(p[1])) { pen = false; continue; }
    d += `${pen ? 'L' : 'M'}${r1(p[0])},${r1(p[1])}`;
    pen = true;
  }
  return d;
}

function areaPath(top, bottom) {
  const t = top.filter((p) => p && isNum(p[1]));
  const b = bottom.filter((p) => p && isNum(p[1]));
  if (!t.length || !b.length) return '';
  let d = `M${r1(t[0][0])},${r1(t[0][1])}`;
  for (let i = 1; i < t.length; i++) d += `L${r1(t[i][0])},${r1(t[i][1])}`;
  for (let i = b.length - 1; i >= 0; i--) d += `L${r1(b[i][0])},${r1(b[i][1])}`;
  return `${d}Z`;
}

function nearestIndex(xs, x) {
  let lo = 0;
  let hi = xs.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid; else hi = mid;
  }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
}

/** Ετικέτες ημερομηνιών: αλλαγές μήνα, τριμήνου ή έτους, ανάλογα με τον χώρο. */
function dateTicks(dates, X, width, minGap = 64) {
  const ds = dates.map(F.parseDate);
  const levels = [
    { test: (a, b) => a.getMonth() !== b.getMonth(), fmt: (d) => (d.getMonth() === 0 ? `${F.date(d, 'month')} ${String(d.getFullYear()).slice(2)}` : F.date(d, 'month')) },
    { test: (a, b) => a.getMonth() !== b.getMonth() && b.getMonth() % 3 === 0, fmt: (d) => (d.getMonth() === 0 ? String(d.getFullYear()) : F.date(d, 'month')) },
    { test: (a, b) => a.getFullYear() !== b.getFullYear(), fmt: (d) => String(d.getFullYear()) },
  ];
  for (const lv of levels) {
    const idx = [];
    for (let i = 1; i < ds.length; i++) if (ds[i] && ds[i - 1] && lv.test(ds[i - 1], ds[i])) idx.push(i);
    const maxLabels = Math.max(2, Math.floor(width / minGap));
    if (idx.length <= maxLabels || lv === levels[levels.length - 1]) {
      let k = 1;
      while (idx.length / k > maxLabels) k++;
      return idx.filter((_, j) => j % k === 0).map((i) => ({ x: X(i), label: lv.fmt(ds[i]) }));
    }
  }
  return [];
}

/** Σχεδιάζει ξανά το γράφημα όταν αλλάζει το πλάτος του container. */
export function mountChart(el, draw, { minWidth = 220 } = {}) {
  let w = 0;
  let raf = 0;
  const run = () => {
    if (!el.isConnected) return;
    const nw = Math.max(minWidth, Math.floor(el.clientWidth));
    if (nw === w) return;
    w = nw;
    draw(w);
  };
  const ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(run); });
  ro.observe(el);
  run();
  return () => { ro.disconnect(); cancelAnimationFrame(raf); };
}

/**
 * Διαδραστικότητα: σταυρόνημα/tooltip με ποντίκι, αφή και πληκτρολόγιο (←/→).
 * show(i) → { x, svg (στοιχεία οδηγού), tip (HTML) }
 */
function bindHover(el, { xs, plot, show }) {
  const svg = el.querySelector('svg');
  const tip = el.querySelector('.ch-tip');
  const g = svg && svg.querySelector('.ch-hover');
  if (!svg || !tip || !g || !xs.length) return;
  let cur = -1;
  const render = (i) => {
    if (i < 0 || i >= xs.length) return;
    cur = i;
    const r = show(i);
    g.innerHTML = r.svg;
    g.style.display = '';
    tip.innerHTML = String(r.tip);
    tip.hidden = false;
    const W = el.clientWidth;
    const tw = tip.offsetWidth;
    let left = r.x + 16;
    if (left + tw > W - 2) left = r.x - tw - 16;
    if (left < 2) left = Math.max(2, Math.min(W - tw - 2, r.x - tw / 2));
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top = `${Math.round(r.tipY ?? plot.top)}px`;
  };
  const hide = () => { cur = -1; g.style.display = 'none'; tip.hidden = true; };
  const fromEvent = (e) => {
    const rect = svg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < plot.left - 12 || x > plot.right + 12) { hide(); return; }
    render(nearestIndex(xs, x));
  };
  svg.addEventListener('pointermove', fromEvent);
  svg.addEventListener('pointerdown', fromEvent);
  svg.addEventListener('pointerleave', (e) => { if (e.pointerType !== 'touch') hide(); });
  el.addEventListener('keydown', (e) => {
    const last = xs.length - 1;
    let i = null;
    const stepBig = Math.max(1, Math.round(xs.length / 20));
    if (e.key === 'ArrowRight') i = cur < 0 ? last : Math.min(last, cur + (e.shiftKey ? stepBig : 1));
    else if (e.key === 'ArrowLeft') i = cur < 0 ? last : Math.max(0, cur - (e.shiftKey ? stepBig : 1));
    else if (e.key === 'Home') i = 0;
    else if (e.key === 'End') i = last;
    else if (e.key === 'Escape') { hide(); return; }
    if (i != null) { e.preventDefault(); render(i); }
  });
  el.addEventListener('blur', hide);
  hide();
}

function frame(el, { W, H, label, body, defs = '' }) {
  el.setAttribute('tabindex', '0');
  el.setAttribute('role', 'figure');
  el.setAttribute('aria-label', label);
  el.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>${defs ? `<defs>${defs}</defs>` : ''}${body}<g class="ch-hover" style="display:none"></g></svg><div class="ch-tip" hidden></div>`;
}

function emptyChart(el, text, H = 160) {
  el.removeAttribute('tabindex');
  el.innerHTML = `<div class="empty" style="min-height:${H}px;padding:24px"><p>${esc(text)}</p></div>`;
}

const tipRow = (label, value, sw = '', cls = '') => `<div class="tip-row"><span>${sw}${esc(label)}</span><b class="${cls}">${esc(value)}</b></div>`;
const swatch = (color, line = false) => `<span class="${line ? 'sw-line' : 'sw'}" style="background:${color}"></span>`;

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------
export function sparkline(values, { w = 80, h = 24, area = false, strokeWidth = 1.5, label = '' } = {}) {
  const vals = (values || []).filter(isNum);
  if (vals.length < 2) return html`<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><line x1="0" y1="${h / 2}" x2="${w}" y2="${h / 2}" stroke="var(--border-strong)" stroke-dasharray="2 3"/></svg>`;
  const [lo, hi] = extent(vals);
  const pad = strokeWidth;
  const X = linear(0, vals.length - 1, pad, w - pad);
  const Y = linear(lo, hi === lo ? lo + 1 : hi, h - pad, pad);
  const pts = vals.map((v, i) => [X(i), Y(v)]);
  const up = vals[vals.length - 1] >= vals[0];
  const color = up ? 'var(--pos-line)' : 'var(--neg-line)';
  let fill = '';
  if (area) {
    const id = uid('sg');
    fill = `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".22"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs><path d="${areaPath(pts, [[X(0), h], [X(vals.length - 1), h]])}" fill="url(#${id})"/>`;
  }
  const last = pts[pts.length - 1];
  const aria = label ? ` role="img" aria-label="${esc(label)}"` : ' aria-hidden="true"';
  return raw(`<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"${aria}>${fill}<path d="${linePath(pts)}" fill="none" stroke="${color}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>${area ? `<circle cx="${r1(last[0])}" cy="${r1(last[1])}" r="2.2" fill="${color}"/>` : ''}</svg>`);
}

// ---------------------------------------------------------------------------
// Γράφημα τιμής (κεριά ή γραμμή) με γραμμές σχεδίου TP/SL και προβολή ορίζοντα
// ---------------------------------------------------------------------------
export function priceChart(el, { candles, plan, mode = 'candles', title = 'Τιμή' }) {
  const data = (candles || []).filter((c) => c && isNum(c.c));
  if (data.length < 2) { emptyChart(el, 'Δεν υπάρχουν αρκετά δεδομένα τιμών για γράφημα.', 260); return () => {}; }
  const pl = plan && isNum(plan.entry_ref) ? plan : null;
  const horizon = pl ? Math.max(1, Math.min(40, Math.round(pl.horizon_days || 10))) : 0;

  return mountChart(el, (W) => {
    const mobile = W < 560;
    const H = mobile ? 260 : 340;
    const m = { l: 6, r: mobile ? 58 : 66, t: 14, b: 26 };
    const volH = Math.round((H - m.t - m.b) * 0.15);
    const n = data.length;
    const slots = n + (pl ? horizon + 1 : 0);
    const step = (W - m.l - m.r) / slots;
    const X = (i) => m.l + step * (i + 0.5);
    const py0 = m.t;
    const py1 = H - m.b - volH - 8;

    const lows = data.map((d) => (mode === 'line' ? d.c : (isNum(d.l) ? d.l : d.c)));
    const highs = data.map((d) => (mode === 'line' ? d.c : (isNum(d.h) ? d.h : d.c)));
    let [lo] = extent(lows);
    let [, hi] = extent(highs);
    if (pl) {
      for (const v of [pl.tp_price, pl.sl_price, pl.entry_ref]) if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    const padY = (hi - lo) * 0.06 || hi * 0.02 || 1;
    lo -= padY; hi += padY;
    const Y = linear(lo, hi, py1, py0);
    const ticks = niceTicks(lo, hi, mobile ? 4 : 5).filter((v) => v >= lo && v <= hi);

    const id = uid('pc');
    let body = '';
    // Πλέγμα
    for (const v of ticks) body += `<line class="ch-grid" x1="${m.l}" x2="${W - m.r}" y1="${r1(Y(v))}" y2="${r1(Y(v))}"/>`;

    // Όγκος
    const vols = data.map((d) => (isNum(d.v) ? d.v : 0));
    const vmax = Math.max(1, ...vols);
    const vy1 = H - m.b;
    const bw = Math.max(1, step * 0.62);
    let volPath = '';
    data.forEach((d, i) => {
      const h = (vols[i] / vmax) * volH;
      if (h > 0.3) volPath += `M${r1(X(i) - bw / 2)},${r1(vy1)}h${r1(bw)}v${r1(-h)}h${r1(-bw)}Z`;
    });
    body += `<path d="${volPath}" fill="var(--text-3)" opacity=".22"/>`;

    // Σχέδιο: ζώνες προβολής (στόχος/stop) στον μελλοντικό ορίζοντα
    const pills = [];
    if (pl) {
      const xa = X(n - 1) + step * 0.5;
      const xb = X(n + horizon);
      const ye = Y(pl.entry_ref);
      if (isNum(pl.tp_price)) {
        const yt = Y(pl.tp_price);
        body += `<rect x="${r1(xa)}" y="${r1(Math.min(yt, ye))}" width="${r1(xb - xa)}" height="${r1(Math.abs(ye - yt))}" fill="var(--pos-line)" opacity=".13"/>`;
        body += `<line x1="${m.l}" x2="${W - m.r}" y1="${r1(yt)}" y2="${r1(yt)}" stroke="var(--pos-line)" stroke-width="1.2" stroke-dasharray="5 4"/>`;
        pills.push({ y: yt, text: F.price(pl.tp_price, { unit: false }), color: 'var(--pill-pos)', label: 'Στόχος' });
      }
      if (isNum(pl.sl_price)) {
        const ys = Y(pl.sl_price);
        body += `<rect x="${r1(xa)}" y="${r1(Math.min(ys, ye))}" width="${r1(xb - xa)}" height="${r1(Math.abs(ye - ys))}" fill="var(--neg-line)" opacity=".13"/>`;
        body += `<line x1="${m.l}" x2="${W - m.r}" y1="${r1(ys)}" y2="${r1(ys)}" stroke="var(--neg-line)" stroke-width="1.2" stroke-dasharray="5 4"/>`;
        pills.push({ y: ys, text: F.price(pl.sl_price, { unit: false }), color: 'var(--pill-neg)', label: 'Stop' });
      }
      body += `<line x1="${r1(xa)}" x2="${r1(xb)}" y1="${r1(ye)}" y2="${r1(ye)}" stroke="var(--text-2)" stroke-width="1" stroke-dasharray="2 3"/>`;
      body += `<line x1="${r1(xa)}" x2="${r1(xa)}" y1="${py0}" y2="${py1}" stroke="var(--border-strong)" stroke-dasharray="2 3"/>`;
      if (xb - xa > 46) body += `<text x="${r1((xa + xb) / 2)}" y="${py0 + 10}" text-anchor="middle" font-size="10">${horizon} ημ.</text>`;
    }

    // Τιμή
    if (mode === 'line') {
      const pts = data.map((d, i) => [X(i), Y(d.c)]);
      body += `<path d="${areaPath(pts, [[X(0), py1], [X(n - 1), py1]])}" fill="url(#${id}-a)"/>`;
      body += `<path d="${linePath(pts)}" fill="none" stroke="var(--chart-price)" stroke-width="1.8" stroke-linejoin="round"/>`;
    } else {
      const cw = Math.max(1, Math.min(10, step * 0.66));
      let upW = ''; let dnW = ''; let upB = ''; let dnB = '';
      data.forEach((d, i) => {
        const o = isNum(d.o) ? d.o : d.c;
        const h = isNum(d.h) ? d.h : Math.max(o, d.c);
        const l = isNum(d.l) ? d.l : Math.min(o, d.c);
        const up = d.c >= o;
        const x = r1(X(i));
        const wick = `M${x},${r1(Y(h))}V${r1(Y(l))}`;
        const yTop = Y(Math.max(o, d.c));
        const bh = Math.max(1, Y(Math.min(o, d.c)) - yTop);
        const body_ = `M${r1(X(i) - cw / 2)},${r1(yTop)}h${r1(cw)}v${r1(bh)}h${r1(-cw)}Z`;
        if (up) { upW += wick; upB += body_; } else { dnW += wick; dnB += body_; }
      });
      body += `<path d="${upW}" stroke="var(--pos-line)" stroke-width="1" fill="none"/><path d="${dnW}" stroke="var(--neg-line)" stroke-width="1" fill="none"/>`;
      body += `<path d="${upB}" fill="var(--pos-line)"/><path d="${dnB}" fill="var(--neg-line)"/>`;
    }

    // Τελευταία τιμή
    const last = data[n - 1];
    pills.push({ y: Y(last.c), text: F.price(last.c, { unit: false }), color: 'var(--pill-price)', label: 'Τελευταία', strong: true });

    // Ετικέτες άξονα y (δεξιά), χωρίς επικάλυψη με τα «χάπια»
    pills.sort((a, b) => a.y - b.y);
    for (let i = 1; i < pills.length; i++) if (pills[i].y - pills[i - 1].y < 19) pills[i].y = pills[i - 1].y + 19;
    const ax = W - m.r + 8;
    for (const v of ticks) {
      const y = Y(v);
      const label = esc(F.num(v, v < 10 ? 2 : 1));
      if (pills.some((p) => Math.abs(p.y - y) < 12)) {
        // η θέση δεξιά είναι πιασμένη από στόχο/stop/τιμή: η ετικέτα μπαίνει αριστερά, μέσα στο γράφημα,
        // ώστε η κλίμακα να διαβάζεται πάντα (ιδίως στο κινητό)
        const tw = label.length * 6.2 + 8;
        body += `<g class="tick-left"><rect x="${m.l + 2}" y="${r1(y - 8)}" width="${r1(tw)}" height="16" rx="4"/><text x="${m.l + 6}" y="${r1(y + 3.5)}">${label}</text></g>`;
        continue;
      }
      body += `<text x="${ax}" y="${r1(y + 3.5)}">${label}</text>`;
    }
    for (const p of pills) {
      const tw = p.text.length * 6.4 + 12;
      body += `<g class="price-label"><rect x="${ax - 4}" y="${r1(p.y - 9)}" width="${r1(tw)}" height="18" rx="5" fill="${p.color}"/><text x="${ax + 2}" y="${r1(p.y + 3.8)}" style="fill:var(--pill-ink);font-weight:650">${esc(p.text)}</text></g>`;
    }

    // Άξονας x
    const dts = dateTicks(data.map((d) => d.d), X, W - m.l - m.r, mobile ? 52 : 70);
    for (const t of dts) body += `<text x="${r1(t.x)}" y="${H - 8}" text-anchor="middle">${esc(t.label)}</text>`;
    body += `<line class="ch-zero" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    body += `<rect class="ch-overlay" x="${m.l}" y="${py0}" width="${W - m.l - m.r}" height="${H - m.b - py0}"/>`;

    const defs = `<linearGradient id="${id}-a" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--chart-price)" stop-opacity=".22"/><stop offset="1" stop-color="var(--chart-price)" stop-opacity="0"/></linearGradient>`;
    const first = data[0];
    const chg = last.c / first.c - 1;
    const label = `${title}: ${n} συνεδριάσεις από ${F.date(first.d)} έως ${F.date(last.d)}, κλείσιμο ${F.price(last.c)} (${F.pct(chg, { digits: 1, sign: true })} στην περίοδο)${pl ? `. Στόχος ${F.price(pl.tp_price)}, stop ${F.price(pl.sl_price)}` : ''}.`;
    frame(el, { W, H, label, body, defs });

    const xs = data.map((_, i) => X(i));
    bindHover(el, {
      xs,
      plot: { left: m.l, right: W - m.r, top: py0 },
      show: (i) => {
        const d = data[i];
        const prev = i > 0 ? data[i - 1].c : null;
        const ch = prev ? d.c / prev - 1 : null;
        const x = xs[i];
        const y = Y(d.c);
        const svg = `<line class="ch-guide" x1="${r1(x)}" x2="${r1(x)}" y1="${py0}" y2="${H - m.b}"/><line class="ch-guide" x1="${m.l}" x2="${W - m.r}" y1="${r1(y)}" y2="${r1(y)}"/><circle class="ch-dot" cx="${r1(x)}" cy="${r1(y)}" r="3.5" fill="var(--chart-price)"/>`;
        let tip = `<div class="tip-date">${esc(F.date(d.d, 'long'))}</div>`;
        if (mode !== 'line' && isNum(d.o)) {
          tip += tipRow('Άνοιγμα', F.price(d.o)) + tipRow('Υψηλό', F.price(d.h)) + tipRow('Χαμηλό', F.price(d.l));
        }
        tip += tipRow('Κλείσιμο', F.price(d.c));
        if (ch != null) tip += tipRow('Μεταβολή', F.pct(ch, { digits: 2, sign: true }), '', F.tone(ch));
        if (isNum(d.v)) tip += tipRow('Όγκος', F.compact(d.v));
        return { x, svg, tip };
      },
    });
  });
}

// ---------------------------------------------------------------------------
// Στοιβαγμένη περιοχή: ιστορικό πιθανοτήτων της σφαίρας
// ---------------------------------------------------------------------------
export function orbAreaChart(el, rows) {
  const data = (rows || []).map((r) => {
    if (!r) return null;
    let a = F.toNum(r.p_tp); let b = F.toNum(r.p_sl); let c = F.toNum(r.p_to);
    if (a == null || b == null) return null;
    c = c ?? Math.max(0, 1 - a - b);
    const s = a + b + c || 1;
    return { d: r.d, tp: a / s, sl: b / s, to: c / s };
  }).filter(Boolean);
  if (data.length < 2) { emptyChart(el, 'Δεν υπάρχει ακόμη ιστορικό προβλέψεων για αυτή τη μετοχή.', 200); return () => {}; }

  return mountChart(el, (W) => {
    const mobile = W < 560;
    const H = mobile ? 190 : 250;
    const m = { l: 6, r: 44, t: 10, b: 26 };
    const n = data.length;
    const X = linear(0, n - 1, m.l, W - m.r);
    const Y = linear(0, 1, H - m.b, m.t);
    const tpTop = data.map((d, i) => [X(i), Y(d.tp)]);
    const toTop = data.map((d, i) => [X(i), Y(d.tp + d.to)]);
    const base = [[X(0), Y(0)], [X(n - 1), Y(0)]];
    const top = [[X(0), Y(1)], [X(n - 1), Y(1)]];
    let body = '';
    body += `<path d="${areaPath(tpTop, base)}" fill="var(--pos-line)" opacity=".62"/>`;
    body += `<path d="${areaPath(toTop, tpTop)}" fill="var(--to-line)" opacity=".38"/>`;
    body += `<path d="${areaPath(top, toTop)}" fill="var(--neg-line)" opacity=".55"/>`;
    body += `<path d="${linePath(tpTop)}" fill="none" stroke="var(--pos-line)" stroke-width="1.5"/>`;
    body += `<path d="${linePath(toTop)}" fill="none" stroke="var(--neg-line)" stroke-width="1.5"/>`;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      body += `<line class="ch-grid" x1="${m.l}" x2="${W - m.r}" y1="${r1(Y(v))}" y2="${r1(Y(v))}" style="stroke:rgba(255,255,255,.14)"/>`;
      body += `<text x="${W - m.r + 8}" y="${r1(Y(v) + 3.5)}">${F.pct(v, { digits: 0 })}</text>`;
    }
    for (const t of dateTicks(data.map((d) => d.d), X, W - m.l - m.r, mobile ? 52 : 70)) body += `<text x="${r1(t.x)}" y="${H - 8}" text-anchor="middle">${esc(t.label)}</text>`;
    body += `<rect class="ch-overlay" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/>`;
    const last = data[n - 1];
    const label = `Ιστορικό σφαίρας για ${n} συνεδριάσεις. Τελευταία: στόχος ${F.prob(last.tp)}, λήξη ${F.prob(last.to)}, stop ${F.prob(last.sl)}.`;
    frame(el, { W, H, label, body });
    const xs = data.map((_, i) => X(i));
    bindHover(el, {
      xs,
      plot: { left: m.l, right: W - m.r, top: m.t },
      show: (i) => {
        const d = data[i];
        const x = xs[i];
        const svg = `<line class="ch-guide" x1="${r1(x)}" x2="${r1(x)}" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--text)"/><circle class="ch-dot" cx="${r1(x)}" cy="${r1(Y(d.tp))}" r="3.5" fill="var(--pos-line)"/><circle class="ch-dot" cx="${r1(x)}" cy="${r1(Y(d.tp + d.to))}" r="3.5" fill="var(--neg-line)"/>`;
        const tip = `<div class="tip-date">${esc(F.date(d.d, 'long'))}</div>${tipRow(F.STATES.tp.label, F.prob(d.tp), swatch('var(--pos-line)'))}${tipRow(F.STATES.to.label, F.prob(d.to), swatch('var(--to-line)'))}${tipRow(F.STATES.sl.label, F.prob(d.sl), swatch('var(--neg-line)'))}`;
        return { x, svg, tip };
      },
    });
  });
}

// ---------------------------------------------------------------------------
// Καμπύλες κεφαλαίου: στρατηγική vs αγορά & διακράτηση vs τυχαίες (διάμεσος + ζώνη p05–p95)
// ---------------------------------------------------------------------------
export const EQUITY_SERIES = [
  { key: 'strategy', label: 'Στρατηγική', color: 'var(--accent-strong)', width: 2.2 },
  { key: 'buy_hold', label: 'Δείκτης (αγορά & διακράτηση)', color: 'var(--chart-bh)', width: 1.6 },
  { key: 'buy_hold_universe', label: 'Ισοσταθμική λίστα', color: 'var(--chart-ew)', width: 1.4, dash: '1 3' },
  { key: 'random_median', label: 'Τυχαίες είσοδοι (διάμεσος)', color: 'var(--chart-rand)', width: 1.5, dash: '5 4' },
];

export function equityChart(el, rows, { capital = null } = {}) {
  const data = (rows || []).filter((r) => r && r.d);
  if (data.length < 2) { emptyChart(el, 'Δεν υπάρχουν δεδομένα καμπύλης κεφαλαίου.', 280); return () => {}; }
  const hasBand = data.some((r) => isNum(r.random_p05) && isNum(r.random_p95));

  return mountChart(el, (W) => {
    const mobile = W < 560;
    const H = mobile ? 250 : 330;
    const m = { l: 6, r: mobile ? 62 : 74, t: 12, b: 26 };
    const n = data.length;
    const X = linear(0, n - 1, m.l, W - m.r);
    const vals = [];
    for (const r of data) for (const k of ['strategy', 'buy_hold', 'buy_hold_universe', 'random_median', 'random_p05', 'random_p95']) if (isNum(r[k])) vals.push(r[k]);
    let [lo, hi] = extent(vals);
    const pad = (hi - lo) * 0.05 || 1;
    lo -= pad; hi += pad;
    const Y = linear(lo, hi, H - m.b, m.t);
    const ticks = niceTicks(lo, hi, mobile ? 4 : 5).filter((v) => v >= lo && v <= hi);
    let body = '';
    for (const v of ticks) {
      body += `<line class="ch-grid" x1="${m.l}" x2="${W - m.r}" y1="${r1(Y(v))}" y2="${r1(Y(v))}"/>`;
      body += `<text x="${W - m.r + 8}" y="${r1(Y(v) + 3.5)}">${esc(F.eur(v, { digits: 0, compact: v >= 100000 }))}</text>`;
    }
    if (isNum(capital) && capital > lo && capital < hi) body += `<line x1="${m.l}" x2="${W - m.r}" y1="${r1(Y(capital))}" y2="${r1(Y(capital))}" stroke="var(--text-3)" stroke-dasharray="2 4"/>`;
    if (hasBand) {
      const up = data.map((r, i) => [X(i), isNum(r.random_p95) ? Y(r.random_p95) : null]);
      const dn = data.map((r, i) => [X(i), isNum(r.random_p05) ? Y(r.random_p05) : null]);
      body += `<path d="${areaPath(up, dn)}" fill="var(--chart-rand)" opacity=".16"/>`;
    }
    for (const s of EQUITY_SERIES.slice().reverse()) {
      const pts = data.map((r, i) => [X(i), isNum(r[s.key]) ? Y(r[s.key]) : null]);
      body += `<path d="${linePath(pts)}" fill="none" stroke="${s.color}" stroke-width="${s.width}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} stroke-linejoin="round"/>`;
    }
    for (const t of dateTicks(data.map((d) => d.d), X, W - m.l - m.r, mobile ? 46 : 64)) body += `<text x="${r1(t.x)}" y="${H - 8}" text-anchor="middle">${esc(t.label)}</text>`;
    body += `<rect class="ch-overlay" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/>`;
    const last = data[n - 1];
    const label = `Καμπύλες κεφαλαίου από ${F.date(data[0].d)} έως ${F.date(last.d)}: στρατηγική ${F.eur(last.strategy, { digits: 0 })}, δείκτης ${F.eur(last.buy_hold, { digits: 0 })}${isNum(last.buy_hold_universe) ? `, ισοσταθμική λίστα ${F.eur(last.buy_hold_universe, { digits: 0 })}` : ''}, διάμεσος τυχαίων ${F.eur(last.random_median, { digits: 0 })}.`;
    frame(el, { W, H, label, body });
    const xs = data.map((_, i) => X(i));
    const base = isNum(capital) ? capital : (isNum(data[0].strategy) ? data[0].strategy : null);
    bindHover(el, {
      xs,
      plot: { left: m.l, right: W - m.r, top: m.t },
      show: (i) => {
        const r = data[i];
        const x = xs[i];
        let svg = `<line class="ch-guide" x1="${r1(x)}" x2="${r1(x)}" y1="${m.t}" y2="${H - m.b}"/>`;
        let tip = `<div class="tip-date">${esc(F.date(r.d, 'long'))}</div>`;
        for (const s of EQUITY_SERIES) {
          if (!isNum(r[s.key])) continue;
          svg += `<circle class="ch-dot" cx="${r1(x)}" cy="${r1(Y(r[s.key]))}" r="3.5" fill="${s.color}"/>`;
          const ret = base ? r[s.key] / base - 1 : null;
          tip += tipRow(s.label, `${F.eur(r[s.key], { digits: 0 })}${ret != null ? ` (${F.pct(ret, { digits: 1, sign: true })})` : ''}`, swatch(s.color, true));
        }
        if (isNum(r.random_p05) && isNum(r.random_p95)) tip += tipRow('Ζώνη τυχαίων 5–95%', `${F.eur(r.random_p05, { digits: 0 })} – ${F.eur(r.random_p95, { digits: 0 })}`, swatch('var(--chart-rand)'));
        return { x, svg, tip };
      },
    });
  });
}

// ---------------------------------------------------------------------------
// Drawdown
// ---------------------------------------------------------------------------
export function drawdownChart(el, rows) {
  const data = (rows || []).filter((r) => r && r.d && isNum(r.strategy));
  if (data.length < 2) { emptyChart(el, 'Δεν υπάρχουν δεδομένα πτώσεων.', 160); return () => {}; }
  return mountChart(el, (W) => {
    const H = W < 560 ? 150 : 170;
    const m = { l: 6, r: 54, t: 8, b: 26 };
    const n = data.length;
    const X = linear(0, n - 1, m.l, W - m.r);
    const [lo] = extent(data.map((r) => r.strategy));
    const minV = Math.min(-0.01, lo * 1.08);
    const Y = linear(minV, 0, H - m.b, m.t);
    const pts = data.map((r, i) => [X(i), Y(Math.min(0, r.strategy))]);
    const id = uid('dd');
    let body = '';
    const ticks = niceTicks(minV, 0, 3).filter((v) => v >= minV && v <= 0);
    for (const v of ticks) {
      body += `<line class="ch-grid" x1="${m.l}" x2="${W - m.r}" y1="${r1(Y(v))}" y2="${r1(Y(v))}"/>`;
      body += `<text x="${W - m.r + 8}" y="${r1(Y(v) + 3.5)}">${esc(F.pct(v, { digits: 0 }))}</text>`;
    }
    body += `<path d="${areaPath(pts, [[X(0), Y(0)], [X(n - 1), Y(0)]])}" fill="url(#${id})"/>`;
    body += `<path d="${linePath(pts)}" fill="none" stroke="var(--neg-line)" stroke-width="1.4"/>`;
    body += `<line class="ch-zero" x1="${m.l}" x2="${W - m.r}" y1="${r1(Y(0))}" y2="${r1(Y(0))}"/>`;
    // Σημείο μέγιστης πτώσης
    let wi = 0;
    data.forEach((r, i) => { if (r.strategy < data[wi].strategy) wi = i; });
    body += `<circle cx="${r1(X(wi))}" cy="${r1(Y(data[wi].strategy))}" r="3.5" fill="var(--neg-line)" stroke="var(--surface)" stroke-width="2"/>`;
    for (const t of dateTicks(data.map((d) => d.d), X, W - m.l - m.r, W < 560 ? 46 : 64)) body += `<text x="${r1(t.x)}" y="${H - 8}" text-anchor="middle">${esc(t.label)}</text>`;
    body += `<rect class="ch-overlay" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/>`;
    const defs = `<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--neg-line)" stop-opacity=".08"/><stop offset="1" stop-color="var(--neg-line)" stop-opacity=".38"/></linearGradient>`;
    const label = `Πτώση από την κορυφή (drawdown). Μέγιστη ${F.pct(data[wi].strategy, { digits: 1 })} στις ${F.date(data[wi].d)}.`;
    frame(el, { W, H, label, body, defs });
    const xs = data.map((_, i) => X(i));
    bindHover(el, {
      xs,
      plot: { left: m.l, right: W - m.r, top: m.t },
      show: (i) => {
        const r = data[i];
        const x = xs[i];
        const svg = `<line class="ch-guide" x1="${r1(x)}" x2="${r1(x)}" y1="${m.t}" y2="${H - m.b}"/><circle class="ch-dot" cx="${r1(x)}" cy="${r1(Y(Math.min(0, r.strategy)))}" r="3.5" fill="var(--neg-line)"/>`;
        const tip = `<div class="tip-date">${esc(F.date(r.d, 'long'))}</div>${tipRow('Πτώση από κορυφή', F.pct(r.strategy, { digits: 2 }), '', r.strategy < 0 ? 'neg' : '')}`;
        return { x, svg, tip, tipY: 4 };
      },
    });
  });
}

// ---------------------------------------------------------------------------
// Βαθμονόμηση: προβλεπόμενη vs πραγματική συχνότητα στόχου, με διαγώνιο αναφοράς
// ---------------------------------------------------------------------------
export function calibrationChart(el, bins) {
  const data = (bins || []).filter((b) => b && isNum(b.predicted) && isNum(b.realized)).sort((a, b) => a.predicted - b.predicted);
  if (!data.length) { emptyChart(el, 'Δεν υπάρχουν αρκετές επιλυμένες προβλέψεις για βαθμονόμηση.', 240); return () => {}; }
  return mountChart(el, (W) => {
    const H = Math.round(Math.min(320, Math.max(240, W * 0.62)));
    const m = { l: 58, r: 14, t: 12, b: 40 };
    let hiV = 0;
    for (const b of data) hiV = Math.max(hiV, b.predicted, b.realized, isNum(b.bin_hi) ? b.bin_hi : 0);
    hiV = Math.min(1, Math.ceil((hiV + 0.04) * 10) / 10);
    const X = linear(0, hiV, m.l, W - m.r);
    const Y = linear(0, hiV, H - m.b, m.t);
    const maxN = Math.max(1, ...data.map((b) => (isNum(b.n) ? b.n : 1)));
    let body = '';
    const ticks = niceTicks(0, hiV, 5).filter((v) => v <= hiV + 1e-9);
    for (const v of ticks) {
      body += `<line class="ch-grid" x1="${m.l}" x2="${W - m.r}" y1="${r1(Y(v))}" y2="${r1(Y(v))}"/>`;
      body += `<line class="ch-grid" x1="${r1(X(v))}" x2="${r1(X(v))}" y1="${m.t}" y2="${H - m.b}"/>`;
      body += `<text x="${m.l - 8}" y="${r1(Y(v) + 3.5)}" text-anchor="end">${esc(F.pct(v, { digits: 0 }))}</text>`;
      body += `<text x="${r1(X(v))}" y="${H - m.b + 16}" text-anchor="middle">${esc(F.pct(v, { digits: 0 }))}</text>`;
    }
    body += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(hiV)}" y2="${Y(hiV)}" stroke="var(--text-3)" stroke-dasharray="4 4"/>`;
    const pts = data.map((b) => [X(b.predicted), Y(b.realized)]);
    body += `<path d="${linePath(pts)}" fill="none" stroke="var(--accent-strong)" stroke-width="1.6"/>`;
    data.forEach((b, i) => {
      const r = 3.5 + 7 * Math.sqrt((isNum(b.n) ? b.n : 1) / maxN);
      body += `<circle cx="${r1(pts[i][0])}" cy="${r1(pts[i][1])}" r="${r1(r)}" fill="var(--accent-strong)" fill-opacity=".28" stroke="var(--accent-strong)" stroke-width="1.4"/>`;
    });
    body += `<text x="${r1((m.l + W - m.r) / 2)}" y="${H - 6}" text-anchor="middle">Προβλεπόμενη P(στόχος)</text>`;
    body += `<text transform="translate(13 ${r1((m.t + H - m.b) / 2)}) rotate(-90)" text-anchor="middle">Πραγματική συχνότητα</text>`;
    body += `<rect class="ch-overlay" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/>`;
    const label = `Βαθμονόμηση πιθανοτήτων σε ${data.length} ομάδες: όσο πιο κοντά στη διαγώνιο, τόσο πιο αξιόπιστες οι πιθανότητες.`;
    frame(el, { W, H, label, body });
    bindHover(el, {
      xs: pts.map((p) => p[0]),
      plot: { left: m.l, right: W - m.r, top: m.t },
      show: (i) => {
        const b = data[i];
        const [x, y] = pts[i];
        const svg = `<circle cx="${r1(x)}" cy="${r1(y)}" r="${r1(4 + 7 * Math.sqrt((isNum(b.n) ? b.n : 1) / maxN) + 3)}" fill="none" stroke="var(--text)" stroke-width="1.5"/>`;
        const diff = b.realized - b.predicted;
        const range = isNum(b.bin_lo) && isNum(b.bin_hi) ? `${F.prob(b.bin_lo)}–${F.prob(b.bin_hi)}` : F.prob(b.predicted);
        const tip = `<div class="tip-date">Ομάδα ${esc(range)}</div>${tipRow('Προβλεπόμενη', F.pct(b.predicted, { digits: 1 }))}${tipRow('Πραγματική', F.pct(b.realized, { digits: 1 }))}${tipRow('Απόκλιση', F.pct(diff, { digits: 1, sign: true }), '', Math.abs(diff) < 0.03 ? 'pos' : 'neg')}${tipRow('Δείγματα', F.int(b.n))}`;
        return { x, svg, tip, tipY: Math.max(4, y - 60) };
      },
    });
  });
}

