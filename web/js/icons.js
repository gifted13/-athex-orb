// Εικονίδια SVG (χειροποίητα, stroke = currentColor).
import { raw } from './dom.js';

const P = {
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  star: '<path d="M12 3.6l2.55 5.17 5.7.83-4.12 4.02.97 5.68L12 16.6l-5.1 2.7.97-5.68L3.75 9.6l5.7-.83z"/>',
  starFill: '<path fill="currentColor" d="M12 3.6l2.55 5.17 5.7.83-4.12 4.02.97 5.68L12 16.6l-5.1 2.7.97-5.68L3.75 9.6l5.7-.83z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M4.6 4.6 6 6M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4"/>',
  moon: '<path d="M20 14.6A8.2 8.2 0 0 1 9.4 4a8.2 8.2 0 1 0 10.6 10.6z"/>',
  refresh: '<path d="M20 11.5A8 8 0 0 0 6 6.3L4 8.5"/><path d="M4 4v4.5h4.5"/><path d="M4 12.5A8 8 0 0 0 18 17.7l2-2.2"/><path d="M20 20v-4.5h-4.5"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  ban: '<circle cx="12" cy="12" r="8.5"/><path d="m6 6 12 12"/>',
  chevronDown: '<path d="m6 9 6 6 6-6"/>',
  chevronLeft: '<path d="m15 6-6 6 6 6"/>',
  chevronRight: '<path d="m9 6 6 6-6 6"/>',
  sort: '<path d="M8 4v16M4 8l4-4 4 4M16 20V4M12 16l4 4 4-4"/>',
  market: '<path d="M3.5 3.5v17h17"/><path d="m7.5 14.5 3.5-4 3 3 5-6.5"/>',
  signals: '<path d="M13 2.5 4.5 13.5h6.5l-1 8 8.5-11H12z"/>',
  backtest: '<path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1L3.5 8.5"/><path d="M3.5 3.5v5h5"/><path d="M12 7.5V12l3 2"/>',
  journal: '<path d="M5 4.5h10.5a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3z"/><path d="M5 17.5a3 3 0 0 1 3-3h10.5"/><path d="M9 8.5h6"/>',
  settings: '<path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17" r="2"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.8v.2"/>',
  alert: '<path d="M12 3.8 2.8 19.5h18.4z"/><path d="M12 10v4.5M12 17.3v.2"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  database: '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"/><path d="M4.5 5.5v13c0 1.55 3.36 2.8 7.5 2.8s7.5-1.25 7.5-2.8v-13"/><path d="M4.5 12c0 1.55 3.36 2.8 7.5 2.8s7.5-1.25 7.5-2.8"/>',
  play: '<path d="M7.5 5v14l11-7z"/>',
  trash: '<path d="M4.5 7h15M10 11v6M14 11v6M6 7l1 12.5h10L18 7M9 7V4.5h6V7"/>',
  external: '<path d="M14 4.5h5.5V10M19.5 4.5 11 13M18 14v5.5H4.5V6H10"/>',
  sparkle: '<path d="M12 3.5 13.8 10 20.5 12l-6.7 2-1.8 6.5-1.8-6.5L3.5 12l6.7-2z"/>',
  list: '<path d="M8.5 6.5h11M8.5 12h11M8.5 17.5h11M4.5 6.5h.01M4.5 12h.01M4.5 17.5h.01"/>',
  wifiOff: '<path d="M3 3l18 18"/><path d="M8.5 16.5a5 5 0 0 1 7 0M5 13a10 10 0 0 1 4.5-2.6M14.5 10.4A10 10 0 0 1 19 13M2 9.5a15 15 0 0 1 4.4-2.8M10.7 5.6A15 15 0 0 1 22 9.5"/><path d="M12 20h.01"/>',
  keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2.5"/><path d="M6.5 10h.01M10 10h.01M14 10h.01M17.5 10h.01M7 14h10"/>',
  layers: '<path d="m12 3.5 9 4.8-9 4.8-9-4.8z"/><path d="m3 12.2 9 4.8 9-4.8"/><path d="m3 16.2 9 4.8 9-4.8"/>',
  wallet: '<path d="M4 7.5h14.5a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5.5A2 2 0 0 1 3.5 17.5v-11A2 2 0 0 1 5.5 4.5h11"/><path d="M16 13.5h.01"/>',
  coins: '<ellipse cx="9" cy="7" rx="5.5" ry="2.5"/><path d="M3.5 7v4c0 1.4 2.5 2.5 5.5 2.5s5.5-1.1 5.5-2.5V7"/><path d="M9.5 16.4c.6.1 1.2.1 1.8.1 3 0 5.5-1.1 5.5-2.5"/><path d="M14.5 10.6c3 0 6 1 6 2.4v4c0 1.4-2.5 2.5-5.5 2.5s-5.5-1.1-5.5-2.5"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9.5 9.5h5v5h-5zM9 2.5v3M15 2.5v3M9 18.5v3M15 18.5v3M2.5 9h3M2.5 15h3M18.5 9h3M18.5 15h3"/>',
  // Οι έξι πράκτορες
  scout: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><path d="M12 12 18 6"/><circle cx="12" cy="12" r=".8" fill="currentColor"/>',
  compass: '<circle cx="12" cy="12" r="8.5"/><path d="m15.5 8.5-2.2 4.8-4.8 2.2 2.2-4.8z"/>',
  scale: '<path d="M12 4v16M8 20h8M5 7.5h14M12 4.5v3"/><path d="M5 7.5 2.5 13.5a3 3 0 0 0 5 0z"/><path d="M19 7.5l-2.5 6a3 3 0 0 0 5 0z"/>',
  sigma: '<path d="M17.5 5H6.5l6 7-6 7h11"/>',
  shield: '<path d="M12 3.2 4.8 6v5.6c0 4.4 3 8.1 7.2 9.2 4.2-1.1 7.2-4.8 7.2-9.2V6z"/><path d="m9 12 2.2 2.2L15.2 10"/>',
  target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5"/><path d="M12 2v3.5M12 18.5V22M2 12h3.5M18.5 12H22"/>',
};

export const AGENT_ICONS = { scout: 'scout', market: 'compass', judge: 'scale', quant: 'sigma', guard: 'shield', executor: 'target' };

/** Επιστρέφει Raw SVG. */
export function icon(name, cls = 'icon') {
  const body = P[name] || P.info;
  return raw(`<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${body}</svg>`);
}
