// Θέμα (σκούρο/φωτεινό/σύστημα) και προτίμηση μειωμένης κίνησης.
import { storage, createBus } from './dom.js';

export const themeBus = createBus();
const mq = window.matchMedia ? matchMedia('(prefers-color-scheme: light)') : null;

/** Η αποθηκευμένη προτίμηση: 'dark' | 'light' | 'system'. */
export function themePref() {
  const t = storage.get('theme', null);
  return t === 'light' || t === 'dark' ? t : 'system';
}

export function effectiveTheme() {
  const p = themePref();
  if (p !== 'system') return p;
  return mq && mq.matches ? 'light' : 'dark';
}

export function applyTheme() {
  const t = effectiveTheme();
  document.documentElement.setAttribute('data-theme', t);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', t === 'light' ? '#f3f5fa' : '#070b15');
  themeBus.emit('theme', t);
  return t;
}

export function setThemePref(pref) {
  if (pref === 'light' || pref === 'dark') storage.set('theme', pref);
  else storage.remove('theme');
  return applyTheme();
}

if (mq && mq.addEventListener) mq.addEventListener('change', () => { if (themePref() === 'system') applyTheme(); });

export function motionPref() {
  return storage.get('motion', 'auto') === 'reduce' ? 'reduce' : 'auto';
}

export function setMotionPref(pref) {
  if (pref === 'reduce') {
    storage.set('motion', 'reduce');
    document.documentElement.dataset.motion = 'reduce';
  } else {
    storage.remove('motion');
    delete document.documentElement.dataset.motion;
  }
}

export function initMotion() {
  if (motionPref() === 'reduce') document.documentElement.dataset.motion = 'reduce';
}
