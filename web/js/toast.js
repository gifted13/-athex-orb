// Μη-μπλοκαριστικές ειδοποιήσεις (aria-live).
import { html, setHTML } from './dom.js';
import { icon } from './icons.js';

const ICONS = { success: 'check', error: 'alert', info: 'info' };
const MAX = 4;

/**
 * toast('Μήνυμα', { type: 'success'|'error'|'info', action: { label, onClick }, timeout })
 */
export function toast(message, { type = 'info', action = null, timeout } = {}) {
  const host = document.getElementById('toasts');
  if (!host) return () => {};
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.setAttribute('role', type === 'error' ? 'alert' : 'status');
  setHTML(el, html`${icon(ICONS[type] || 'info')}<span class="toast-msg">${message}</span>
    ${action ? html`<button type="button" class="btn btn-sm btn-ghost toast-action">${action.label}</button>` : ''}
    <button type="button" class="icon-btn toast-close" aria-label="Κλείσιμο ειδοποίησης" style="width:28px;height:28px;border:0;background:transparent">${icon('x', 'icon-sm')}</button>`);
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    el.classList.add('out');
    setTimeout(() => el.remove(), 220);
  };
  el.querySelector('.toast-close').addEventListener('click', close);
  if (action) {
    el.querySelector('.toast-action').addEventListener('click', () => {
      try { action.onClick(); } finally { close(); }
    });
  }
  host.appendChild(el);
  while (host.children.length > MAX) host.firstElementChild.remove();
  const ms = timeout ?? (type === 'error' ? 6000 : action ? 5500 : 3200);
  let t = setTimeout(close, ms);
  el.addEventListener('pointerenter', () => clearTimeout(t));
  el.addEventListener('pointerleave', () => { t = setTimeout(close, 1800); });
  return close;
}
