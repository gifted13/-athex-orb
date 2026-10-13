// Βοηθητικά για DOM, ασφαλή HTML templates και αποθήκευση στον browser.

/** Τιμή HTML που δεν πρέπει να «δραπετευτεί» ξανά. */
export class Raw {
  constructor(s) { this.s = s; }
  toString() { return this.s; }
}

export const raw = (s) => new Raw(s == null ? '' : String(s));

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

function value(v) {
  if (v == null) return '';
  if (v === true || v === false) return String(v); // π.χ. aria-pressed="${fav}"
  if (v instanceof Raw) return v.s;
  if (Array.isArray(v)) return v.map(value).join('');
  return esc(String(v));
}

/** Tagged template: όλες οι τιμές γίνονται escape εκτός από Raw. */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += value(values[i]) + strings[i + 1];
  return new Raw(out);
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function setHTML(el, content) {
  if (el) el.innerHTML = value(content);
}

export function debounce(fn, ms) {
  let t = null;
  const d = (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
  d.cancel = () => clearTimeout(t);
  return d;
}

let uidCounter = 0;
export const uid = (prefix = 'u') => `${prefix}${++uidCounter}`;

export const isMobile = () => window.matchMedia && matchMedia('(max-width: 759px)').matches;
export const isDrawerLayout = () => window.matchMedia && matchMedia('(max-width: 1099px)').matches;

/** localStorage τυλιγμένο σε try/catch (ιδιωτική περιήγηση, μπλοκαρισμένα cookies κ.λπ.). */
export const storage = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem(`orb.${key}`);
      return v == null ? fallback : JSON.parse(v);
    } catch {
      return fallback;
    }
  },
  set(key, val) {
    try { localStorage.setItem(`orb.${key}`, JSON.stringify(val)); } catch { /* αγνοείται */ }
  },
  remove(key) {
    try { localStorage.removeItem(`orb.${key}`); } catch { /* αγνοείται */ }
  },
};

/** Μικρό event bus. */
export function createBus() {
  const target = new EventTarget();
  return {
    on(type, fn) {
      const h = (e) => fn(e.detail);
      target.addEventListener(type, h);
      return () => target.removeEventListener(type, h);
    },
    emit(type, detail) { target.dispatchEvent(new CustomEvent(type, { detail })); },
  };
}

/** Είναι το στοιχείο πεδίο κειμένου (για να μην «κλέβουν» τα shortcuts πληκτρολόγηση); */
export function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}
