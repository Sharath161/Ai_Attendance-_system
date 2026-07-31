/* Small DOM helpers shared by every view. */
export const $ = sel => document.querySelector(sel);

export const el = html => {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstChild;
};

export const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export const render = html => { $('#content').innerHTML = html; };

export function toast(message, kind = '') {
  const t = $('#toast');
  t.textContent = message;
  t.className = `toast show ${kind}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = 'toast'; }, 3200);
}

export function message(selector, text, kind = '') {
  const node = $(selector);
  if (!node) return;
  node.className = `msg ${kind}`;
  node.textContent = text;
}

/** Percentage bar colour band: green >= 75, amber >= 50, red below. */
export const barClass = pct => (pct >= 75 ? '' : pct >= 50 ? 'mid' : 'low');

export const fmtDate = iso => new Date(iso).toLocaleString();
export const fmtTime = iso => new Date(iso).toLocaleTimeString();

export function dataURLtoBlob(dataURL) {
  const [head, b64] = dataURL.split(',');
  const mime = head.match(/:(.*?);/)[1];
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}
