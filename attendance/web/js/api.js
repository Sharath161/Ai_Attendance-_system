/* HTTP client for the attendance API — owns the token and error shape. */
export const store = {
  get token() { return localStorage.getItem('att_token') || ''; },
  set token(v) { v ? localStorage.setItem('att_token', v) : localStorage.removeItem('att_token'); },
  get user() { return JSON.parse(localStorage.getItem('att_user') || 'null'); },
  set user(v) { v ? localStorage.setItem('att_user', JSON.stringify(v)) : localStorage.removeItem('att_user'); },
};

let onUnauthorized = () => {};
export const setUnauthorizedHandler = fn => { onUnauthorized = fn; };

export function authHeader() {
  return store.token ? { Authorization: `Bearer ${store.token}` } : {};
}

export async function api(path, { method = 'GET', body, formData } = {}) {
  const headers = { ...authHeader() };
  if (body && !formData) headers['Content-Type'] = 'application/json';

  const res = await fetch(`/api${path}`, {
    method,
    headers,
    body: formData ? body : (body ? JSON.stringify(body) : undefined),
  });

  // A 401 from the login endpoint means bad credentials — surface the server's
  // message. A 401 anywhere else means the token is no longer valid.
  if (res.status === 401 && !path.startsWith('/auth/login')) {
    onUnauthorized();
    throw new Error('Session expired');
  }

  const isJson = (res.headers.get('content-type') || '').includes('json');
  const data = isJson ? await res.json() : await res.text();
  if (!res.ok) throw new Error(data?.error?.message || data?.detail || `Error ${res.status}`);
  return data;
}

export const get = p => api(p);
export const post = (p, body) => api(p, { method: 'POST', body });
export const del = p => api(p, { method: 'DELETE' });
