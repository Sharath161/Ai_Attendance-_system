/* Offline shell for the Attendance PWA.
 *
 * Strategy:
 *   - /api/*   never cached — attendance data must always be live
 *   - shell    network-first, falling back to cache when offline.
 *
 * Cache-first would pin users to stale JavaScript after a deploy (a code fix
 * would stay invisible until the cache name changed), so the network wins
 * whenever it is reachable and the cache exists purely for offline use.
 */
const CACHE = 'attendance-v2';
const SHELL = ['/', '/css/styles.css', '/js/main.js', '/js/api.js', '/js/ui.js',
               '/js/camera.js', '/js/views/student.js', '/js/views/staff.js',
               '/js/views/admin.js', '/manifest.json', '/icon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api/') || e.request.method !== 'GET') return;  // never cache API
  e.respondWith(
    fetch(e.request)
      .then(res => {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(e.request).then(hit => hit || caches.match('/')))
  );
});
