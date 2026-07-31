/* Application shell: session handling, role-aware tab routing, boot. */
import { get, post, setUnauthorizedHandler, store } from './api.js';
import { $, el, esc, message } from './ui.js';
import { initCamera } from './camera.js';
import { viewCheckin, viewFace, viewHome } from './views/student.js';
import { viewCourses, viewDash, viewKiosk, viewReports, viewSessions } from './views/staff.js';
import { viewUsers } from './views/admin.js';

const VIEWS = {
  home: viewHome, checkin: viewCheckin, face: viewFace,
  dash: viewDash, kiosk: viewKiosk, sessions: viewSessions,
  courses: viewCourses, reports: viewReports, users: viewUsers,
};

const TABS = {
  student: [['home', '◎', 'My attendance'], ['checkin', '⊙', 'Check in'], ['face', '☺', 'My face']],
  staff: [['dash', '▦', 'Dashboard'], ['kiosk', '⊙', 'Kiosk'], ['sessions', '▤', 'Sessions'],
          ['courses', '◈', 'Courses'], ['reports', '◫', 'Reports']],
};

// ── session ──────────────────────────────────────────────────────────────────
function signOut() {
  store.token = null;
  store.user = null;
  $('#view-main').classList.remove('active');
  $('#view-login').classList.add('active');
}
setUnauthorizedHandler(signOut);

function enterApp() {
  const user = store.user;
  $('#view-login').classList.remove('active');
  $('#view-main').classList.add('active');
  $('#userName').textContent = user.full_name;
  $('#userRole').textContent = user.role;
  buildTabs(user.role);
  go(user.role === 'student' ? 'home' : 'dash');
}

// ── navigation ───────────────────────────────────────────────────────────────
function buildTabs(role) {
  const tabs = [...(role === 'student' ? TABS.student : TABS.staff)];
  if (role === 'admin') tabs.push(['users', '⚙', 'Users']);

  const bar = $('#tabbar');
  bar.innerHTML = '';
  tabs.forEach(([id, icon, label]) => {
    const button = el(`<button data-tab="${id}"><span class="ic">${icon}</span>${esc(label)}</button>`);
    button.onclick = () => go(id);
    bar.appendChild(button);
  });
}

function go(tab) {
  document.querySelectorAll('#tabbar button')
    .forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  (VIEWS[tab] || viewHome)();
}

// ── login ────────────────────────────────────────────────────────────────────
$('#loginForm').addEventListener('submit', async event => {
  event.preventDefault();
  message('#loginMsg', 'Signing in…');
  try {
    const res = await post('/auth/login', {
      email: $('#email').value.trim(),
      password: $('#password').value,
    });
    store.token = res.token;
    store.user = res.user;
    message('#loginMsg', '');
    enterApp();
  } catch (err) {
    message('#loginMsg', err.message, 'err');
  }
});

$('#logout').onclick = signOut;

// ── boot ─────────────────────────────────────────────────────────────────────
initCamera();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

if (store.token && store.user) {
  get('/auth/me')
    .then(user => { store.user = user; enterApp(); })
    .catch(() => signOut());
}
