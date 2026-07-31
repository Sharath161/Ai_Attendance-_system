/* Attendance PWA — role-aware single-page client for the attendance API. */
const $ = s => document.querySelector(s);
const el = (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content.firstChild; };
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

let TOKEN = localStorage.getItem('att_token') || '';
let USER = JSON.parse(localStorage.getItem('att_user') || 'null');
let TAB = 'home';

/* ── api ─────────────────────────────────────────────────────────────────── */
async function api(path, { method = 'GET', body, raw } = {}) {
  const headers = {};
  if (TOKEN) headers['Authorization'] = `Bearer ${TOKEN}`;
  if (body && !raw) headers['Content-Type'] = 'application/json';
  const r = await fetch(`/api${path}`, { method, headers, body: raw ? body : (body ? JSON.stringify(body) : undefined) });
  if (r.status === 401) { signOut(); throw new Error('Session expired'); }
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(data?.error?.message || data?.detail || `Error ${r.status}`);
  return data;
}

function toast(msg, kind = '') {
  const t = $('#toast'); t.textContent = msg; t.className = `toast show ${kind}`;
  clearTimeout(t._h); t._h = setTimeout(() => t.className = 'toast', 3200);
}

/* ── auth ────────────────────────────────────────────────────────────────── */
$('#loginForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = $('#loginMsg'); msg.className = 'msg'; msg.textContent = 'Signing in…';
  try {
    const r = await api('/auth/login', { method: 'POST', body: { email: $('#email').value.trim(), password: $('#password').value } });
    TOKEN = r.token; USER = r.user;
    localStorage.setItem('att_token', TOKEN); localStorage.setItem('att_user', JSON.stringify(USER));
    msg.textContent = ''; enterApp();
  } catch (err) { msg.className = 'msg err'; msg.textContent = err.message; }
});

$('#logout').onclick = signOut;
function signOut() {
  TOKEN = ''; USER = null; localStorage.removeItem('att_token'); localStorage.removeItem('att_user');
  $('#view-main').classList.remove('active'); $('#view-login').classList.add('active');
}

function enterApp() {
  $('#view-login').classList.remove('active'); $('#view-main').classList.add('active');
  $('#userName').textContent = USER.full_name;
  $('#userRole').textContent = USER.role;
  buildTabs(); go(USER.role === 'student' ? 'home' : 'dash');
}

/* ── tabs ────────────────────────────────────────────────────────────────── */
function buildTabs() {
  const tabs = USER.role === 'student'
    ? [['home','◎','My attendance'],['checkin','⊙','Check in'],['face','☺','My face']]
    : [['dash','▦','Dashboard'],['kiosk','⊙','Kiosk'],['sessions','▤','Sessions'],['courses','◈','Courses'],['reports','◫','Reports']];
  if (USER.role === 'admin') tabs.push(['users','⚙','Users']);
  const bar = $('#tabbar'); bar.innerHTML = '';
  tabs.forEach(([id, ic, label]) => {
    const b = el(`<button data-tab="${id}"><span class="ic">${ic}</span>${esc(label)}</button>`);
    b.onclick = () => go(id); bar.appendChild(b);
  });
}

function go(tab) {
  TAB = tab;
  document.querySelectorAll('#tabbar button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  ({ home: viewHome, checkin: viewCheckin, face: viewFace, dash: viewDash,
     kiosk: viewKiosk, sessions: viewSessions, courses: viewCourses,
     reports: viewReports, users: viewUsers }[tab] || viewHome)();
}
const render = h => { $('#content').innerHTML = h; };

/* ── student: my attendance ──────────────────────────────────────────────── */
async function viewHome() {
  render('<h2>My attendance</h2><div class="empty">Loading…</div>');
  try {
    const d = await api('/me/attendance');
    const courses = d.courses.map(c => {
      const cls = c.percentage >= 75 ? '' : (c.percentage >= 50 ? 'mid' : 'low');
      return `<li><div style="flex:1">
        <div class="t">${esc(c.code)} — ${esc(c.name)}</div>
        <div class="s">${c.attended} of ${c.total} sessions</div>
        <div class="bar ${cls}"><i style="width:${c.percentage}%"></i></div>
      </div><div class="right"><strong>${c.percentage}%</strong></div></li>`;
    }).join('') || '<div class="empty">You are not enrolled on any course yet.</div>';

    const hist = d.history.slice(0, 25).map(h => `<li><div>
        <div class="t">${esc(h.course_code)}</div>
        <div class="s">${new Date(h.marked_at).toLocaleString()}</div>
      </div><span class="pill ${h.method === 'face' ? 'ok' : 'warn'}">${esc(h.method)}</span></li>`).join('')
      || '<div class="empty">No check-ins yet.</div>';

    render(`<h2>My attendance</h2><ul class="list">${courses}</ul>
            <h3 style="margin-top:18px">Recent check-ins</h3><ul class="list">${hist}</ul>`);
  } catch (e) { render(`<div class="empty">${esc(e.message)}</div>`); }
}

/* ── student: check in ───────────────────────────────────────────────────── */
function viewCheckin() {
  render(`<h2>Check in</h2>
    <div class="card">
      <p class="s" style="color:var(--mut);margin:0 0 4px">
        Your teacher must have an open session. Look straight at the camera.</p>
      <button class="primary" id="btnCheckin">Open camera & check in</button>
      <div id="ciMsg" class="msg"></div>
    </div>`);
  $('#btnCheckin').onclick = () => openCam({
    title: 'Check in', action: 'Capture & check in',
    onCapture: async (b64) => {
      const r = await api('/checkin', { method: 'POST', body: { image: b64 } });
      const good = r.status === 'marked';
      toast(r.message || r.status, good ? 'ok' : 'bad');
      $('#ciMsg').className = `msg ${good ? 'ok' : 'err'}`;
      $('#ciMsg').textContent = r.message + (r.confidence ? ` (confidence ${r.confidence})` : '');
      return { close: good, message: r.message };
    }
  });
}

/* ── student: enrol own face ─────────────────────────────────────────────── */
function viewFace() {
  render(`<h2>My face</h2>
    <div class="card">
      <div>Status: ${USER.face_enrolled
        ? '<span class="pill ok">enrolled</span>'
        : '<span class="pill bad">not enrolled</span>'}</div>
      <p class="s" style="color:var(--mut)">We capture 3 photos. Face the camera in good light —
         one photo is enough, three is more reliable.</p>
      <button class="primary" id="btnEnrol">Capture 3 photos & enrol</button>
      <div id="enMsg" class="msg"></div>
    </div>`);
  $('#btnEnrol').onclick = () => openCam({
    title: 'Enrol your face', action: 'Capture 3 photos', burst: 3,
    onBurst: async (images) => {
      const r = await api('/face/enroll', { method: 'POST', body: { images } });
      USER.face_enrolled = true; localStorage.setItem('att_user', JSON.stringify(USER));
      toast(`Enrolled — ${r.embeddings_added} samples`, 'ok');
      return { close: true };
    }
  });
}

/* ── staff: dashboard ────────────────────────────────────────────────────── */
async function viewDash() {
  render('<h2>Dashboard</h2><div class="empty">Loading…</div>');
  try {
    const [s, h] = await Promise.all([api('/stats'), api('/health')]);
    const face = h.face_service?.status === 'ok';
    render(`<h2>Dashboard</h2>
      <div class="stat-grid">
        ${[['students','Students'],['courses','Courses'],['open_sessions','Open sessions'],
           ['attendance_records','Check-ins'],['face_enrolled','Faces enrolled'],['sessions','Total sessions']]
          .map(([k,l]) => `<div class="stat"><div class="n">${s[k]}</div><div class="l">${l}</div></div>`).join('')}
      </div>
      <div class="card" style="margin-top:14px">
        <h3>Face recognition service</h3>
        <div>${face ? '<span class="pill ok">online</span>' : '<span class="pill bad">unreachable</span>'}
          <span class="s" style="color:var(--mut)"> ${esc(h.face_service_url)}</span></div>
        ${face ? `<div class="s" style="color:var(--mut);margin-top:6px">
            model ${esc(h.face_service.model)} · threshold ${h.face_service.threshold} ·
            ${h.face_service.subjects} subjects</div>` : ''}
      </div>`);
  } catch (e) { render(`<div class="empty">${esc(e.message)}</div>`); }
}

/* ── staff: kiosk ────────────────────────────────────────────────────────── */
async function viewKiosk() {
  const sessions = (await api('/sessions')).filter(s => s.status === 'open');
  render(`<h2>Kiosk</h2>
    <div class="card">
      <label>Session</label>
      <select id="kSession">
        <option value="">Auto-detect from student's courses</option>
        ${sessions.map(s => `<option value="${s.id}">${esc(s.course_code)} — ${esc(s.title || 'Session')}</option>`).join('')}
      </select>
      <button class="primary" id="btnKiosk">Open kiosk camera</button>
      <div id="kMsg" class="msg">${sessions.length ? '' : 'No open sessions — open one under Sessions.'}</div>
    </div>
    <ul class="list" id="kLog"></ul>`);
  $('#btnKiosk').onclick = () => openCam({
    title: 'Kiosk check-in', action: 'Capture', keepOpen: true,
    onCapture: async (b64) => {
      const sid = $('#kSession').value || null;
      const r = await api('/kiosk/checkin', { method: 'POST', body: { image: b64, session_id: sid } });
      const good = r.status === 'marked';
      toast(r.message || r.status, good ? 'ok' : 'bad');
      if (r.student) $('#kLog').prepend(el(`<li><div><div class="t">${esc(r.student.full_name)}</div>
        <div class="s">${new Date().toLocaleTimeString()}</div></div>
        <span class="pill ${good ? 'ok' : 'bad'}">${esc(r.status)}</span></li>`));
      return { close: false, message: r.message };
    }
  });
}

/* ── staff: sessions ─────────────────────────────────────────────────────── */
async function viewSessions() {
  const [courses, sessions] = await Promise.all([api('/courses'), api('/sessions')]);
  render(`<h2>Sessions</h2>
    <div class="card">
      <h3>Open a session</h3>
      <label>Course</label>
      <select id="sCourse">${courses.map(c => `<option value="${c.id}">${esc(c.code)} — ${esc(c.name)}</option>`).join('')}</select>
      <label>Title (optional)</label><input id="sTitle" placeholder="e.g. Week 5 lecture">
      <button class="primary" id="btnOpen">Open session</button>
      <div id="sMsg" class="msg"></div>
    </div>
    <ul class="list">${sessions.map(s => `<li>
        <div><div class="t">${esc(s.course_code)} — ${esc(s.title || 'Session')}</div>
        <div class="s">${new Date(s.opened_at).toLocaleString()} · ${s.present} present</div></div>
        <div class="right">
          <span class="pill ${s.status === 'open' ? 'open' : ''}">${esc(s.status)}</span>
          ${s.status === 'open' ? `<button class="ghost small" data-close="${s.id}" style="margin-left:6px">Close</button>` : ''}
          <button class="ghost small" data-view="${s.id}" style="margin-left:6px">View</button>
        </div></li>`).join('') || '<div class="empty">No sessions yet.</div>'}</ul>`);

  $('#btnOpen').onclick = async () => {
    try {
      await api('/sessions', { method: 'POST', body: { course_id: $('#sCourse').value, title: $('#sTitle').value.trim() || null } });
      toast('Session opened', 'ok'); viewSessions();
    } catch (e) { $('#sMsg').className = 'msg err'; $('#sMsg').textContent = e.message; }
  };
  document.querySelectorAll('[data-close]').forEach(b => b.onclick = async () => {
    await api(`/sessions/${b.dataset.close}/close`, { method: 'POST' }); toast('Session closed'); viewSessions();
  });
  document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => sessionDetail(b.dataset.view));
}

async function sessionDetail(id) {
  const d = await api(`/sessions/${id}/attendance`);
  render(`<h2>${esc(d.session.course_code || 'Session')}</h2>
    <button class="ghost small" id="back">← Sessions</button>
    <h3 style="margin-top:14px">Present (${d.present.length})</h3>
    <ul class="list">${d.present.map(p => `<li><div><div class="t">${esc(p.full_name)}</div>
      <div class="s">${new Date(p.marked_at).toLocaleTimeString()} · ${esc(p.method)}${p.confidence ? ' · ' + p.confidence.toFixed(3) : ''}</div></div>
      <button class="ghost small" data-un="${p.student_id}">Undo</button></li>`).join('') || '<div class="empty">Nobody yet.</div>'}</ul>
    <h3 style="margin-top:16px">Absent (${d.absent.length})</h3>
    <ul class="list">${d.absent.map(a => `<li><div class="t">${esc(a.full_name)}</div>
      <button class="ghost small" data-mark="${a.id}">Mark present</button></li>`).join('') || '<div class="empty">Everyone is present.</div>'}</ul>`);
  $('#back').onclick = viewSessions;
  document.querySelectorAll('[data-mark]').forEach(b => b.onclick = async () => {
    await api('/attendance/manual', { method: 'POST', body: { session_id: id, student_id: b.dataset.mark } });
    toast('Marked present', 'ok'); sessionDetail(id);
  });
  document.querySelectorAll('[data-un]').forEach(b => b.onclick = async () => {
    await api(`/attendance/${id}/${b.dataset.un}`, { method: 'DELETE' }); toast('Removed'); sessionDetail(id);
  });
}

/* ── staff: courses ──────────────────────────────────────────────────────── */
async function viewCourses() {
  const courses = await api('/courses');
  render(`<h2>Courses</h2>
    <div class="card">
      <h3>New course</h3>
      <label>Code</label><input id="cCode" placeholder="CS101">
      <label>Name</label><input id="cName" placeholder="Introduction to Computing">
      <button class="primary" id="btnCourse">Create course</button>
      <div id="cMsg" class="msg"></div>
    </div>
    <ul class="list">${courses.map(c => `<li>
      <div><div class="t">${esc(c.code)}</div><div class="s">${esc(c.name)} · ${c.student_count ?? 0} students</div></div>
      <button class="ghost small" data-roster="${c.id}">Roster</button></li>`).join('') || '<div class="empty">No courses yet.</div>'}</ul>`);
  $('#btnCourse').onclick = async () => {
    try {
      await api('/courses', { method: 'POST', body: { code: $('#cCode').value.trim(), name: $('#cName').value.trim() } });
      toast('Course created', 'ok'); viewCourses();
    } catch (e) { $('#cMsg').className = 'msg err'; $('#cMsg').textContent = e.message; }
  };
  document.querySelectorAll('[data-roster]').forEach(b => b.onclick = () => roster(b.dataset.roster));
}

async function roster(courseId) {
  const [enrolled, all] = await Promise.all([api(`/courses/${courseId}/students`), api('/users?role=student')]);
  const ids = new Set(enrolled.map(s => s.id));
  render(`<h2>Roster</h2><button class="ghost small" id="back">← Courses</button>
    <h3 style="margin-top:14px">Enrolled (${enrolled.length})</h3>
    <ul class="list">${enrolled.map(s => `<li><div><div class="t">${esc(s.full_name)}</div>
      <div class="s">${esc(s.student_number || s.email)} ${s.face_enrolled ? '· face ✓' : '· no face'}</div></div>
      <button class="ghost small" data-rm="${s.id}">Remove</button></li>`).join('') || '<div class="empty">Nobody enrolled.</div>'}</ul>
    <h3 style="margin-top:16px">Add students</h3>
    <ul class="list">${all.filter(s => !ids.has(s.id)).map(s => `<li><div class="t">${esc(s.full_name)}</div>
      <button class="ghost small" data-add="${s.id}">Add</button></li>`).join('') || '<div class="empty">All students enrolled.</div>'}</ul>`);
  $('#back').onclick = viewCourses;
  document.querySelectorAll('[data-add]').forEach(b => b.onclick = async () => {
    await api(`/courses/${courseId}/enrol?student_id=${b.dataset.add}`, { method: 'POST' }); roster(courseId);
  });
  document.querySelectorAll('[data-rm]').forEach(b => b.onclick = async () => {
    await api(`/courses/${courseId}/enrol/${b.dataset.rm}`, { method: 'DELETE' }); roster(courseId);
  });
}

/* ── staff: reports ──────────────────────────────────────────────────────── */
async function viewReports() {
  const courses = await api('/courses');
  render(`<h2>Reports</h2>
    <div class="card">
      <label>Course</label>
      <select id="rCourse">${courses.map(c => `<option value="${c.id}">${esc(c.code)} — ${esc(c.name)}</option>`).join('')}</select>
      <div class="row"><button id="btnRun">Run report</button>
        <button class="ghost" id="btnCsv">Export CSV</button></div>
    </div>
    <div id="rOut"></div>`);
  const run = async () => {
    const d = await api(`/reports/course/${$('#rCourse').value}`);
    $('#rOut').innerHTML = `<ul class="list">${d.students.map(s => {
      const cls = s.percentage >= 75 ? '' : (s.percentage >= 50 ? 'mid' : 'low');
      return `<li><div style="flex:1"><div class="t">${esc(s.full_name)}</div>
        <div class="s">${s.attended}/${s.total} sessions</div>
        <div class="bar ${cls}"><i style="width:${s.percentage}%"></i></div></div>
        <div class="right"><strong>${s.percentage}%</strong></div></li>`;
    }).join('') || '<div class="empty">No students enrolled.</div>'}</ul>`;
  };
  $('#btnRun').onclick = run;
  $('#btnCsv').onclick = async () => {
    const r = await fetch(`/api/reports/course/${$('#rCourse').value}/export.csv`, { headers: { Authorization: `Bearer ${TOKEN}` } });
    const blob = await r.blob(); const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'attendance.csv'; a.click();
  };
  run();
}

/* ── admin: users ────────────────────────────────────────────────────────── */
async function viewUsers() {
  const users = await api('/users');
  render(`<h2>Users</h2>
    <div class="card">
      <h3>Add user</h3>
      <label>Full name</label><input id="uName" placeholder="Jane Doe">
      <label>Email</label><input id="uEmail" type="email" placeholder="jane@example.com">
      <label>Password</label><input id="uPass" type="password" placeholder="min 6 characters">
      <label>Role</label><select id="uRole"><option>student</option><option>teacher</option><option>admin</option></select>
      <label>Student number (optional)</label><input id="uNum" placeholder="STU-001">
      <button class="primary" id="btnUser">Create user</button>
      <div id="uMsg" class="msg"></div>
    </div>
    <ul class="list">${users.map(u => `<li>
      <div><div class="t">${esc(u.full_name)} <span class="badge">${esc(u.role)}</span></div>
      <div class="s">${esc(u.email)} ${u.face_enrolled ? '· face ✓' : ''}</div></div>
      <div class="right">
        ${u.role === 'student' ? `<button class="ghost small" data-face="${u.id}">Enrol face</button>` : ''}
        <button class="ghost small danger" data-del="${u.id}" style="margin-left:6px">Delete</button>
      </div></li>`).join('')}</ul>`);
  $('#btnUser').onclick = async () => {
    try {
      await api('/users', { method: 'POST', body: {
        full_name: $('#uName').value.trim(), email: $('#uEmail').value.trim(),
        password: $('#uPass').value, role: $('#uRole').value,
        student_number: $('#uNum').value.trim() || null } });
      toast('User created', 'ok'); viewUsers();
    } catch (e) { $('#uMsg').className = 'msg err'; $('#uMsg').textContent = e.message; }
  };
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (!confirm('Delete this user and their face data?')) return;
    await api(`/users/${b.dataset.del}`, { method: 'DELETE' }); toast('Deleted'); viewUsers();
  });
  document.querySelectorAll('[data-face]').forEach(b => b.onclick = () => openCam({
    title: 'Enrol student face', action: 'Capture 3 photos', burst: 3,
    onBurst: async (images) => {
      const fd = new FormData();
      images.forEach((b64, i) => fd.append('images', dataURLtoBlob(b64), `s${i}.jpg`));
      const r = await fetch(`/api/face/enroll/${b.dataset.face}`, {
        method: 'POST', headers: { Authorization: `Bearer ${TOKEN}` }, body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d?.error?.message || 'Enrolment failed');
      toast(`Enrolled — ${d.embeddings_added} samples`, 'ok'); viewUsers();
      return { close: true };
    }
  }));
}

function dataURLtoBlob(dataURL) {
  const [head, b64] = dataURL.split(','); const mime = head.match(/:(.*?);/)[1];
  const bin = atob(b64); const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

/* ── camera ──────────────────────────────────────────────────────────────── */
let stream = null, facing = 'user', camCfg = null;
async function startStream() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing }, audio: false });
    $('#cam').srcObject = stream;
  } catch (e) {
    $('#camMsg').className = 'msg err';
    $('#camMsg').textContent = location.protocol === 'https:' || location.hostname === 'localhost'
      ? `Camera error: ${e.message}`
      : 'Camera needs HTTPS (or localhost). Open this site over https://';
  }
}
function snap() {
  const cam = $('#cam'), cv = $('#cv');
  cv.width = cam.videoWidth || 640; cv.height = cam.videoHeight || 480;
  cv.getContext('2d').drawImage(cam, 0, 0, cv.width, cv.height);
  return cv.toDataURL('image/jpeg', 0.9);
}
function openCam(cfg) {
  camCfg = cfg;
  $('#camTitle').textContent = cfg.title;
  $('#camAction').textContent = cfg.action || 'Capture';
  $('#camMsg').className = 'msg'; $('#camMsg').textContent = '';
  $('#camModal').classList.add('show');
  startStream();
}
function closeCam() {
  $('#camModal').classList.remove('show');
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}
$('#camClose').onclick = closeCam;
$('#camFlip').onclick = () => { facing = facing === 'user' ? 'environment' : 'user'; startStream(); };
$('#camAction').onclick = async () => {
  const btn = $('#camAction'), msg = $('#camMsg');
  btn.disabled = true;
  try {
    let res;
    if (camCfg.burst) {
      const shots = [];
      for (let i = 0; i < camCfg.burst; i++) {
        msg.className = 'msg'; msg.textContent = `Capturing ${i + 1} of ${camCfg.burst}…`;
        shots.push(snap()); await new Promise(r => setTimeout(r, 600));
      }
      res = await camCfg.onBurst(shots);
    } else {
      msg.className = 'msg'; msg.textContent = 'Processing…';
      res = await camCfg.onCapture(snap());
    }
    msg.className = 'msg ok'; msg.textContent = res?.message || 'Done';
    if (res?.close !== false) closeCam();
  } catch (e) {
    msg.className = 'msg err'; msg.textContent = e.message;
  } finally { btn.disabled = false; }
};

/* ── boot ────────────────────────────────────────────────────────────────── */
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
if (TOKEN && USER) {
  api('/auth/me').then(u => { USER = u; localStorage.setItem('att_user', JSON.stringify(u)); enterApp(); })
                 .catch(() => signOut());
}
