/* Teacher/admin views: dashboard, kiosk, sessions, courses, reports. */
import { authHeader, del, get, post } from '../api.js';
import { $, barClass, el, esc, fmtDate, fmtTime, message, render, toast } from '../ui.js';
import { openCamera } from '../camera.js';

export async function viewDash() {
  render('<h2>Dashboard</h2><div class="empty">Loading…</div>');
  try {
    const [stats, health] = await Promise.all([get('/stats'), get('/health')]);
    const online = health.face_service?.status === 'ok';
    const cards = [['students', 'Students'], ['courses', 'Courses'],
                   ['open_sessions', 'Open sessions'], ['attendance_records', 'Check-ins'],
                   ['face_enrolled', 'Faces enrolled'], ['sessions', 'Total sessions']]
      .map(([k, label]) => `<div class="stat"><div class="n">${stats[k]}</div>
                            <div class="l">${label}</div></div>`).join('');

    render(`<h2>Dashboard</h2>
      <div class="stat-grid">${cards}</div>
      <div class="card" style="margin-top:14px">
        <h3>Face recognition service</h3>
        <div>${online ? '<span class="pill ok">online</span>'
                      : '<span class="pill bad">unreachable</span>'}
          <span class="s" style="color:var(--mut)"> ${esc(health.face_service_url)}</span></div>
        ${online ? `<div class="s" style="color:var(--mut);margin-top:6px">
            model ${esc(health.face_service.model)} ·
            threshold ${health.face_service.threshold} ·
            ${health.face_service.subjects} subjects</div>` : ''}
      </div>`);
  } catch (err) {
    render(`<div class="empty">${esc(err.message)}</div>`);
  }
}

export async function viewKiosk() {
  const open = (await get('/sessions')).filter(s => s.status === 'open');
  render(`<h2>Kiosk</h2>
    <div class="card">
      <label>Session</label>
      <select id="kSession">
        <option value="">Auto-detect from the student's courses</option>
        ${open.map(s => `<option value="${s.id}">${esc(s.course_code)} — ${esc(s.title || 'Session')}</option>`).join('')}
      </select>
      <button class="primary" id="btnKiosk">Open kiosk camera</button>
      <div id="kMsg" class="msg">${open.length ? '' : 'No open sessions — open one under Sessions.'}</div>
    </div>
    <ul class="list" id="kLog"></ul>`);

  $('#btnKiosk').onclick = () => openCamera({
    title: 'Kiosk check-in', action: 'Capture',
    onCapture: async image => {
      const res = await post('/kiosk/checkin',
                             { image, session_id: $('#kSession').value || null });
      const ok = res.status === 'marked';
      toast(res.message || res.status, ok ? 'ok' : 'bad');
      if (res.student) {
        $('#kLog').prepend(el(`<li><div><div class="t">${esc(res.student.full_name)}</div>
          <div class="s">${new Date().toLocaleTimeString()}</div></div>
          <span class="pill ${ok ? 'ok' : 'bad'}">${esc(res.status)}</span></li>`));
      }
      return { close: false, message: res.message };
    },
  });
}

export async function viewSessions() {
  const [courses, sessions] = await Promise.all([get('/courses'), get('/sessions')]);
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
        <div class="s">${fmtDate(s.opened_at)} · ${s.present} present</div></div>
        <div class="right">
          <span class="pill ${s.status === 'open' ? 'open' : ''}">${esc(s.status)}</span>
          ${s.status === 'open' ? `<button class="ghost small" data-close="${s.id}" style="margin-left:6px">Close</button>` : ''}
          <button class="ghost small" data-view="${s.id}" style="margin-left:6px">View</button>
        </div></li>`).join('') || '<div class="empty">No sessions yet.</div>'}</ul>`);

  $('#btnOpen').onclick = async () => {
    try {
      await post('/sessions', { course_id: $('#sCourse').value,
                                title: $('#sTitle').value.trim() || null });
      toast('Session opened', 'ok');
      viewSessions();
    } catch (err) { message('#sMsg', err.message, 'err'); }
  };
  document.querySelectorAll('[data-close]').forEach(b => b.onclick = async () => {
    await post(`/sessions/${b.dataset.close}/close`);
    toast('Session closed');
    viewSessions();
  });
  document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => sessionDetail(b.dataset.view));
}

async function sessionDetail(id) {
  const d = await get(`/sessions/${id}/attendance`);
  render(`<h2>${esc(d.session.course_code || 'Session')}</h2>
    <button class="ghost small" id="back">← Sessions</button>
    <h3 style="margin-top:14px">Present (${d.present.length})</h3>
    <ul class="list">${d.present.map(p => `<li><div><div class="t">${esc(p.full_name)}</div>
      <div class="s">${fmtTime(p.marked_at)} · ${esc(p.method)}${p.confidence ? ' · ' + p.confidence.toFixed(3) : ''}</div></div>
      <button class="ghost small" data-un="${p.student_id}">Undo</button></li>`).join('')
      || '<div class="empty">Nobody yet.</div>'}</ul>
    <h3 style="margin-top:16px">Absent (${d.absent.length})</h3>
    <ul class="list">${d.absent.map(a => `<li><div class="t">${esc(a.full_name)}</div>
      <button class="ghost small" data-mark="${a.id}">Mark present</button></li>`).join('')
      || '<div class="empty">Everyone is present.</div>'}</ul>`);

  $('#back').onclick = viewSessions;
  document.querySelectorAll('[data-mark]').forEach(b => b.onclick = async () => {
    await post('/attendance/manual', { session_id: id, student_id: b.dataset.mark });
    toast('Marked present', 'ok');
    sessionDetail(id);
  });
  document.querySelectorAll('[data-un]').forEach(b => b.onclick = async () => {
    await del(`/attendance/${id}/${b.dataset.un}`);
    toast('Removed');
    sessionDetail(id);
  });
}

export async function viewCourses() {
  const courses = await get('/courses');
  render(`<h2>Courses</h2>
    <div class="card">
      <h3>New course</h3>
      <label>Code</label><input id="cCode" placeholder="CS101">
      <label>Name</label><input id="cName" placeholder="Introduction to Computing">
      <button class="primary" id="btnCourse">Create course</button>
      <div id="cMsg" class="msg"></div>
    </div>
    <ul class="list">${courses.map(c => `<li>
      <div><div class="t">${esc(c.code)}</div>
      <div class="s">${esc(c.name)} · ${c.student_count ?? 0} students</div></div>
      <button class="ghost small" data-roster="${c.id}">Roster</button></li>`).join('')
      || '<div class="empty">No courses yet.</div>'}</ul>`);

  $('#btnCourse').onclick = async () => {
    try {
      await post('/courses', { code: $('#cCode').value.trim(), name: $('#cName').value.trim() });
      toast('Course created', 'ok');
      viewCourses();
    } catch (err) { message('#cMsg', err.message, 'err'); }
  };
  document.querySelectorAll('[data-roster]').forEach(b => b.onclick = () => roster(b.dataset.roster));
}

async function roster(courseId) {
  const [enrolled, all] = await Promise.all([
    get(`/courses/${courseId}/students`), get('/users?role=student')]);
  const ids = new Set(enrolled.map(s => s.id));

  render(`<h2>Roster</h2><button class="ghost small" id="back">← Courses</button>
    <h3 style="margin-top:14px">Enrolled (${enrolled.length})</h3>
    <ul class="list">${enrolled.map(s => `<li><div><div class="t">${esc(s.full_name)}</div>
      <div class="s">${esc(s.student_number || s.email)} ${s.face_enrolled ? '· face ✓' : '· no face'}</div></div>
      <button class="ghost small" data-rm="${s.id}">Remove</button></li>`).join('')
      || '<div class="empty">Nobody enrolled.</div>'}</ul>
    <h3 style="margin-top:16px">Add students</h3>
    <ul class="list">${all.filter(s => !ids.has(s.id)).map(s => `<li><div class="t">${esc(s.full_name)}</div>
      <button class="ghost small" data-add="${s.id}">Add</button></li>`).join('')
      || '<div class="empty">All students enrolled.</div>'}</ul>`);

  $('#back').onclick = viewCourses;
  document.querySelectorAll('[data-add]').forEach(b => b.onclick = async () => {
    await post(`/courses/${courseId}/enrol?student_id=${b.dataset.add}`);
    roster(courseId);
  });
  document.querySelectorAll('[data-rm]').forEach(b => b.onclick = async () => {
    await del(`/courses/${courseId}/enrol/${b.dataset.rm}`);
    roster(courseId);
  });
}

export async function viewReports() {
  const courses = await get('/courses');
  render(`<h2>Reports</h2>
    <div class="card">
      <label>Course</label>
      <select id="rCourse">${courses.map(c => `<option value="${c.id}">${esc(c.code)} — ${esc(c.name)}</option>`).join('')}</select>
      <div class="row"><button id="btnRun">Run report</button>
        <button class="ghost" id="btnCsv">Export CSV</button></div>
    </div>
    <div id="rOut"></div>`);

  const run = async () => {
    const d = await get(`/reports/course/${$('#rCourse').value}`);
    $('#rOut').innerHTML = `<ul class="list">${d.students.map(s => `
      <li><div style="flex:1"><div class="t">${esc(s.full_name)}</div>
        <div class="s">${s.attended}/${s.total} sessions</div>
        <div class="bar ${barClass(s.percentage)}"><i style="width:${s.percentage}%"></i></div></div>
        <div class="right"><strong>${s.percentage}%</strong></div></li>`).join('')
      || '<div class="empty">No students enrolled.</div>'}</ul>`;
  };

  $('#btnRun').onclick = run;
  $('#btnCsv').onclick = async () => {
    const res = await fetch(`/api/reports/course/${$('#rCourse').value}/export.csv`,
                            { headers: authHeader() });
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'attendance.csv';
    a.click();
  };
  run();
}
