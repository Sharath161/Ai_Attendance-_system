/* Student views: attendance summary, check-in, face enrolment. */
import { get, post, store } from '../api.js';
import { $, barClass, esc, fmtDate, message, render, toast } from '../ui.js';
import { openCamera } from '../camera.js';

export async function viewHome() {
  render('<h2>My attendance</h2><div class="empty">Loading…</div>');
  try {
    const data = await get('/me/attendance');

    const courses = data.courses.map(c => `
      <li><div style="flex:1">
        <div class="t">${esc(c.code)} — ${esc(c.name)}</div>
        <div class="s">${c.attended} of ${c.total} sessions</div>
        <div class="bar ${barClass(c.percentage)}"><i style="width:${c.percentage}%"></i></div>
      </div><div class="right"><strong>${c.percentage}%</strong></div></li>`).join('')
      || '<div class="empty">You are not enrolled on any course yet.</div>';

    const history = data.history.slice(0, 25).map(h => `
      <li><div>
        <div class="t">${esc(h.course_code)}</div>
        <div class="s">${fmtDate(h.marked_at)}</div>
      </div><span class="pill ${h.method === 'face' ? 'ok' : 'warn'}">${esc(h.method)}</span></li>`).join('')
      || '<div class="empty">No check-ins yet.</div>';

    render(`<h2>My attendance</h2><ul class="list">${courses}</ul>
            <h3 style="margin-top:18px">Recent check-ins</h3><ul class="list">${history}</ul>`);
  } catch (err) {
    render(`<div class="empty">${esc(err.message)}</div>`);
  }
}

export function viewCheckin() {
  render(`<h2>Check in</h2>
    <div class="card">
      <p class="s" style="color:var(--mut);margin:0 0 4px">
        Your teacher must have an open session. Look straight at the camera.</p>
      <button class="primary" id="btnCheckin">Open camera &amp; check in</button>
      <div id="ciMsg" class="msg"></div>
    </div>`);

  $('#btnCheckin').onclick = () => openCamera({
    title: 'Check in',
    action: 'Capture & check in',
    onCapture: async image => {
      const res = await post('/checkin', { image });
      const ok = res.status === 'marked';
      toast(res.message || res.status, ok ? 'ok' : 'bad');
      message('#ciMsg', res.message + (res.confidence ? ` (confidence ${res.confidence})` : ''),
              ok ? 'ok' : 'err');
      return { close: ok, message: res.message };
    },
  });
}

export function viewFace() {
  const user = store.user;
  render(`<h2>My face</h2>
    <div class="card">
      <div>Status: ${user.face_enrolled
        ? '<span class="pill ok">enrolled</span>'
        : '<span class="pill bad">not enrolled</span>'}</div>
      <p class="s" style="color:var(--mut)">We capture 3 photos. Face the camera in good
         light — one photo is enough, three is more reliable.</p>
      <button class="primary" id="btnEnrol">Capture 3 photos &amp; enrol</button>
      <div id="enMsg" class="msg"></div>
    </div>`);

  $('#btnEnrol').onclick = () => openCamera({
    title: 'Enrol your face',
    action: 'Capture 3 photos',
    burst: 3,
    onBurst: async images => {
      const res = await post('/face/enroll', { images });
      store.user = { ...store.user, face_enrolled: true };
      toast(`Enrolled — ${res.embeddings_added} samples`, 'ok');
      return { close: true };
    },
  });
}
