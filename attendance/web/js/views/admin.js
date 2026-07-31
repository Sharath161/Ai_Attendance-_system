/* Admin view: user management and staff-assisted face enrolment. */
import { authHeader, del, get, post } from '../api.js';
import { $, dataURLtoBlob, esc, message, render, toast } from '../ui.js';
import { openCamera } from '../camera.js';

export async function viewUsers() {
  const users = await get('/users');
  render(`<h2>Users</h2>
    <div class="card">
      <h3>Add user</h3>
      <label>Full name</label><input id="uName" placeholder="Jane Doe">
      <label>Email</label><input id="uEmail" type="email" placeholder="jane@example.com">
      <label>Password</label><input id="uPass" type="password" placeholder="min 6 characters">
      <label>Role</label>
      <select id="uRole"><option>student</option><option>teacher</option><option>admin</option></select>
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
      await post('/users', {
        full_name: $('#uName').value.trim(),
        email: $('#uEmail').value.trim(),
        password: $('#uPass').value,
        role: $('#uRole').value,
        student_number: $('#uNum').value.trim() || null,
      });
      toast('User created', 'ok');
      viewUsers();
    } catch (err) { message('#uMsg', err.message, 'err'); }
  };

  document.querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (!confirm('Delete this user and their face data?')) return;
    await del(`/users/${b.dataset.del}`);
    toast('Deleted');
    viewUsers();
  });

  document.querySelectorAll('[data-face]').forEach(b => b.onclick = () => openCamera({
    title: 'Enrol student face',
    action: 'Capture 3 photos',
    burst: 3,
    onBurst: async images => {
      const form = new FormData();
      images.forEach((dataURL, i) => form.append('images', dataURLtoBlob(dataURL), `shot${i}.jpg`));
      const res = await fetch(`/api/face/enroll/${b.dataset.face}`,
                              { method: 'POST', headers: authHeader(), body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error?.message || 'Enrolment failed');
      toast(`Enrolled — ${data.embeddings_added} samples`, 'ok');
      viewUsers();
      return { close: true };
    },
  }));
}
