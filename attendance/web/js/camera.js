/* Camera modal: single capture or a timed burst, shared by enrolment and check-in. */
import { $ } from './ui.js';

let stream = null;
let facing = 'user';
let config = null;

async function startStream() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing }, audio: false });
    $('#cam').srcObject = stream;
  } catch (err) {
    const secure = location.protocol === 'https:' || location.hostname === 'localhost';
    $('#camMsg').className = 'msg err';
    $('#camMsg').textContent = secure
      ? `Camera error: ${err.message}`
      : 'Camera requires HTTPS (or localhost). Open this site over https://';
  }
}

function snap() {
  const cam = $('#cam'), cv = $('#cv');
  cv.width = cam.videoWidth || 640;
  cv.height = cam.videoHeight || 480;
  cv.getContext('2d').drawImage(cam, 0, 0, cv.width, cv.height);
  return cv.toDataURL('image/jpeg', 0.9);
}

export function openCamera(cfg) {
  config = cfg;
  $('#camTitle').textContent = cfg.title;
  $('#camAction').textContent = cfg.action || 'Capture';
  $('#camMsg').className = 'msg';
  $('#camMsg').textContent = '';
  $('#camModal').classList.add('show');
  startStream();
}

export function closeCamera() {
  $('#camModal').classList.remove('show');
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}

async function handleCapture() {
  const button = $('#camAction'), msg = $('#camMsg');
  button.disabled = true;
  try {
    let result;
    if (config.burst) {
      const shots = [];
      for (let i = 0; i < config.burst; i++) {
        msg.className = 'msg';
        msg.textContent = `Capturing ${i + 1} of ${config.burst}…`;
        shots.push(snap());
        await new Promise(r => setTimeout(r, 600));
      }
      result = await config.onBurst(shots);
    } else {
      msg.className = 'msg';
      msg.textContent = 'Processing…';
      result = await config.onCapture(snap());
    }
    msg.className = 'msg ok';
    msg.textContent = result?.message || 'Done';
    if (result?.close !== false) closeCamera();
  } catch (err) {
    msg.className = 'msg err';
    msg.textContent = err.message;
  } finally {
    button.disabled = false;
  }
}

export function initCamera() {
  $('#camClose').onclick = closeCamera;
  $('#camFlip').onclick = () => { facing = facing === 'user' ? 'environment' : 'user'; startStream(); };
  $('#camAction').onclick = handleCapture;
}
