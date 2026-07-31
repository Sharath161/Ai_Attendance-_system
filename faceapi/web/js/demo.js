/* Interactive camera demo: enrol + identify from the browser. */
const $ = s => document.querySelector(s);
const cam = $("#cam"), cv = $("#cv"), out = $("#out");
let stream = null, facing = "user";

async function startCam() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:facing }, audio:false });
    cam.srcObject = stream;
  } catch (e) { out.textContent = "Camera error: " + e.message; }
}
function snap() {
  const w = cam.videoWidth, h = cam.videoHeight;
  cv.width = w; cv.height = h;
  cv.getContext("2d").drawImage(cam, 0, 0, w, h);
  return cv.toDataURL("image/jpeg", 0.9);   // data: URI (API strips the prefix)
}
async function post(path, body) {
  const r = await fetch(path, { method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(body) });
  return { ok:r.ok, status:r.status, data: await r.json() };
}

$("#start").onclick = startCam;
$("#flip").onclick = () => { facing = facing === "user" ? "environment" : "user"; startCam(); };

$("#enroll").onclick = async () => {
  const sid = $("#sid").value.trim();
  if (!sid) { out.textContent = "Enter a Subject ID first."; return; }
  if (!stream) { out.textContent = "Start the camera first."; return; }
  out.textContent = "Capturing 3 shots…";
  const images = [];
  for (let i = 0; i < 3; i++) { images.push(snap()); await new Promise(r => setTimeout(r, 550)); }
  const r = await post("/v1/enroll", { subject_id:sid, name:$("#sname").value.trim() || null, images });
  out.innerHTML = r.ok
    ? `<span class="pill ok">ENROLLED</span> ${sid} — ${r.data.embeddings_added} face(s) added, ${r.data.faces_rejected} rejected.`
    : `<span class="pill bad">ERROR ${r.status}</span> ${r.data.error?.message || JSON.stringify(r.data)}`;
};

$("#identify").onclick = async () => {
  if (!stream) { out.textContent = "Start the camera first."; return; }
  out.textContent = "Identifying…";
  const r = await post("/v1/identify", { image: snap() });
  const d = r.data;
  if (!r.ok) { out.innerHTML = `<span class="pill bad">ERROR ${r.status}</span> ${d.error?.message||""}`; return; }
  if (d.status === "matched") {
    const m = d.match;
    out.innerHTML = `<span class="pill ok">MATCH</span> <b>${m.name || m.subject_id}</b> `
      + `<span class="mut">(${m.subject_id}) · score ${m.score.toFixed(3)} · thr ${d.threshold}</span>`;
  } else {
    const top = (d.candidates||[]).slice(0,3).map(c => `${c.subject_id}: ${c.score.toFixed(3)}`).join("  ·  ");
    out.innerHTML = `<span class="pill bad">${d.status.toUpperCase()}</span> `
      + (top ? `<div class="mut" style="margin-top:6px">closest — ${top}</div>` : "");
  }
};
