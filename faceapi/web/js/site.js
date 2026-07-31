/* Product site: code-example tabs + live service status. */
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('[data-pane]').forEach(p =>
    p.classList.toggle('on', p.dataset.pane === b.dataset.tab));
});
// live service status in the footer
fetch('/health').then(r => r.json()).then(h => {
  const d = document.createElement('a');
  d.className = 'status ok'; d.href = '/health';
  d.textContent = `● service online · ${h.subjects} subjects enrolled`;
  document.querySelector('.links').appendChild(d);
}).catch(() => {});
