const phaseColors = {
  training:   '#29adff',
  generating: '#6c5ce7',
  scoring:    '#39ff14',
  finetuning: '#ffb300',
};

export function setPhase(p) {
  document.getElementById('phase-text').textContent = p;
  document.getElementById('stat-phase').textContent = p;
  document.getElementById('mural-phase').textContent = p;
  const dot = document.getElementById('phase-dot');
  const c = phaseColors[p] || '#39ff14';
  dot.style.background = c;
  dot.style.boxShadow = `0 0 8px ${c}`;
  document.querySelectorAll('.particle-track').forEach(d => d.style.background = c);
}
