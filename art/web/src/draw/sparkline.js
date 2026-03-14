export function updateSparkline(svgId, history, maxLen) {
  const el = document.getElementById(svgId);
  const lines = el.querySelectorAll('polyline');
  const data = history.slice(-maxLen);
  if (data.length < 2) return;
  let max = -Infinity, min = Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] > max) max = data[i];
    if (data[i] < min) min = data[i];
  }
  if (max < 0.001) max = 0.001;
  const range = max - min || 0.001;
  const pts = data.map((v, i) => `${(i / (maxLen - 1)) * 200},${34 - ((v - min) / range) * 32}`).join(' ');
  lines.forEach(l => l.setAttribute('points', pts));
}
