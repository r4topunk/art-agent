const KEY = 'art.settings';

export function loadSettings() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; }
}

export function saveSettings(patch) {
  const current = loadSettings();
  localStorage.setItem(KEY, JSON.stringify({ ...current, ...patch }));
}
