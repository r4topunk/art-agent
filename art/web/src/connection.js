import { state } from './state.js';
import { syncToolbarUI } from './mural/controls.js';
import { updateTimingLabels } from './dom/mural-toolbar.js';

export function connect(handler) {
  // On HTTPS, ws:// is blocked (mixed content). Skip straight to static data.
  if (location.protocol === 'https:') {
    loadStaticPieces(handler);
    return;
  }

  try {
    const ws = new WebSocket(`ws://${location.hostname || 'localhost'}:8765`);
    let connected = false;
    ws.onopen = () => {
      connected = true;
      document.getElementById('conn').className = 'connected';
      document.getElementById('conn-text').textContent = 'Linked';
    };
    ws.onclose = () => {
      document.getElementById('conn').className = 'disconnected';
      document.getElementById('conn-text').textContent = 'Disconnected';
      if (!connected) {
        loadStaticPieces(handler);
      } else {
        setTimeout(() => connect(handler), 2500);
      }
    };
    ws.onerror = () => ws.close();
    ws.onmessage = e => {
      try {
        const m = JSON.parse(e.data);
        handler(m.event, m.data);
      } catch (err) {
        console.warn('ws parse error', err);
      }
    };
  } catch (_) {
    loadStaticPieces(handler);
  }
}

let staticLoaded = false;

async function loadStaticPieces(handler) {
  if (staticLoaded) return;
  staticLoaded = true;

  // Offline mode: hide tab bar, expand page area
  const tabBar = document.getElementById('tab-bar');
  if (tabBar) tabBar.style.display = 'none';
  const pages = document.getElementById('pages');
  if (pages) pages.style.inset = '0';

  // Default to Game of Life mode
  state.muralMode = 'gameoflife';
  syncToolbarUI();
  updateTimingLabels();

  // Activate mural tab
  const muralTab = document.querySelector('.tab[data-tab="mural"]');
  if (muralTab) muralTab.click();

  try {
    const resp = await fetch('/data/pieces.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const { pieces } = await resp.json();
    handler('gen_pieces', { pieces });
  } catch (err) {
    console.warn('Failed to load static pieces:', err);
  }
}
