export function connect(handler) {
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
    // If we never connected, load static data instead of retrying forever
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
}

let staticLoaded = false;

async function loadStaticPieces(handler) {
  if (staticLoaded) return;
  staticLoaded = true;
  document.getElementById('conn-text').textContent = 'Offline';
  try {
    const resp = await fetch('/data/pieces.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const { pieces } = await resp.json();
    document.getElementById('conn-text').textContent = `Offline (${pieces.length} pieces)`;
    handler('gen_pieces', { pieces });
  } catch (err) {
    console.warn('Failed to load static pieces:', err);
    document.getElementById('conn-text').textContent = 'No data';
  }
}
