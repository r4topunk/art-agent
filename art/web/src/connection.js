export function connect(handler) {
  const ws = new WebSocket(`ws://${location.hostname || 'localhost'}:8765`);
  ws.onopen = () => {
    document.getElementById('conn').className = 'connected';
    document.getElementById('conn-text').textContent = 'Linked';
  };
  ws.onclose = () => {
    document.getElementById('conn').className = 'disconnected';
    document.getElementById('conn-text').textContent = 'Disconnected';
    setTimeout(() => connect(handler), 2500);
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
