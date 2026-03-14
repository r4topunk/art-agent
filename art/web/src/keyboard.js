import { switchTab } from './tabs.js';
import { toggleMuralPause } from './mural/controls.js';

export function initKeyboard() {
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'm' || e.key === 'M') switchTab('mural');
    if (e.key === 'n' || e.key === 'N') switchTab('main');
    if (e.key === ' ' && document.getElementById('page-mural').classList.contains('active')) {
      e.preventDefault();
      toggleMuralPause();
    }
    if (e.key === 'c' || e.key === 'C') {
      document.body.classList.toggle('no-crt');
    }
  });
}
