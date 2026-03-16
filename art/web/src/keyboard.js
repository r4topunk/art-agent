import { switchTab } from './tabs.js';
import { toggleMuralPause, toggleFullscreen, muralZoom } from './mural/controls.js';

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
    if ((e.key === 'f' || e.key === 'F') && document.getElementById('page-mural').classList.contains('active')) {
      toggleFullscreen();
    }
    if (document.getElementById('page-mural').classList.contains('active')) {
      if (e.key === '+' || e.key === '=') muralZoom(+1);
      if (e.key === '-') muralZoom(-1);
    }
  });
}
