import { state } from './state.js';
import { renderMural } from './mural/render.js';
import { startTimersForMode } from './mural/controls.js';

export function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active');
  document.getElementById(`page-${name}`).classList.add('active');
  if (name === 'mural') {
    if (state.muralMode !== 'gameoflife') renderMural();
    startTimersForMode();
  }
}
