// Styles
import './styles/variables.css';
import './styles/crt.css';
import './styles/tabs.css';
import './styles/neural.css';
import './styles/mural.css';

// Modules
import { buildSpine } from './dom/spine.js';
import { buildGallery } from './dom/gallery.js';
import { wireMuralToolbar } from './dom/mural-toolbar.js';
import { switchTab } from './tabs.js';
import { initKeyboard } from './keyboard.js';
import { connect } from './connection.js';
import { initHandlers, handle } from './handlers.js';
import { renderMural } from './mural/render.js';
import * as controls from './mural/controls.js';
import { loadSettings } from './persist.js';

// Build DOM
const layerCanvases = buildSpine();
const galleryCanvases = buildGallery();

// Wire handlers
initHandlers({ layerCanvases, galleryCanvases });

// Wire toolbar
wireMuralToolbar(controls);

// Wire tabs
document.querySelectorAll('.tab[data-tab]').forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// Wire resize
window.addEventListener('resize', () => {
  if (document.getElementById('page-mural').classList.contains('active')) renderMural();
});

// Fullscreen: sync body class + re-render mural to fill new dimensions
document.addEventListener('fullscreenchange', () => {
  document.body.classList.toggle('fullscreen', !!document.fullscreenElement);
  if (document.getElementById('page-mural').classList.contains('active')) renderMural();
});

// Double-click canvas to toggle fullscreen
document.getElementById('mural-canvas-wrap').addEventListener('dblclick', () => {
  if (document.getElementById('page-mural').classList.contains('active')) controls.toggleFullscreen();
});

// Init keyboard shortcuts
initKeyboard();

// Restore persisted settings
const _saved = loadSettings();
controls.syncToolbarUI();
if (_saved.crtDisabled) document.body.classList.add('no-crt');
switchTab(_saved.activeTab ?? 'main');

// Connect WebSocket
connect(handle);
