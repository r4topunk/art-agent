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

// Init keyboard shortcuts
initKeyboard();

// Connect WebSocket
connect(handle);
