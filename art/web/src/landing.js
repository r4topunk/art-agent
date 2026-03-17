import { state } from './state.js';
import { switchTab } from './tabs.js';
import { initAudio, toggleSound } from './mural/sonify.js';
import { syncSoundToolbarBtn } from './dom/mural-toolbar.js';
import { saveSettings } from './persist.js';

const TEXTS = {
  en: {
    subtitle: 'Neural pixel art · Cellular automata · Generative sound',
    desc: 'A living system where a small neural network learns to generate 16×16 pixel art tiles. Selected pieces become seeds for four modes — Game of Life, Quad Fusion, Reaction-Diffusion, and Morphogenesis — that evolve on screen while an audio engine synthesizes sound from cell activity in real time. Switch between modes to discover different visual and sonic worlds.',
    'shortcuts-title': 'Keyboard Shortcuts',
    'k-mural': 'Mural view',
    'k-neural': 'Neural view',
    'k-pause': 'Pause / Play',
    'k-full': 'Fullscreen',
    'k-sound': 'Toggle sound',
    'k-crt': 'CRT effect',
    'k-zoom': 'Zoom in / out',
    'k-vol': 'Volume down / up',
    start: 'Start',
    note: 'Sound will start at 50% volume',
  },
  pt: {
    subtitle: 'Pixel art neural · Autômatos celulares · Som generativo',
    desc: 'Um sistema vivo onde uma pequena rede neural aprende a gerar tiles de pixel art 16×16. As peças selecionadas viram sementes para quatro modos — Game of Life, Quad Fusion, Reação-Difusão e Morfogênese — que evoluem na tela enquanto um motor de áudio sintetiza som a partir da atividade das células em tempo real. Alterne entre os modos para descobrir diferentes mundos visuais e sonoros.',
    'shortcuts-title': 'Atalhos de Teclado',
    'k-mural': 'Visualização mural',
    'k-neural': 'Visualização neural',
    'k-pause': 'Pausar / Continuar',
    'k-full': 'Tela cheia',
    'k-sound': 'Ligar / desligar som',
    'k-crt': 'Efeito CRT',
    'k-zoom': 'Zoom + / −',
    'k-vol': 'Volume − / +',
    start: 'Iniciar',
    note: 'O som iniciará em 50% do volume',
  },
};

let currentLang = 'en';

function applyLang(lang) {
  currentLang = lang;
  const texts = TEXTS[lang];
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (texts[key]) el.textContent = texts[key];
  });
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });
}

function startExhibition() {
  // Hide landing
  document.getElementById('landing').classList.add('hidden');

  // Switch to mural
  switchTab('mural');

  // Enable sound at 50%
  state.volumePercent = 50;
  saveSettings({ volumePercent: 50 });

  // Init audio and enable sound
  initAudio();
  const a = state.audio;
  if (!a.enabled) {
    toggleSound();
  }
  // Set volume to 50%
  if (a.masterGain) {
    a.masterGain.gain.value = 0.5;
  }
  syncSoundToolbarBtn();

  // Update volume UI
  const volVal = document.getElementById('vol-val');
  if (volVal) volVal.textContent = '50%';
  const volSlider = document.getElementById('range-volume');
  if (volSlider) volSlider.value = 50;
}

export function initLanding() {
  const landing = document.getElementById('landing');
  if (!landing) return;

  // Language buttons
  landing.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => applyLang(btn.dataset.lang));
  });

  // Start button
  document.getElementById('landing-start').addEventListener('click', startExhibition);

  // Also allow Enter key to start
  document.addEventListener('keydown', function onEnter(e) {
    if (landing.classList.contains('hidden')) return;
    if (e.key === 'Enter') {
      e.preventDefault();
      startExhibition();
      document.removeEventListener('keydown', onEnter);
    }
  });

  // Detect browser language for default
  const browserLang = navigator.language || '';
  if (browserLang.startsWith('pt')) {
    applyLang('pt');
  }
}
