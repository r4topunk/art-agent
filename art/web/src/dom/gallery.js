export function buildGallery() {
  const galleryDiv = document.getElementById('gallery');
  const galleryCanvases = [];
  for (let i = 0; i < 36; i++) {
    const c = document.createElement('canvas');
    c.width = 16; c.height = 16;
    galleryDiv.appendChild(c);
    galleryCanvases.push(c);
  }
  return galleryCanvases;
}
