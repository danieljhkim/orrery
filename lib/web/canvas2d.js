// DPR-aware 2D canvas setup, an optional pan/zoom camera, and a HUD overlay.
//
//   const { canvas, ctx, size } = createCanvas2D({ mount: wrap, height: 560 });
//   const view = attachPanZoom(canvas);       // drag to pan, wheel to zoom
//   view.apply(ctx);                          // world -> screen transform
//   const hud = createHud(wrap);  hud.set('t = 0.00');
//
// The canvas fills its mount's width (CSS) and re-resolves on window resize;
// drawing code should read size() each frame rather than caching pixels.
export function createCanvas2D({ mount, height = 560 } = {}) {
  const canvas = document.createElement('canvas');
  canvas.style.height = `${height}px`;
  mount.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const listeners = [];

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    listeners.forEach((cb) => cb(w, h));
  }
  window.addEventListener('resize', resize);
  queueMicrotask(resize);

  return {
    canvas,
    ctx,
    size: () => ({ w: canvas.clientWidth, h: canvas.clientHeight }),
    onResize: (cb) => listeners.push(cb),
    clear(color) {
      const { w, h } = this.size();
      ctx.save();
      ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
      if (color) { ctx.fillStyle = color; ctx.fillRect(0, 0, w, h); }
      else ctx.clearRect(0, 0, w, h);
      ctx.restore();
    },
  };
}

// Pan/zoom camera: world point (0,0) starts centered, `scale` is px per world
// unit. Mutates nothing outside itself; call view.apply(ctx) before drawing in
// world coordinates, view.screenToWorld(x, y) for picking.
export function attachPanZoom(canvas, { scale = 100, minScale = 1e-6, maxScale = 1e6, onChange } = {}) {
  const view = { x: 0, y: 0, scale }; // x,y = world coords at canvas center
  let dragging = false, lastX = 0, lastY = 0;

  canvas.style.cursor = 'grab';
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    view.x -= (e.clientX - lastX) / view.scale;
    view.y -= (e.clientY - lastY) / view.scale;
    lastX = e.clientX; lastY = e.clientY;
    onChange?.(view);
  });
  canvas.addEventListener('pointerup', () => { dragging = false; canvas.style.cursor = 'grab'; });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = Math.exp(-e.deltaY * 0.0015);
    view.scale = Math.min(maxScale, Math.max(minScale, view.scale * factor));
    onChange?.(view);
  }, { passive: false });

  view.apply = (ctx) => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.translate(w / 2, h / 2);
    ctx.scale(view.scale, view.scale);
    ctx.translate(-view.x, -view.y);
  };
  view.screenToWorld = (sx, sy) => {
    const r = canvas.getBoundingClientRect();
    return [
      view.x + (sx - r.left - r.width / 2) / view.scale,
      view.y + (sy - r.top - r.height / 2) / view.scale,
    ];
  };
  return view;
}

export function createHud(mount) {
  const el = document.createElement('div');
  el.className = 'orrery-hud';
  mount.appendChild(el);
  return { el, set: (text) => { el.textContent = text; } };
}
