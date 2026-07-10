// Fixed-timestep simulation loop with play/pause and a speed multiplier.
//
//   const loop = createLoop({
//     step: (dt) => { ...advance physics by dt (sim units)... },
//     render: () => { ...draw current state... },
//     dt: 1 / 120,   // fixed physics step, sim seconds
//     speed: 1,      // sim seconds per real second
//   });
//   loop.start();
//
// Physics always advances in fixed dt increments (accumulator pattern), so
// changing `speed` changes how many steps run per frame — never the step size.
export function createLoop({ step, render, dt = 1 / 120, speed = 1, maxStepsPerFrame = 400 }) {
  let running = false;
  let acc = 0;
  let last = 0;
  let simTime = 0;
  let rafId = 0;

  function frame(now) {
    if (!running) return;
    const elapsed = Math.min((now - last) / 1000, 0.1); // clamp tab-switch spikes
    last = now;
    acc += elapsed * loop.speed;
    let steps = 0;
    while (acc >= dt && steps < maxStepsPerFrame) {
      step(dt);
      simTime += dt;
      acc -= dt;
      steps++;
    }
    if (steps === maxStepsPerFrame) acc = 0; // running behind: drop the backlog
    render();
    rafId = requestAnimationFrame(frame);
  }

  const loop = {
    dt,
    speed,
    get running() { return running; },
    get time() { return simTime; },
    start() {
      if (running) return;
      running = true;
      last = performance.now();
      rafId = requestAnimationFrame(frame);
    },
    pause() {
      running = false;
      cancelAnimationFrame(rafId);
    },
    toggle() { running ? loop.pause() : loop.start(); return running; },
    reset() { acc = 0; simTime = 0; },
    renderOnce() { render(); },
  };
  return loop;
}
