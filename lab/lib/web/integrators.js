// Time integrators over flat Float64Array state (N particles * dim, packed).
//
// Conventions:
//   pos, vel, acc : Float64Array of equal length
//   computeAcc(pos, accOut) : fills accOut from pos (must overwrite, not accumulate)
//   deriv(y, dyOut) : fills dyOut = dy/dt for generic ODE state y

// Leapfrog kick-drift-kick: symplectic, 2nd order. The right default for
// gravity / orbital mechanics — conserves energy over long runs where RK4 drifts.
// `acc` must hold computeAcc(pos) from the previous step (or call primeAcc once).
export function leapfrogKDK(pos, vel, acc, dt, computeAcc) {
  const n = pos.length, h = dt / 2;
  for (let i = 0; i < n; i++) vel[i] += acc[i] * h;   // kick
  for (let i = 0; i < n; i++) pos[i] += vel[i] * dt;  // drift
  computeAcc(pos, acc);
  for (let i = 0; i < n; i++) vel[i] += acc[i] * h;   // kick
}

export function primeAcc(pos, acc, computeAcc) {
  computeAcc(pos, acc);
}

// Semi-implicit (symplectic) Euler: cheapest stable choice for casual dynamics.
export function semiImplicitEuler(pos, vel, acc, dt, computeAcc) {
  computeAcc(pos, acc);
  for (let i = 0; i < pos.length; i++) {
    vel[i] += acc[i] * dt;
    pos[i] += vel[i] * dt;
  }
}

// Classic RK4 for a generic first-order system y' = f(y).
// Non-symplectic: prefer leapfrog for long-lived orbits, RK4 for driven /
// dissipative systems where accuracy per step matters more than energy drift.
export function rk4(y, dt, deriv) {
  const n = y.length;
  const k1 = new Float64Array(n), k2 = new Float64Array(n),
        k3 = new Float64Array(n), k4 = new Float64Array(n),
        tmp = new Float64Array(n);
  deriv(y, k1);
  for (let i = 0; i < n; i++) tmp[i] = y[i] + k1[i] * dt / 2;
  deriv(tmp, k2);
  for (let i = 0; i < n; i++) tmp[i] = y[i] + k2[i] * dt / 2;
  deriv(tmp, k3);
  for (let i = 0; i < n; i++) tmp[i] = y[i] + k3[i] * dt;
  deriv(tmp, k4);
  for (let i = 0; i < n; i++) y[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6;
}

// Pairwise softened Newtonian gravity: fills acc from pos (dim-packed) and
// masses. eps2 is the softening length squared (avoids the r→0 singularity).
export function nBodyGravity(pos, masses, acc, { G = 1, eps2 = 1e-8, dim = 3 } = {}) {
  acc.fill(0);
  const n = masses.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let r2 = eps2;
      for (let d = 0; d < dim; d++) {
        const dx = pos[j * dim + d] - pos[i * dim + d];
        r2 += dx * dx;
      }
      const inv = G / (r2 * Math.sqrt(r2));
      for (let d = 0; d < dim; d++) {
        const dx = pos[j * dim + d] - pos[i * dim + d];
        acc[i * dim + d] += dx * inv * masses[j];
        acc[j * dim + d] -= dx * inv * masses[i];
      }
    }
  }
}
