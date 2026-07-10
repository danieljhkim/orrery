// Minimal vector helpers over plain arrays ([x,y] or [x,y,z]).
// Out-params avoid per-frame allocation in hot loops; every function
// also returns its result so casual code can chain.

export const add = (a, b, out = []) => { for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]; return out; };
export const sub = (a, b, out = []) => { for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i]; return out; };
export const scale = (a, s, out = []) => { for (let i = 0; i < a.length; i++) out[i] = a[i] * s; return out; };
export const addScaled = (a, b, s, out = []) => { for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i] * s; return out; };
export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const norm = (a) => Math.sqrt(dot(a, a));
export const normalize = (a, out = []) => scale(a, 1 / (norm(a) || 1), out);
export const lerp = (a, b, t, out = []) => { for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t; return out; };
export const cross = (a, b, out = []) => {
  const [ax, ay, az] = a, [bx, by, bz] = b;
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
};
export const dist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); };
