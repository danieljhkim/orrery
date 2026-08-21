# Level-core wind tunnel

This deterministic 3-D apparatus puts the draw-sourced level closure in a
uniform wind. A fixed-width Gaussian cell draw sources the instantaneous
comoving level with the seven-point Poisson operator. There is no velocity or
fixed-flux condition at the core. In the core frame, wind enters only through
the upstream outer face.

Run:

```sh
UV_CACHE_DIR=/tmp/orbit-uv-cache-10935 uv run lab/sims/level-core-wind-tunnel/main.py --check-determinism
```

The command writes `assets/results.json` and
`runs/2026-08-21-seed-42.json`. The determinism option executes the entire
three-rung, four-wind experiment twice in memory and requires byte-identical
JSON encodings.

## Apparatus decision

The comoving-frame formulation keeps the elliptic draw level exactly centered
and supplies `+x` wind at the upstream outer face. For each rung, a
predictor-corrector Hamilton-Jacobi march follows the positive-`x` potential
branch satisfying

`|grad Phi|^2 = U^2 + 2 sigma`.

This choice isolates the theorem's Bernoulli speed from the direction field.
Its tradeoff is explicit: it can represent only a single-valued downstream
graph. When transverse characteristics form a caustic, the march limits them
at `0.999 q` and records the affected fraction. A failed steady-existence
diagnostic is therefore a measured result, not silently smoothed away.

The `41^3 -> 61^3 -> 81^3` ladder holds the 24-unit box, 0.75-unit Gaussian
core width, physical shell radii, probe radius, and control volume fixed. Wind
ratios `{0.03, 0.1, 0.3, 1}` use the finest-rung static river speed at radius 5
as their common reference.

## Predeclared gates

The machine-readable artifacts are authoritative and contain every case,
finest-rung shift, criterion, and verdict.

- G1 reports the speed monopole, normalized dipole vector, and axisymmetric
  `l=1..3` coefficients on radii 3 and 5 for all four winds, together with
  discretization errors and pointwise Bernoulli residuals.
- G2 reports unit-direction multipoles, fore-aft asymmetry, angular distance
  from `U + v_GP`, stagnation geometry, and the steady-branch diagnostic.
- G3 reports `integral s v_x dV`, advective momentum flux through the fixed far
  cube, their signed residual, convergence, and a four-wind drag power law.
- G4 executes uniform no-core wind at every rung and wind strength.

The copied coefficient-one `strain_consumption_3d` function must hash to
`aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461`,
byte-identical to ORB-10751, ORB-10932, and ORB-10934. The momentum surface
ledger intentionally reports only advective flux; it does not invent an
unprovided pressure/level stress. Theory reconciliation is deferred to Kepler,
and no principia files are changed.
