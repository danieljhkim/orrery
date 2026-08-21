# Level-core dynamical relaxation wind tunnel

This apparatus advances the moving closed-system equations themselves. It does
not impose Bernoulli speed or a fixed-flux core boundary. A normalized Gaussian
draw sources the exactly comoving elliptic level, and wind enters only through
the upstream face.

Run the cataloged experiment with:

```sh
UV_CACHE_DIR=/tmp/orbit-uv-cache-10937 uv run lab/sims/level-core-dynamical-relaxation/main.py --check-determinism
```

The command runs the complete experiment twice, asserts byte-identical JSON,
and writes `assets/results.json` plus `runs/2026-08-21-seed-42.json`.

## Numerical apparatus

The literal equations are advanced on a fixed 24-unit cube with a physical
Gaussian width of 0.75 at `25^3`, `33^3`, and `41^3`. A cell-centered donor-cell
operator supplies shock-safe truncation diffusion but no physical viscosity or
relaxation term. SSP-RK2 uses an adaptive `CFL=0.22`; density has a `1e-10`
positivity floor. The pure-wind initial state ramps the draw force over three
time units. The alternate-attractor run starts from wind plus the static GP
field with full draw. Upstream velocity and density are prescribed; the other
five faces copy the adjacent interior value as a one-sided open treatment.

The observation horizon is `T=60`. This is still shorter than the two lowest-
wind box crossing times, so `not_steady_within_horizon` is deliberately weaker
than a claim of perpetual unsteadiness.

## Predeclared gates

- G1 calls a case steady only when volume RMS `dv/dt < 2e-3` and `dn/dt < 2e-3`
  for five final samples, all after `0.75 T`. Every `(U,rung)` retains its
  residual series; failures include a trend, dominant probe frequency, and
  growth/saturation classification.
- G2 is evaluated only for admitted steady cases. It measures pointwise and
  shell-multipole departures from `sqrt(U^2 + 2 c^2 sigma)`; adjacent-rung
  Bernoulli-RMS and normalized-dipole shifts must each be below 0.05.
- G3 compares realized direction multipoles to both boosted GP and ORB-10935's
  marched branch, searches for stagnation below `max(1e-3,0.05 U)`, and reports
  curl norms.
- G4 records the full consumed-momentum and advective surface-flux series,
  last-five-sample means/variability, finest wind scaling, and the ORB-10935
  comparator (`20.3 U^0.098`, drag).
- G5 advances pure wind/no core through the identical integrator and boundary
  path to `T=60`; velocity, density, and consumption must remain unchanged to
  floating precision on every rung and wind.

The ORB-10751 consumption function is frozen byte-for-byte at SHA-256
`aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461`.
The results artifact records the measured hash. Theory reconciliation is left
to Kepler; no principia files are touched.
