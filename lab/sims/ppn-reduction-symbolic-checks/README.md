# PPN reduction symbolic checks

This deterministic SymPy fixture turns the four exact algebra rows in
principia's `theory/ppn-reduction-of-the-settled-flow.md` into cataloged
pass/fail gates and executes its predeclared half-order hazard.

Run it with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11040 uv run lab/sims/ppn-reduction-symbolic-checks/main.py --check-determinism
```

The command executes every gate twice, requires byte-identical sorted JSON,
and writes `assets/results.json`.

## Exact gates

1. **Statement 0:** pulls the mass-frame acoustic metric through
   `x_reservoir = x_mass + U t`, verifies the result is exactly the PG metric
   of `u = v + U`, substitutes `v = u - U`, and verifies there is no remaining
   dependence on `U` at fixed `u`.
2. **Pure-wind flatness:** evaluates all 256 components of the four-dimensional
   Riemann tensor for the mass-frame pure-wind metric and proves they vanish
   identically, at all orders in the three symbolic wind components. The
   metric is independently checked as a pullback of Minkowski space.
3. **Diagonalization:** applies the exact temporal shift
   `grad(lambda) = -u/(c^2-|u|^2)`, verifies the shift vanishes and the spatial
   stretch is `delta_ij + u_i u_j/(c^2-|u|^2)`, then obtains the exact
   Schwarzschild lapse and `g_rr` for radial static GP flow.
4. **Two-piece obstruction:** verifies the curl product identity for three
   arbitrary symbolic functions. Both pieces vanish for a generic radial GP
   profile. The irrotational anisotropic field `(x, 2y, 3z)` has a nonzero
   speed-gradient piece, so vorticity and speed anisotropy are independently
   capable of obstructing the shift.
5. **Half-order bookkeeping:** expands `u/c = sqrt(epsilon) a + epsilon_w b`
   through linear wind order and `epsilon*epsilon_w`, classifies the temporal
   longitudinal/transverse split and the diagonal metric terms, and uses a
   nonzero symbolic `R_0i0j` witness to test whether a spatial coordinate
   choice can remove the mixed half-order term.

## Predeclared hazard result

**KILL on the generic wake branch:** the temporal shift moves the
`sqrt(epsilon)*epsilon_w` cross term into `h_00` and `h_ij`; it does not remove
it. Even when both leading flow pieces are potential, the fixture's
anisotropic wake witness has nonzero mixed-order curvature. A realized wake
can avoid this super-PPN effect only by satisfying additional cancellations
(including the relevant contraction/curvature conditions); the exact algebra
does not supply them. Reconciliation of that verdict and the measured wake is
Kepler's theory-lane work. This fixture does not edit principia.

The machine-readable artifact is authoritative and records every expression,
classification, exact zero, gate verdict, and the prominent hazard verdict.

## Provenance

The fixture implements Orbit task `ORB-11040`, requested by principia
`theory/ppn-reduction-of-the-settled-flow.md`, and belongs to the almanac thread
`15-discussions/26-08/from-galactic-motion-to-the-moving-gravity-medium-problem.md`.
