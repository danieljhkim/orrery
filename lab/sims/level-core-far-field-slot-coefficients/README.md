# Level-core far-field slot coefficients

This fixture moves the ORB-10938 settled-flow apparatus into a 48-unit box
with a one-unit Gaussian core.  It is a new sim rather than an in-place
revision because both the geometry and the measured question change: the
shell ladder crosses the local gravity/wind matching radius and reaches the
wind-dominated zone where principia's PPN reduction defines its slots.

Run the authoritative deterministic experiment with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11041 \
  uv run lab/sims/level-core-far-field-slot-coefficients/main.py \
  --check-determinism
```

The command writes identical JSON to `assets/results.json` and
`runs/2026-08-28-seed-42.json`.

## Apparatus and feasibility

The sim imports ORB-10938's evolution module instead of copying it.  The
rolling rule, continuity equation, donor-cell/SSP-RK2 path, open faces, and
coefficient-one shear stencil are therefore the same implementation; the
stencil is checked at runtime against SHA-256
`aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461`.
Bernoulli speed is imposed nowhere.

The `41^3 -> 57^3` ladder, three winds, and `T=480` horizon are the explicit
feasibility tradeoff.  The shell radii `{3,5,8,11,14,17,20}` cross matching
for all winds before the half-width 24 boundary.  A failed adjacent-rung tail
test is reported as `no_scaling_regime`; the fixture never promotes a clean
fine-rung power law when its coarse rung disagrees.

## Gates

- G1 inherits ORB-10938's joint residual and density-trough/cavitation gates.
- G2 fits the disturbance-speed wind-axis `l=1,2,3` terms and both exact
  non-gauge fields.  Because the obstruction is one derivative above `g0i`,
  `r * obstruction` is compared with the `Phi ~ 1/r` potential ladder.
- G3 extracts the alpha1, alpha2, and `2 alpha3 - alpha1` lattice proxies only
  when the corresponding radial regime and adjacent-rung verdict converge.
  Faster decay returns a zero asymptotic slot; no regime returns no number.
- G4 tests `O(U)` scaling across all three winds only after the outer-shell
  amplitudes pass the adjacent-rung tolerance.
- G5 certifies the pure-wind fixed point at every rung/wind through the same
  update.  Exact one-step invariance inductively covers the recorded horizon.

The results are dimensionless lattice-unit coefficients only.  They are not
observational fits or physical alpha claims, and no principia files are
modified.

## Result

All six rung/wind runs meet the joint settlement criterion at `T=480`; the
minimum densities floor at `0.175--0.250`, with no cutoff-ward cavitation
candidate.  The outer shell is wind dominated by local ratios `2.25--3.46`,
and the first matching shells lie at radii 11, 8, and 5 respectively.

The central answer is mixed and the worst outcome is explicit.  The
disturbance-speed dipole has a converged slot-matching tail for all three
winds (`p = -0.94, -1.08, -1.11` against `r^p`), giving resolved nonzero
`2 alpha3 - alpha1` lattice proxies.  The speed quadrupole converges to a
slot-matching tail only at the middle wind, giving an alpha2 proxy
`52.2 +/- 13.2`; the other two alpha2 cases have no converged scaling regime.
The total non-gauge g0i dipole and its vorticity piece have no converged
scaling regime at any wind, so no alpha1 number is extractable.  The
Bernoulli-anisotropy obstruction decays faster than the PPN-equivalent tail
at the two higher winds, but has no regime at the lowest wind.

Across wind, the Bernoulli obstruction is resolution-converged and is not
`O(U)` (`U^-2.08` on this ladder).  Vorticity and the total obstruction remain
unresolved under refinement, so the prior per-wind convergence debt is not
retired.  The Galilean null passes exactly on every rung and wind.  These are
lattice-apparatus findings for theory handoff, not a phenomenological
confrontation with observational alpha bounds.
