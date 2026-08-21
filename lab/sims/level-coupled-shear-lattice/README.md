# Level-coupled shear lattice

This apparatus tests whether a draw-sourced standing scarcity level, inertial
rolling, and the frozen coefficient-one von Mises shear rule dynamically settle
on the level-matching amplitude

`A² = 2 c² sigma_s r_s`.

It is a spherical finite-volume experiment. Core draw cells are summed by the
discrete Gauss law and the exterior level is integrated inward; the exterior
is not filled with an analytic `1/r` profile. Composition runs use `N`
individually counted unit-core draws in a deterministic quasi-uniform cluster,
then retain their radial monopole on the flow lattice. A semi-Lagrangian
momentum lattice then evolves the material rolling rule. A second transported
field evolves continuity and local shear destruction using the exact frozen
rule from `shear-consumption-lattice`. The outer Robin face represents rest at
infinity on the finite domain; it is not a fitted amplitude.

Run:

```text
uv run lab/sims/level-coupled-shear-lattice/main.py
```

The command writes the concise `assets/results.json` and the authoritative
full-precision `runs/2026-08-21-seed-42.json`. Verify a byte-identical in-memory
rerun with:

```text
uv run lab/sims/level-coupled-shear-lattice/main.py --check-determinism --no-write
```

## Predeclared gates

- **Level:** four draw strengths span three decades. The measured amplitude is
  fitted against `sigma_s r_s`, every run reports
  `A²/(2 c² sigma_s r_s)`, and mass one is evolved from rest and a seeded
  perturbed velocity/density state.
- **Density:** equal draw strength is repeated for core radii 0.5, 1, and 2, a
  fourfold span. The fitted `A(r_s)` exponent adjudicates level independence
  against the flux prediction `-3/2`.
- **Composition:** 1–32 unit draws are co-clustered at fixed packing density,
  represented by `r_cluster = N^(1/3)`. The far-field amplitude is fitted to
  `A(N)` and tested against exponent `1/2`.
- **Convergence:** the level case uses 96/192/384 shells plus CFL 0.4 → 0.2
  step halving. Successive spatial amplitude shifts must shrink.

The committed results artifact and run record contain each numeric estimate,
regression standard error, 95% interval, pass/kill criterion, and verdict.

## Limitations

The spherical reduction preserves the draw monopole but coarse-grains the
composition gate's nonspherical near field, so it does not test interactions
between resolved individual cores. The draw count sources the standing level
one-way: shear destruction evolves substrate density and checks the steady
continuity balance, but does not change the conserved draws. The finite outer
Robin face supplies the isolated rest-at-infinity reference. Regression errors
and numerical-convergence shifts are reported separately. This is an
apparatus-level handoff; principia theory documents are deliberately unchanged.
