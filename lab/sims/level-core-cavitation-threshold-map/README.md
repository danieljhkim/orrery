# Level-core cavitation threshold map

This cataloged fixture maps the live flooring-versus-cavitation uncertainty
left by ORB-10938 and ORB-11041. It imports ORB-10938's evolution module
directly and verifies the frozen shear stencil against SHA-256
`aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461`
at runtime. Bernoulli speed is imposed nowhere.

Run the authoritative experiment with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11170 \
  uv run lab/sims/level-core-cavitation-threshold-map/main.py \
  --check-determinism
```

The command executes the experiment twice, requires byte-identical canonical
JSON, and writes `assets/results.json` and
`runs/2026-09-04-seed-42.json`.

## Apparatus

The ratio throughout is `U/v_GP(r_core)`, with `r_core = 2 sigma` for the
Gaussian core. At fixed half-width 12, the sim samples three core sigmas and
five winds. Every cell touching a change among `{floors, candidate,
cavitated}` receives a `33^3 -> 41^3` ladder. The cavitating-side boundary
corner is then run at half-widths 12, 18, and 24 with a matched-spacing
two-rung ladder at every box. This is the predeclared feasibility tradeoff:
converge the cells that determine the reading instead of producing a dense
unconverged grid.

The joint residual settlement and trough classification are inherited from
ORB-10938. A `candidate` has `n_min < 0.01` and a final-window log slope below
`-0.002/time`; `cavitated` means contact with the `1e-10` positivity cutoff.
The results retain the joint settlement verdict beside every density verdict.

## Gates and interpretation

- G1 reports the full `(U/v_GP(r_core), core sigma, half-width)` table,
  boundary brackets, every decisive resolution ladder, and exactly one of
  the predeclared pinned/core-tracking/box-tracking readings.
- G2 fits `n_inf = a + b U/v_GP(r_core) + c sigma` on admitted flooring-side
  cells and evaluates whether its boundary-side value reaches zero.
- G3 compares Earth, Sun, and a representative neutron star only through the
  dimensionless ratio `U/v_GP` at the body's radius. It is explicitly **not**
  a lattice-to-physical normalization; the substrate-inertia debt remains
  open. Velocity scales are cited to principia
  `studies/lorentz-violation-bounds.md` § “The velocity scales,” with the
  task-predeclared 220 km/s Sun-in-Galaxy value labeled separately.
- G4 certifies the pure-wind/no-core Galilean fixed point at every swept
  geometry through the identical imported update.

No floor term or other new physics is introduced here, and no principia file
is modified. The numerical result and the predeclared G1 reading are recorded
in `assets/results.json`; theory reconciliation belongs to kepler.

## Result

At half-width 12, all three core sigmas give the same sampled boundary:
`U/v_GP(r_core) = 0.15` is a cavitation candidate and `0.30` floors, for a
bracket midpoint of `0.225`. The decisive endpoints carry `33^3 -> 41^3`
ladders. The box control overturns a pinned reading: at sigma 1 and ratio
0.15, half-width 12 is a candidate while half-widths 18 and 24 floor on their
matched-spacing ladders. G1 is therefore **G1c — box-tracking**; the
ORB-10938 kill was boundary-driven in this apparatus.

G2 admits seven flooring-side samples whose joint field settled or whose local
trough saturated. The fitted law is
`n_inf = 0.0502 + 0.3038 U/v_GP(r_core) - 0.0359 sigma` (RMSE 0.0410).
Its three boundary-side predictions are 0.0875--0.1144, all above the 0.01
cavitation threshold, so the measured transition is discontinuous at the
sampled boundary rather than a floor tending continuously to zero.

The G3 ratios are Earth 2.682, Sun 0.356, and a representative 1.4-solar-mass,
12 km neutron star 0.00210. Relative to the measured 0.225 midpoint, Earth and
Sun are above and the neutron star is below. Because G1 is box-tracking rather
than pinned, these are conditional dimensionless arithmetic only and admit no
physical gate verdict. The Galilean null passes exactly at all 12 unique sweep
geometries.
