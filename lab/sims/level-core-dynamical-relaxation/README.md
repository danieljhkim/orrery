# Level-core extended dynamical relaxation

This apparatus advances the moving closed-system equations themselves. It does
not impose Bernoulli speed or a fixed-flux core boundary. A normalized Gaussian
draw sources the exactly comoving elliptic level, and wind enters only through
the upstream face. ORB-10938 extends ORB-10937's `T=60` initial-value experiment
to `T=600`, long enough to distinguish settlement from a stalled residual or
continued slow decay.

Run the cataloged experiment with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-10938 uv run lab/sims/level-core-dynamical-relaxation/main.py --check-determinism
```

The command executes the complete experiment twice, asserts byte-identical JSON
encodings, and writes `assets/results.json` plus
`runs/2026-08-21-extended-seed-42.json`. The prior ORB-10937 run record remains
at `runs/2026-08-21-seed-42.json` as the finite-horizon baseline.

## Numerical apparatus

The literal equations are advanced on a fixed 24-unit cube with a physical
Gaussian width of 0.75. The complete four-wind ladder uses `33^3` and `41^3`;
`61^3` supplies a full-horizon anchor at `U/v_GP(r=5)=0.3`. A cell-centered
donor-cell operator supplies truncation diffusion but no physical viscosity or
relaxation term. SSP-RK2 uses adaptive `CFL=0.22`; density has a `1e-10`
positivity floor. The ramped-pure-wind start raises the draw force over three
time units. The attractor probe evolves that start and the wind/GP blend
together at `33^3`, `U/v_GP=0.3`, recording their relative velocity and density
L2 differences throughout `T=600`.

Upstream velocity and density are prescribed. The other five faces copy the
adjacent interior value as a one-sided open treatment. The full requested
horizon was feasible and completed at both base rungs and the finer anchor.

## Predeclared gates

- G1 reports a settled/stalled/still-decaying verdict for every `(U,rung)` and
  separate velocity/density verdicts. Settlement requires both volume-RMS
  residuals below `2e-3` for the final 11 samples; a failing sector with late
  log slope below `-0.002/time` is still decaying, otherwise stalled. Minimum
  and `1.5 <= r <= 3` shell-mean density must each change by at most 1% over the
  final window for the separate trough-saturation diagnostic to pass. G1 also
  names cavitation separately: a cutoff contact is cavitated, while final
  `n_min < 0.01` with log slope below `-0.002/time` is a cavitation candidate;
  cutoff contact or adjacent-rung candidate agreement kills the excitation
  reading. Adjacent-rung velocity/density residual slopes and the full late
  `n_min` trajectory are recorded.
- G2 records net six-face mass influx, integrated consumption, their ratio,
  and its late trend for every case. Flux balance targets a ratio of one.
- G3 admits every velocity-settled case without waiting for density. It reports
  `|v|^2 - U^2 - 2 c^2 sigma` pointwise and in shell multipoles, along with the
  equivalent speed residual and temporal/adjacent-rung errors.
- G4 applies the same admission and compares direction multipoles to both the
  boosted static GP field and ORB-10935's marched branch, then reports
  stagnation. Vorticity is reported on every run, settled or not.
- G5 records the late consumed-momentum integral and advective far-surface
  flux, comparing the measured law to ORB-10937's `55.91 U^0.979` and
  ORB-10935's `20.3 U^0.098` fits.
- G6 advances pure wind/no core to `T=600` through the identical integrator and
  boundary path at every executed rung/wind.

## Result

All nine `(rung,wind)` runs meet the joint volume-RMS residual criterion by
`T=600`, and the two `33^3`, `U/v_GP=0.3` starts converge to relative L2
differences of 0.75% in velocity and 0.25% in density. That residual verdict is
not the whole story: the trough saturates only at `41^3`, `U/v_GP=1`. At
`U/v_GP=0.3`, `n_min` remains below 0.01 and keeps an exponential cutoff-ward
trend on `33^3`, `41^3`, and `61^3`, so the amended G1 cavitation gate kills the
excitation reading. The two lower winds have a cavitation candidate only on
`41^3` and therefore do not pass the adjacent-rung kill criterion.

Mass influx/consumption reaches 0.40-0.67 at the two lowest winds, 0.93-0.95 at
`U/v_GP=0.3`, and 0.998-0.999 at the highest wind. On velocity-settled cases,
the converged squared Bernoulli residual exceeds its temporal/resolution error
at both measured shells for `41^3`, `U/v_GP=0.3` and again at radius 3 on the
`61^3` anchor, refuting the closure. The settled `41^3` consumed-momentum fit is
`81.24 U^1.332`, rather than either prior fit. The extended Galilean null passes.

The ORB-10751 consumption function remains frozen byte-for-byte at SHA-256
`aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461`.
The results artifact records the measured match. Theory reconciliation is left
to Kepler; no principia files are touched.
