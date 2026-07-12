# Two-substance defect gravity bridge

This experiment asks whether the conserved two-substance lattice used by the
companion vortex-interaction sim supplies the missing mechanics claimed by
gravity-as-scarcity. The questions and outcomes were fixed before the run:

| question | observable | supporting outcome |
| --- | --- | --- |
| stationary far field | spherical accumulated signed depletion | a resolved `1/r` tail from the conserved defect |
| moving wake | depletion-centroid lag and fitted pattern velocity | a following wake with pattern/core velocity ratio near one |
| gravitational charge | factorial core-size × winding sweep | far amplitude varies uniquely with void volume or flow energy |

Run the deterministic apparatus with:

```sh
uv run lab/sims/two-substance-defect-gravity/main.py
```

It writes the complete numerical record to `assets/results.json` and the four
diagnostic panels to `assets/gravity-bridge.png`.

## Apparatus

The stationary experiment uses a `97^3` cubic lattice. Each cell begins with
`rho_A = rho_B = 0.5`; a balanced defect removes both substances from a
Gaussian core and places exactly the same signed amount in a broader Gaussian
halo. The two fields are separately conserved to floating-point tolerance and
remain positive. Spherical shell sums define the potential-like accumulated
depletion. An uncompensated core is run as a positive control: if a monopole is
present, this estimator must recover its `1/r` exterior.

The wake experiment evolves a `161^2` slice with diffusion plus finite-time
relaxation toward the same compensated profile while its center is translated
at 0.15, 0.30, and 0.45 cells per time. This is a forced trajectory, not an
emergent moving soliton. The density response and diffusive replenishment flow
are measured rather than translated analytically. The 0.30 run is repeated at
half the time step.

The source sweep crosses four core radii with windings 0–3. Core size varies
positive void volume; winding varies a regularized individual-phase kinetic
energy diagnostic. This is deliberately factorial, so either quantity could
change independently. No gravity coupling is inserted.

## Measured result

- **Far field: negative.** The positive void volume is `122.8834` cell-density
  units and the expelled halo is `-122.8834`, leaving a signed monopole of
  `-2.0e-14` (`1.6e-16` of the positive volume). Across radii 36–44 the
  compensated signal is `3.95e-7` of the uncompensated control and is residual
  halo tail, not a resolved power-law field. The control follows `1/r` with
  relative RMSE `2.13e-16`; its `1/r^2` alternative has relative RMSE `0.0524`.
  The null is therefore not a blind estimator. The parent profile requires a
  net source, a nonconserving sink, or another long-range equation absent here.
- **Wake: qualified positive.** Finite relaxation produces a lagging asymmetric
  depletion/replenishment pattern. At imposed speed `0.30`, the fitted pattern
  speed is `0.2999992` (ratio `0.999997`) with a `0.586`-cell mean lag; the
  behind/ahead positive-depletion ratio is `2.21`. Ratios at speeds `0.15` and
  `0.45` are `0.999990` and `0.999997`. Halving the time step changes fitted
  speed by `5.0e-7` fraction and lag by `2.34%`. This supports the kinematic
  possibility of a following wake, but the core is externally driven, so it
  does not establish dynamical defect motion or stability.
- **Which quantity gravitates: neither is selected.** In the 16-configuration
  factorial sweep, positive void volume explains `R² = 0.9925` of the compact
  response, while flow energy explains `R² = 0.00032`; changing winding at a
  fixed core changes compact scarcity by exactly zero. All conserved
  configurations retain a zero far monopole within floating-point tolerance,
  so far-field predictor fits are undefined. A void-sourced Poisson field and
  an energy-sourced Poisson field can each be added as alternative postulates,
  but the present lattice contains no measurement or equation that chooses one.

## Interpretation boundary

This result tests the apparatus, not the theory ledger. It does not convert the
null monopole into a verdict about equivalence-principle data, and it does not
identify the diagnostic winding energy with gravity. Reconciliation of the
gravity-bridge and kill-condition-A rows belongs to principia/kepler. The
experimental handoff is: the conserved defect model supplies a compact void and
a forced wake, but no nonzero gravitational far field and no unique gravitating
charge.
