# Two-substance packet coupling and wake test

This fixture tests whether the local common response accompanying a
relative-mode packet is genuinely packet-bound or is a wake driven by the
inserted nonlinear coupling. It preserves the original two-substance dynamics
and distinguishes the signed global common monopole from local absolute common
content.

Run the deterministic apparatus with:

```sh
uv run lab/sims/two-substance-dynamical-packet/main.py
```

It writes the complete machine-readable record to `assets/results.json` and a
four-panel summary to `assets/packet-budget.png`. The dated validated run is
recorded in `RUN-2026-08-21.md`.

## Predeclared apparatus

The periodic 2-D lattice evolves `n1`, `n2` and their individual vector fluxes
under

```text
H = (|j1|^2 + |j2|^2)/2
  + c_plus^2 n_plus^2/4 + c_minus^2 n_minus^2/4
  + lambda n_plus n_minus^2/4.
```

Each density changes only through a telescoping discrete divergence. The
four-amplitude ladder `0.02, 0.04, 0.06, 0.08` is run at `lambda = 0, 0.2,
0.5, 0.8`. The zero-coupling arm is the causal null: without the cubic term,
the declared equations contain no driver from `n_minus` into `n_plus`.

Every run records time-resolved y-integrated common and relative profiles.
Signed, absolute, positive, and negative content; their centroids and absolute
widths; translated-relative and common/relative correlations; fitted profile
speeds; centroid lag; and lag normalized by packet width are machine-readable.
For samples after 40% of the run:

- **packet-attached** requires a resolved signal and at least 80% of resolved
  samples with `|lag|/width <= 0.75` and absolute-profile correlation at least
  `0.50`, plus common and relative fitted speeds within 10%;
- **trailing wake** requires at least 80% resolved late samples, median
  `lag/width < -0.75`, median correlation at least `0.50`, and a common-profile
  speed more than 10% below the relative packet speed;
- all other resolved outcomes are reported as mixed or detached.

The primary `lambda=0.8` four-amplitude fit is repeated on grids `128, 192,
256` and timesteps `0.09, 0.06, 0.03`. Convergence requires a finest-rung
exponent shift no larger than `0.02`, prefactor change no larger than 5%,
normalized-lag shift no larger than `0.10`, and an unchanged profile
classification. Speed alone is not a convergence gate.

## Measured result (2026-08-21)

- **The common response is imposed by the coupling.** At `lambda=0`, the
  largest common magnitude is `2.42e-12`, below the declared `2.62e-10`
  numerical floor. At nonzero lambdas, the energy exponents are `0.999918`,
  `0.999495`, and `0.998717`, while the prefactors are `0.61458`, `1.53432`,
  and `2.44866`. Across nonzero lambda, the common response has lambda
  exponents `0.99744–0.99984`: it is linear in the inserted coupling to this
  accuracy, not an endogenous lambda-independent effect.
- **The representative structure is a trailing wake.** Its median late lag is
  `-0.84129` packet widths, its median absolute-profile correlation is
  `0.70179`, and its common-profile speed is `0.67265` versus `0.96959` for the
  relative packet. It therefore passes the predeclared wake criterion and
  fails the attachment speed/lag criterion.
- **The primary G3 observable converges under the declared tests.** The finest
  grid-rung shifts are `8.66e-5` in exponent, `0.881%` in prefactor, and
  `0.0119` packet widths in lag. The finest timestep shifts are `4.31e-7`,
  `0.00244%`, and `0.00287`, respectively. Both refinement arms retain the
  same wake classification.
- **Endogenous defect emission is unsupported.** The existing packet equations
  directly initialize the disturbance; the companion defect apparatus pins
  analytic cores and externally translates a relaxation target. There is no
  operator coupling an autonomous defect and its energy to outgoing
  `J_minus`/`n_minus` flux with a closed defect-plus-field energy/refill budget.
  No ad hoc source was inserted.
- **The original cubic Hamiltonian is globally unbounded for every nonzero
  lambda.** Minimizing over `n_plus` leaves a negative quartic term in
  `n_minus`. This sampled weak-field domain remained finite and positive
  (`min n1 = 0.97524`, `min n2 = 0.96000`), with nonnegative sampled energy
  density, but that local observation does not cure the analytic instability.
  No stabilizing variant was added.
- **Conserved budgets remain clean.** Each substance drifts by at most
  `1.82e-12` against a `6.55e-11` floor; the common monopole remains below
  `5.79e-13`, the relative total below `8.96e-13`, and maximum relative energy
  drift is `1.32e-6`.

The absolute local common profile is not a net gravitational mass. This
fixture introduces no gravity-source law, `E/c^2` normalization, observational
fit, or Principia change.
