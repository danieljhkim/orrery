# Two-substance dynamical packet budget

This fixture asks the radiation-sector question upstream of any gravity source
law: does an actual propagating disturbance of the relative mode
`n_minus = n1 - n2` carry a local, co-moving structure in the common mode
`n_plus = n1 + n2 - 2 n0`, and how does that structure scale with packet
energy?

Run the deterministic apparatus with:

```sh
uv run lab/sims/two-substance-dynamical-packet/main.py
```

It writes the complete machine-readable record to `assets/results.json` and the
field/budget summary to `assets/packet-budget.png`. The dated, predeclared run
record is `RUN-2026-08-21.md`.

## Predeclared apparatus

The 2-D periodic lattice evolves densities `n1`, `n2` and individual vector
fluxes `j1`, `j2`. Its Hamiltonian density is

```text
H = (|j1|^2 + |j2|^2)/2
  + c_plus^2 n_plus^2/4 + c_minus^2 n_minus^2/4
  + lambda n_plus n_minus^2/4.
```

Each density obeys `partial_t n_s = -div(j_s)` and each flux obeys
`partial_t j_s = -grad(partial H / partial n_s)`. Periodic centered differences
and a symmetric kick-drift-kick update make each density sum a telescoping
finite-volume budget. The `c_minus` stiffness carries forward ORB-10163's
relative-phase hypothesis: in the irrotational sector `j_s = grad(theta_s)`,
so `J_minus` is the gradient of `theta1 - theta2`. Treating that relative
phase/current channel as physical is a model choice. The symmetric cubic
`lambda` coupling is a new, explicit model hypothesis; it is not derived from
a microscopic theory.

The initialization is mode (i): a localized zero-integral derivative-Gaussian
has `n1` up and `n2` down in its central lobe, exactly zero initial `n_plus`, and
right-moving relative flux. Four amplitudes (`0.02`, `0.04`, `0.06`, `0.08`)
form the energy ladder. The co-moving measurement window is fixed in advance at
`|x-x_centroid| <= 3 sigma_x`, `|y| <= 3 sigma_y`.

## Gates and interpretation

- **G1:** separately conserved substance totals, grid/time convergence of the
  small-amplitude speed, and a packet verdict based on energy retention plus
  translated-profile correlation.
- **G2:** global common/relative integrals versus time. These are calibration
  checks, not discoveries.
- **G3 (primary):** the signed co-moving common profile and the exponent in
  `integral |n_plus| = a E^p`.
- **G4:** relaxation of the fixed initialization region, including each
  substance's integrated boundary inflow and closure residual.
- **G5:** the global far-monopole null compared with the summation floor.

The apparatus is intentionally 2-D. It does not settle three-dimensional
propagation, derive an energy-to-gravity normalization, reopen the ORB-10164
far-monopole result, fit observations, or edit the principia theory ledger.

## Measured result (2026-08-21)

- **G1 passes.** Both substance totals drift by at most `1.82e-12`, below the
  `6.55e-11` summation floor. The finest grid-speed change is `0.412%` and the
  finest timestep change is `0.0370%`; the measured small-amplitude packet
  speed is `0.89345`. All four ladder members retain at least `95.76%` of their
  relative energy in the co-moving window and retain translated-profile
  correlations of at least `0.7267`: the explicit verdict is **propagates with
  bounded spreading**, not immediate dispersion. Total-energy drift is at
  most `1.32e-6` fraction.
- **G2 passes as calibration.** The maximum global common- and relative-mode
  integrals are `4.22e-13` and `7.81e-13`, respectively, below the same
  `6.55e-11` floor. Their full time series is in `assets/results.json`.
- **G3 is the primary positive result.** The co-moving common-mode magnitude
  follows `integral |n_plus| = 2.61433 E^0.998735` across the four-energy
  ladder. The resolved structure contains a positive pocket (`+0.64946`) and
  compensating negative structure (`-0.76999`) in the predeclared window.
  Thus this declared nonlinear dynamics supplies an energy-proportional common
  hook. The normalization is not a derivation of `E/c^2`, and the hook depends
  on the declared `lambda` coupling.
- **G4 passes.** `97.46%` of the initially localized energy leaves the fixed
  initialization region. The individual `n1` and `n2` integrated-flux closure
  residuals are `1.78e-15` and `-3.02e-14`; the final local common integral is
  `-0.10048`, recording the trailing refill/relaxation structure rather than a
  fully relaxed zero at this finite time.
- **G5 passes.** The maximum global common-mode monopole is `4.22e-13`, null at
  the `6.55e-11` instrument floor. Local co-moving common content and a global
  far monopole are therefore distinct observables in this conserving model.
