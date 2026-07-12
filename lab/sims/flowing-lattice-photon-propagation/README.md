# Flowing-lattice photon propagation

## Declared test

This apparatus propagates a geometric-optics wave at constant local speed in
the rest frame of an inward-moving lattice. Its Hamiltonian is
`H = v(x)·k + c_local(x)|k|`, so the laboratory-frame group velocity is the
lattice flow plus the comoving signal velocity. It measures signed deflection
and fixed-endpoint transit time, then superposes the static branch's
wrong-signed index gradient and measures the competing contributions.

The primary `|v| = sqrt(r_s/r)` arm is an imposed Gullstrand–Painlevé
surrogate. It is a controlled test of the conditional propagation claim, not
an emergent rule of the counting lattice: ORB-10162 instead measured
`v ∝ r^(-2.00062 ± 0.00002)` for central-only consumption and
`v ∝ r^(-0.99454 ± 0.00036)` for equal-per-shell consumption. Both measured
shapes are included as secondary arms. Because ORB-10162's velocity scale is
apparatus-specific, all profiles are normalized to the primary speed at the
impact radius; only their measured radial shapes are transplanted.

Run:

```text
uv run lab/sims/flowing-lattice-photon-propagation/main.py
```

The command rewrites `assets/results.json` and `assets/propagation.png`.
`results.json` is the full-precision report. Numerical uncertainties are the
maximum change from the finest run across DOP853 maximum steps 1.6, 0.8, and
0.4 `r_s/c`; invariant checks abort the run if signs, profile arms,
convergence, or contaminant scaling fail.

## Apparatus

Units set the far-field comoving wave speed and Schwarzschild radius to one.
A ray enters at `(-400 r_s, 20 r_s)` with horizontal wave vector and exits at
`x=+400 r_s`. The acoustic-Schwarzschild, `gamma=1` weak-field references are
`alpha = 2 r_s/b = 0.1 rad` and the finite-endpoint Shapiro expression
`r_s ln((r1+r2+R)/(r1+r2-R))`.

The standing contaminant uses the static arm's point-density profile
`n(r)=1-A(r_s/r)^2`. Constant one-hop-per-tick propagation in that lattice has
comoving speed `c_local=1/n`; its gradient deflects away. For each nonzero
`A`, the report contains flow-only, index-only, and combined deflections, the
ratio `|alpha_flow|/|alpha_index|`, their numerical uncertainties, and the
nonlinear remainder.

## Results

The imposed `r^-1/2` flow deflects the ray **toward** the well by
`0.113973184 ± 3.3e-15 rad`, or `1.13973` times the leading-order `gamma=1`
reference. The transit anomaly is a **delay** of
`9.372513732 ± 4.6e-13 r_s/c`, `1.27016` times the finite-endpoint
leading-order Shapiro reference. The excesses over unity are finite-field
measurements at `r_s/b=0.05`, not fitted adjustments to the reference.

With the same flow speed at the impact radius, the ORB-10162 central-only
shape produces `0.182659853 ± 4.4e-15 rad` toward-deflection and a
`7.802663395 ± 3.5e-13` delay. The equal-per-shell shape produces
`0.135935178 ± 5.0e-15 rad` toward-deflection and a
`6.148496052 ± 8.0e-13` delay. These controlled shape comparisons do not
assign a gravitational normalization to ORB-10162's lattice velocities.

The static index term deflects in the opposite direction throughout the
amplitude sweep. The measured flow-drag/index-gradient magnitude ratio falls
monotonically from `7256.22` at `A=0.002` to `121.017` at `A=0.12`; at the
largest amplitude the separate index contribution is `+0.000941794 rad`
(away), while the combined ray remains at `-0.112659795 rad` (toward). The
full sweep, nonlinear remainders, transit anomalies, and per-run uncertainties
are retained in `assets/results.json`.

These are apparatus measurements only. Theory and literature reconciliation
belongs to kepler.
