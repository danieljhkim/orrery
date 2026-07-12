# Dynamical-consumption lattice flow

This experiment evolves a spherical radial counting lattice from a full state.
Nearest-neighbour point hopping is proportional to the density deficit across
each face, a fixed-density outer ghost shell replenishes the lattice, and local
sinks destroy points. The `central-only` rule destroys points only beside the
mass. The `equal-per-shell` rule destroys the same absolute mean budget in every
equal-width radial shell. Both use identical hopping physics and sink strength
is chosen for an approximately 5% inner standing deficit.

Run `uv run lab/sims/dynamical-consumption-lattice/main.py`. The seeded run
writes the full measurements to `assets/results.json` and the diagnostic plot
to `assets/profiles.png`.

## Measured run (seed 42)

Both fields reached the declared stationarity tolerance: the largest final
step change was below `1e-9` of the maximum standing deficit. The shell-balance
residual was `9.4e-9` of the total sink for central-only and `7.9e-11` for
equal-per-shell.

Fits use radii 10–500, spanning 1.699 decades (491 face samples):

| local consumption rule | measured `v(r)` exponent ± SE | 95% CI | nearest candidate by log-RMSE |
| --- | ---: | ---: | --- |
| central-only | -2.00062 ± 0.00002 | [-2.00065, -2.00059] | `r^-2` (0.00060; versus 0.831 for `r^-1`, 1.246 for `r^-1/2`) |
| equal-per-shell | -0.99454 ± 0.00036 | [-0.99524, -0.99383] | `r^-1` (0.00803; versus 0.835 for `r^-2`, 0.411 for `r^-1/2`) |

The central-only sink is localized to the innermost shell, so it has no
distributed sink exponent. Poisson counting over 50 time units for the
equal-per-shell rule measures consumption per shell as
`r^(0.00029 ± 0.00026)` (95% CI [-0.00022, 0.00079]), discriminating the
implemented `r^0` rule from `r^(1/2)`.

The central-only standing deficit matches the finite-reservoir static shape
`A(1/r - 1/R)` with `R² = 0.99999992` and relative RMSE `3.60e-5`; a comoving
point accumulates no destruction before the innermost sink, so its dilution is
zero throughout the fitted range and does not reproduce that snapshot. Under
equal-per-shell consumption, the standing deficit does not match the static
shape (`R² = -0.838`, relative RMSE 0.293), and the comoving accumulated
dilution also does not (`R² = -17.30`, relative RMSE 1.237).

Using the lattice signal speed `c = D/dr`, neither measured branch satisfies
`sigma = v²/(2c²)`: the median flow/standing-sigma ratio over the fitted range
is `2.03e-9` for central-only and `4.13e-8` for equal-per-shell, with RMS
log10 discrepancies of 8.42 and 7.21 dex respectively.

These are apparatus measurements and numerical consistency checks only.
Interpretation of the theory ledger belongs to principia/kepler.
