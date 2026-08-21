# Moving-core acoustic rays

This deterministic fixture traces null rays through the acoustic metric of a
static or uniformly swept level core. It is a **model-side scale estimate**, not
an observational fit or falsifier. Its primary deliverable is the spread of
deflection and Shapiro-modulation estimates across three candidate direction
fields while the moving wake remains unsettled.

Run the cataloged experiment with:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-10939 uv run lab/sims/moving-core-acoustic-rays/main.py --check-determinism
```

The command rebuilds the predecessor fields, runs the entire ray and field
convergence ladder twice, requires byte-identical JSON encodings, and writes
`assets/results.json` plus `runs/2026-08-21-seed-42.json`.

## Apparatus

The stationary acoustic Hamiltonian is `H=v(x)·k+|k|`, giving
`dx/dt=v+k/|k|` and `dk/dt=-(grad v)^T k` at lattice convention `c=1`.
DOP853 traces rays across maximum steps `0.24, 0.12, 0.06`. Cataloged 3-D
fields are sampled on their `z=0` plane with bilinear velocity interpolation;
the Jacobian is the analytic derivative of that same cell polynomial.

The moving bracket is predeclared as:

- (b) the analytic point-mass GP field plus a constant Galilean wind;
- (c) ORB-10935's `81^3` marched Hamilton-Jacobi field, including its returned
  `0.999 q` caustic limiter values and an inferred, hashed node mask; and
- (d) ORB-10937's original finest `41^3`, `T=60` field, regenerated from the
  cataloged equations for every wind. The velocity is frozen for ray tracing
  even though density was still relaxing and attractor uniqueness was open.

Field-grid shifts use `61^3 -> 81^3` for (c) and `33^3 -> 41^3` for (d).
Four wind ratios, three impact parameters, and four ray orientations are kept
in the machine-readable record.

## GR comparator

G1 uses an analytic point-mass Painleve-Gullstrand field and compares the ray
integrator to an independent exact Schwarzschild null-orbit quadrature across
an impact ladder. The weak `4GM/(c^2 b)` value and the exact strong-field
correction are both recorded.

For G3, the source velocity in the substrate lab is `beta=-U`. Incoming lab
directions are Lorentz-aberrated into the mass rest frame, the static
Schwarzschild deflection is applied there, and the outgoing direction is
aberrated back. This is not the Galilean `U+v_GP` construction. The mass-rest
phase delay is mapped with the corresponding null-wave Doppler factor; the
chosen convention and every intermediate direction are stored explicitly.

## Results

G1 passes: across `b/r_s={8,12,20,32}`, the measured deflections agree with
the exact Schwarzschild target to `5.5e-10` through `9.9e-9` relative. The
strong-field correction above `4GM/(c^2 b)` ranges from 24.0% to 4.88%, while
maximum-step shifts remain below `1.4e-11 rad`.

At the headline `b=3`, field (b)'s modulation is approximately linear in
`U/c`: its half-range is `2.84-3.00` times `U/c`, with fitted exponent
`p=1.015`. Field (d) gives `2.05-6.11` per unit `U/c` and `p=0.717`. Field
(c) is not a robust low-wind scale estimate: its inferred limiter mask covers
73.5%, 71.1%, 57.2%, and 3.00% of nodes across the wind ladder, its grid shifts
are large, and its fitted amplitude decreases with `U` (`p=-0.722`). That is a
measured failure of this clipped candidate branch, not a hidden interpolation.

This sensitivity is G5's result. At `b=3`, the cross-field spread in G2's
normalized modulation amplitude falls from `3.622` at wind ratio `0.03` to
`0.151` at ratio `1`; the corresponding closed-minus-GR deflection spread
falls from `3.620` to `0.257`. The wake choice therefore dominates the
low-wind photon estimate and remains material even at the highest wind.

Full run results, log-fit quality, per-step errors, ray-by-ray caustic exposure,
GR differentials, and all impact parameters are authoritative in
`assets/results.json`. No raw `U/c` number is converted to a PPN coefficient;
that reduction, sourced observational walls, and theory reconciliation remain
separate obligations.
