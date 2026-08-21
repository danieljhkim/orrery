# Level-core two-core superposition

This deterministic full 3-D apparatus tests the last superposition prediction
of the level-matching closure. Two fixed-width Gaussian draw cores source a
standing level through a discrete Poisson solve with an isolated outer face.
The cores impose no velocity or flux boundary. Literal shared-capacity counting
stores the level as `sigma = 1-exp(-H)`, and paired background subtraction
isolates the target core's modulation.

Run:

```sh
UV_CACHE_DIR=/tmp/orbit-uv-cache-10934 uv run lab/sims/level-core-two-core-superposition/main.py --check-determinism
```

The command writes `assets/results.json` and
`runs/2026-08-21-seed-42.json`. It executes the experiment twice in memory and
requires byte-identical JSON encodings.

## Apparatus decision

The sim extends ORB-10932's draw-sourced level coupling to a genuinely
nonspherical 3-D field. This avoids carrying ORB-10755's fixed-flux core
boundary into the discriminator. The tradeoff is the same one-way coupling as
ORB-10932: the frozen shear rule diagnoses destruction of transported
substrate, while externally conserved core draws continue to source the
standing level.

The 25^3 -> 41^3 -> 65^3 ladder keeps the 24-unit box, 0.75-unit Gaussian core
width, core positions, draws, and physical 2--6 exterior measurement annulus
fixed. The wider window suppresses the finite-width core's near-field shape
without entering the outer boundary layer.
Every rung is interpolated in log(D)-log(A) onto the same eleven-point
depletion grid before fitting. A fixed-draw separation control repeats the
finest measurement at separations 6, 8, and 10.

## Predeclared results

The machine-readable artifacts are authoritative. The family gate compares:

- free headroom `A=(1-D)^p`;
- free flux form `A=1/(1+c D^beta)`;
- fixed ORB-10157 headroom `A=(1-D)^1.071`.

A family is decisive when its log-RMSE is at least three times smaller than
the competing free family. The recorded verdict is `confirmed`: the free
headroom form decisively beats the flux form, its resolution shifts shrink,
and the continuum-extrapolated `p` is consistent with 1.071 within the
combined apparatus error. The finest small-D derivative stays bounded and the
far-field level fits pass `A_pair^2 = A_1^2 + A_2^2`. The dated run record also
contains the full log-RMSE table, every fitted parameter and uncertainty,
resolution samples, separation measurements, and gate criteria.

The copied coefficient-one `strain_consumption_3d` function hashes to
`aa1155e07536c3318c0afb0baabbbf472d66658046be4d21d816f135632c8461`,
byte-identical to ORB-10751 and ORB-10932. Theory reconciliation is deliberately
deferred to Kepler; this task does not edit principia.
