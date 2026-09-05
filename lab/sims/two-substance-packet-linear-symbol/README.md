# Two-substance packet linear symbol and polarization controls

This deterministic fixture implements the verification contract delivered by
Principia task `ORB-11219` for the existing two-density/gradient-flux packet
Hamiltonian. It asks whether setting `c_plus = c_minus` changes two scalar
sound branches into a Maxwell-like transverse sector. It does not add a
coupling, modify the packet apparatus, make an observational claim, or edit
Principia.

Run the full fixture and its byte-reproducibility check from the repository
root:

```sh
uv run lab/sims/two-substance-packet-linear-symbol/main.py --check-determinism
```

The committed `assets/results.json` is the authoritative machine-readable run.
It records the Orrery and Principia source commits, apparatus SHA-256,
equations, cases, solver, deterministic Fourier seeds, thresholds, exact rerun
commands, every gate, and the result-neutral verdict rule.

## Independent calculations

The matrix lane assembles the six-field Fourier operator in the species basis
`(n1,n2,j1x,j1y,j2x,j2y)`. The time lane separately uses the exact nonlinear
chemical potentials, periodic centered derivative, and symmetric
kick-drift-kick step copied from the frozen packet apparatus. Small isolated
Fourier perturbations measure phase frequency and flux direction without
changing the model.

The predeclared U0/U1/U2/B0/T0 controls cover unequal and equal speeds,
`lambda=0` and `lambda=0.8`, the admissible mixed background, both sound
branches, axial and diagonal wavevectors, and transverse current. Three spatial
and three temporal rungs separate centered-difference dispersion and lattice
anisotropy from continuum mode content. An explicit indefinite-Hessian witness
is rejected before evolution.

## Scoped result

The validated run refutes the apparatus hypothesis: it finds four propagating
roots in two positive-frequency longitudinal acoustic branches, plus two
zero-frequency transverse current modes and no algebraic/Gauss constraint.
Equal speeds make the two longitudinal eigenvalues degenerate; this is not
reported as Maxwell evidence. Transverse seeds remain frozen and conserve each
species' discrete curl. This result concerns only the frozen Hamiltonian, not
every possible emergent-gauge construction.

The `lambda=0.8` mixed background is locally stable for the tested B0 values,
but the cubic Hamiltonian remains globally unbounded. A positive sampled patch
is not treated as a cure.
