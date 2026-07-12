# Two-substance vortex lattice

This 2-D static lattice is a minimal sign test for the two-substance vortex
vacuum proposal. Each cell holds separately conserved A and B densities under
a shared unit-capacity pressure. Integer windings enter only through the
relative phase `theta_A - theta_B`; positive and negative polarity therefore
mean `(w_A, w_B) = (1, 0)` and `(0, 1)`. A balanced defect is `(1, 1)`.

Run the reproducible sweep with:

```sh
uv run lab/sims/two-substance-vortex-lattice/main.py
```

The generated `assets/results.json` is the numerical record and
`assets/interaction-sweep.png` plots the pair potential, signed force, and
short-range decomposition.

## Result (257 x 257 lattice, seed 42)

- Like polarity repels (mean far force `+0.13747`); opposite polarity attracts
  (`-0.13744`). This gives the electrostatic sign, not the reversed Bjerknes
  sign.
- The fitted far-force exponent is `-1.051`. A logarithmic potential fit has
  RMSE `0.0070`, versus `0.1383` for a `1/r` potential. The apparatus therefore
  resolves the expected 2-D log potential / `1/r` force.
- Across 8--16 cells, the density/capacity term adds `+0.01360` to the mean
  like-polarity phase force `+0.54240`, a `2.51%` strengthening from overlapping
  same-substance halos. It is not monotonic contact strengthening, however: the
  density term switches sign below 12 cells and contributes `-0.0738` at the
  closest 8-cell probe. The ontology's halo mechanism appears as a small
  intermediate-range correction, while core overlap softens rather than
  strengthens repulsion in this parameterization.
- A balanced defect has zero relative-phase charge and a `0.584` central void
  fraction. Its far force with either polarity is `2.58e-5`, only `1.88e-4` of
  the charged-pair force scale; this is the finite density-tail residual and is
  consistent with far-field neutrality.

## What the apparatus can decide

- **Sign:** pair energy is measured after subtracting both isolated-defect
  energies. Positive force means increasing separation lowers energy
  (repulsion); negative force means attraction.
- **Law:** a 2-D vortex has a logarithmic pair potential and a `1/r` force.
  The sweep compares a log-potential fit with an inverse-distance potential
  and fits the force exponent. This apparatus cannot establish the 3-D
  `1/r^2` force: a follow-up must simulate vortex lines or rings in 3-D.
- **Short range:** the force is split into relative-phase and density/capacity
  terms, exposing whether overlapping conserved halos strengthen or weaken
  the like-polarity interaction.
- **Balanced state:** equal co-winding cancels the relative phase exactly. The
  shared density core remains depleted, so neutrality and a void core are
  independently measurable.

## Model boundary

The relative-phase coupling is a declared model choice, not an emergent result:
it makes the topological charge `q = w_A - w_B`. The sweep genuinely measures
the consequences (interaction sign, lattice corrections, and force law), but
does not demonstrate that a microscopic two-fluid vacuum must select this
coupling. Defects are pinned analytic winding fields with relaxed-form core and
halo profiles; the script does not test dynamical defect stability or motion.
