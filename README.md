# orrery

A cabinet of accumulated physics simulations — interactive canvas/three.js sims
and Python Monte Carlo experiments, mostly born from discussions (gravity models,
Bell tests, field visualizations). Named for the clockwork solar-system models.

## Quickstart

```sh
lab/tools/serve.sh                # static server on :8000
open http://localhost:8000/lab/gallery/    # browsable catalog of all sims

uv run lab/sims/vortex-bell/vortex_bell.py   # python sims run through uv
```

Prose lives at the top level (`theory/`, `studies/`); everything that runs
lives under `lab/`.

## Add a sim

```sh
lab/tools/new-sim.sh my-sim --kind web --title "My Sim"
```

See [CLAUDE.md](CLAUDE.md) for the sim contract (metadata, provenance,
versioning) and repo conventions.
