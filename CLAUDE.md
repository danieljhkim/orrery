# orrery

The constellation's cabinet of physics sims. Simulations born in discussions with
Daniel accumulate **here** — as first-class, cataloged sims — instead of being
scattered as one-shot files in almanac's `_attachments/` and `40-outputs/`.

An orrery is a hand-cranked clockwork model of the solar system: a collection of
working physical models you can play with. That's this repo.

## Layout

Prose at the top level, everything executable under `lab/`:

```
theory/             Our own theories — one living doc per line of inquiry, each
                    with an evidence ledger (claim → status → sims/studies).
                    Contract in theory/README.md.
studies/            Sourced notes on established physics (real citations only).
lab/
  sims/<slug>/      One directory per sim. Always contains sim.json (metadata).
    sim.json          slug, title, kind, entry, status, created, topics,
                      family?, summary, provenance
    index.html|*.py   the sim itself (entry point named in sim.json)
    variants/         optional: kept iterations that the entry superseded
    assets/           optional: images, captured outputs
  lib/web/          Shared ES modules for interactive sims — loop.js (fixed-dt
                    loop), panel.js (declarative controls), canvas2d.js (DPR
                    canvas + pan/zoom + HUD), integrators.js (leapfrog/RK4/
                    n-body), vec.js, style.css (house dark theme).
  lib/py/orrery/    Shared Python helpers (rng, unit_vectors, chsh_S).
  vendor/three/     Pinned three.js r128 (global build). No CDN references.
  templates/        Starter web + py sims used by lab/tools/new-sim.sh.
  tools/            new-sim.sh · build-gallery.py · serve.sh
  gallery/          index.html — generated catalog, never edit by hand.
```

## Stewardship

This repo is maintained by **kepler**, the constellation's physicist agent
(`agentbase/kepler/memory` — memory-only, runs one-shot via worker `agent_invoke`; its
AGENTS.md carries the invocation contract). Kepler enforces the sim contract, curates
`studies/`, and keeps `theory/` evidence ledgers consistent with the sims: when a sim or
study contradicts a theory claim, the claim's status changes in the same change-set. Refuted
theory branches keep their docs (`status: refuted`) — they are results, not embarrassments.

## The sim contract

- **One directory per sim**, kebab-case slug, with a `sim.json`. `kind` is
  `web` (self-contained HTML, no build step) or `py` (numpy script).
- **Web sims**: plain ES modules, import shared code relatively
  (`../../lib/web/loop.js`), three.js from `../../vendor/three/three.min.js`.
  **Never reference a CDN** — vendor new libs under `lab/vendor/`. Must run
  from a static file server (`lab/tools/serve.sh`), zero build step.
- **Py sims**: run via `uv run lab/sims/<slug>/<entry>` (uv resolves numpy/
  matplotlib and puts `lab/lib/py/orrery` on the path from pyproject.toml).
  Seed RNGs (`orrery.rng(42)`) so results are reproducible; the module
  docstring is the sim's abstract — model, question, result.
- **Provenance both ways**: `provenance.almanac` in sim.json points at the
  almanac discussion/project note (vault-relative path); the note should link
  back to `codebases/orrery/lab/sims/<slug>/`. New sims from discussions land here,
  not in almanac — almanac keeps the *writeup*, orrery keeps the *sim*.

## Adding a sim

```
lab/tools/new-sim.sh <slug> --kind web|py --title "Human Title"
```

then replace the template physics, fill in `sim.json` (topics, summary,
provenance), and run `python3 lab/tools/build-gallery.py` (new-sim.sh runs it once
for you). Check the result in a browser via `lab/tools/serve.sh` before committing.

## Evolving a sim

- **Same model, better version** → iterate in place; git history is the record.
  If an old iteration is worth keeping runnable, move it to `variants/`.
- **Different model / new question** → new slug, and set `family` in both
  sims' sim.json to group them (e.g. `gravity-as-scarcity`, `bell-tests`).
- A sim replaced by a better slug gets `"status": "superseded"`.

## Conventions

- Independent repo; default branch **agent-main**, commit directly (no PR gate).
  Registered in `operations/scripts/repos.tsv`.
- Imported legacy sims (`"status": "imported"`) predate lab/lib/web and are
  self-contained — fine to leave as-is; adopt lab/lib/web when touching them.
- Keep lab/lib/ small: extract a helper only when a second sim needs it.
