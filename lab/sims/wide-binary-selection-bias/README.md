# Wide-binary selection-bias injection/recovery control

This offline fixture executes Principia's frozen ORB-11221 protocol against the
actual Astrolabe ORB-11217 implementation. It imports Astrolabe's repaired
spherical selector and its binned scaled-velocity, shifted-field, comparator,
and sensitivity functions; the independent Astropy candidate oracle is used
only as a regression control. The population is Newtonian by construction.

The committed `assets/results.json` contains one row per populated
run/seed/acceleration bin plus known-label recovery, chance-rate, candidate
oracle, uncertainty, cap, and power evidence. `RUN-2026-09-05.md` gives the
readable verdict. Exploratory sensitivity rows are excluded from the frozen
primary decision. Nothing here is an observational or nature-level claim.

Run from clean sibling checkouts:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11222 \
  uv run --with ../astrolabe lab/sims/wide-binary-selection-bias/main.py \
  --astrolabe-root ../astrolabe --principia-root ../principia
```

The script refuses checkouts that do not contain the merged prerequisite
commits or whose apparatus, protocol, or gate hashes differ from frozen inputs.
Use `--self-test` for a focused apparatus check or `--profile pilot --output
/tmp/wbsel-pilot.json` for the six-realization smoke matrix.

The nonuniform quality and density gradients are preregistered stress shapes,
not measured Gaia completeness laws. A failed control is reported, never tuned
away.

The protocol prose declares 47 realizations, but the seeds in its 12 frozen
matrix rows sum to 44. The fixture executes those 44 enumerated realizations
exactly and records the discrepancy; it does not invent three unregistered
runs.

## Diagnosis of the failed controls

This sim is the frozen record of its run and is not edited in light of later
work. The failed controls are diagnosed separately in
[`wide-binary-control-diagnosis`](../wide-binary-control-diagnosis/) (ORB-11241),
which pins this sim's `main.py` and `assets/results.json` by SHA-256 and
reanalyses them. Note that this sim pins the protocol and gate by worktree file
hash, so it no longer runs against Principia HEAD: ORB-11234 appended a dated
outcome section to both files. The diagnosis sim pins the same frozen terms by
git blob at the ORB-11221 commit instead.
