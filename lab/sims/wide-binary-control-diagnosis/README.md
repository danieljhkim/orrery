# Wide-binary control-failure diagnosis

ORB-11222 ran Principia's frozen ORB-11221 matrix against Astrolabe's ORB-11217
selector in the sibling sim [`wide-binary-selection-bias`](../wide-binary-selection-bias/)
and returned `unresolved`: R0 sanity, cap isolation, R3 shifted-field
calibration and R4 power all failed, while the independent candidate oracle
passed all 44 enumerated realizations. This sim asks one question per failed
control — **is the failure a defect of the synthetic fixture and its decision
layer, or a faithful limitation of the apparatus and the method?**

It does not retune anything. The frozen thresholds, seed matrix, shift ladder
and run rows are read from the ORB-11221 git blob and left as they are; the
ORB-11222 fixture and its `assets/results.json` are pinned by SHA-256 and
reanalysed rather than edited. That sim stays the frozen record of its run.
No sibling repository is written. This is apparatus diagnosis only: no
observational claim, no claim about gravity.

## What it does

Two kinds of evidence, kept separate in the output:

- **Reanalysis** of ORB-11222's committed per-seed evidence (D2, D4, and the
  committed half of D3 and D5). No rerun needed; the numbers are the ones the
  frozen run already published.
- **Live probes** (D1, and the live half of D3 and D5) that import the frozen
  fixture's own `generate()` and call the pinned Astrolabe functions, so the
  populations are identical to the frozen run by construction. Probe ladders
  are diagnostic instruments — they are never compared to a frozen threshold
  and never enter the frozen verdict.

| | question |
|---|---|
| D1 | what do the frozen gates do at n=3–5 seeds when there is genuinely nothing to detect? |
| D2 | how big is the zero-injection B_i floor, and what is it made of? |
| D3 | was the shifted-field estimator fed the contamination the protocol specified? |
| D4 | did the apparatus fail to recover the injection, or did the gate? |
| D5 | is the heavy-cap excursion the documented cap behaving as documented? |

`assets/results.json` carries the machine-readable evidence and a per-control
classification (`fixture-defect`, `protocol-arithmetic-defect`,
`faithful-apparatus-limitation`, `undetermined`). `RUN-2026-09-05.md` is the
readable verdict with exact revisions and hashes.

## Running it

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11241 \
  uv run --with ../astrolabe lab/sims/wide-binary-control-diagnosis/main.py \
  --astrolabe-root ../astrolabe --principia-root ../principia
```

`--self-test` runs a focused check; `--profile pilot --output /tmp/wbdiag.json`
runs the short matrix.

The script refuses checkouts whose apparatus, frozen protocol blob, frozen gate
blob, ORB-11222 fixture or ORB-11222 evidence hashes differ from the pins.

## A note on pinning

The ORB-11222 fixture pins the protocol and gate by *worktree file* hash. That
sim no longer runs against Principia's worktree: ORB-11234 appended a dated
outcome section to both files, so its `protocol_exact` and `gate_exact` checks
now refuse. This sim pins the same frozen terms by **git blob** at the frozen
ORB-11221 commit instead, so the diagnosis stays exactly reproducible without
asking Principia to revert prose. The frozen thresholds, seeds and matrix rows
are byte-identical across that revision; only narrative was added.

## Ownership

Principia owns whether the `wide-binary-selection-bias-control` gate's decision
layer should be revised — this repository does not amend a frozen protocol from
the outside. The one open question this diagnosis raises about Astrolabe's
shifted-field estimator calibration is routed to Astrolabe's own Orbit task and
is recorded as reproducible in magnitude but not diagnosed to a mechanism here.
A failed control is reported, never tuned away.
