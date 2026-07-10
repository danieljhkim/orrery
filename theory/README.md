# theory/

Our own theories — one living document per line of inquiry, maintained by **kepler**
(`agentbase/kepler/memory`), the physicist agent. Almanac keeps the *discussion* (how an idea
unfolded); this directory keeps the *current state of the theory* and its standing against
evidence.

## The contract

Each doc carries frontmatter (`title`, `status`, `families`, `almanac`, `created`, `updated`)
and an **evidence ledger** — a table mapping each claim to its status and the sims/studies
that back it:

| Claim status | Meaning |
|---|---|
| `supported` | a sim or study backs it, linked in the row |
| `mixed` | evidence cuts both ways — the row says how |
| `untested` | stated but not yet simulated/sourced |
| `refuted` | evidence kills it — the row links what did |
| `conjecture` | not yet backed by anything; flagged pending a study note |

Doc-level `status`: `exploratory` (being built), `growing` (active work), `refuted`
(the central claim is dead — doc stays, see below), `resolved` (converged with established
physics; kept as a correspondence record).

## Rules of the house

- **Theory bends to evidence.** A sim or study that contradicts a claim changes the claim's
  status in the same commit that lands the evidence.
- **Refuted branches stay.** A dead line of inquiry keeps its doc, its status, and its
  refuting evidence. It is a result.
- **Cite or label conjecture.** Established-physics facts lean on a `../studies/` note with a
  real citation, or are marked `conjecture — to verify`.
