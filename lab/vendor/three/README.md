# vendor/three

`three.min.js` — three.js **r128** (global build), MIT license, fetched from
`https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`.

Vendored so sims work offline and never rot when a CDN moves. All imported
sims were written against the r128 global-script API (`window.THREE`); load it
with:

```html
<script src="../../vendor/three/three.min.js"></script>
```

If a new sim needs a newer three.js, vendor the module build alongside this
one (e.g. `three-r1xx.module.js`) rather than replacing r128 — the old sims
depend on r128's API.
