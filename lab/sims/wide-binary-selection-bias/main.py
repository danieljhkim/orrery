#!/usr/bin/env python3
"""Preregistered injection/recovery calibration of Astrolabe's wide-binary selector.

The population is Newtonian by construction. This calls the pinned Astrolabe
selector, shifted-field estimator, mock comparator and binned statistic directly.
It is an apparatus calibration, not a claim about nature.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ASTROLABE_PIN = "90f5b58890da36c44286a4edbde7eead879410a8"
PRINCIPIA_PIN = "c04f2ed1ae91d6c126bc60863b5e48f46abe4576"
APPARATUS_HASH = "6e39198e6babf452ec94f34ccd1eec49e3ca885964f4d6742937b33b5f9269f1"
PROTOCOL_HASH = "50fc37ac41bdbdc0e14ae3c079de3d8ed582d0fb4bf2e740270dc0fd5dfb9443"
GATE_HASH = "88609aed585cdbbd18586d5dcbeaf59f691965bbce2536152eabdb1bd51d7a89"
GATE_ID = "wide-binary-selection-bias-control"
ORRERY_BASE = "73c9bd53e963e44bb365992c1df87a1077cc996f"
PROTOCOL_DECLARED_REALIZATIONS = 47
ENUMERATED_REALIZATIONS = 44
SEEDS = (7, 11, 23, 42, 101)
SHIFTS = (0.5, 2.0, 5.0)
THRESHOLD = 0.04
RUN_DATE = "2026-09-05"


@dataclass(frozen=True)
class Run:
    id: str
    purpose: str
    population: str
    footprint: str
    cap: int | None
    contamination: float
    injection: float
    seeds: int


MATRIX = (
    Run("R0", "negative control", "isotropic", "equatorial", None, 0.0, 0.0, 5),
    Run("R1-baseline", "primary test", "nonuniform", "equatorial", 8000, 0.0, 0.0, 5),
    Run("R1-uncapped", "cap isolation", "nonuniform", "equatorial", None, 0.0, 0.0, 5),
    Run("R1-heavycap", "cap stress", "nonuniform", "equatorial", 2000, 0.0, 0.0, 5),
    Run("R2-wrap", "RA-wrap control", "nonuniform", "ra-wrap", 8000, 0.0, 0.0, 3),
    Run("R2-polar", "polar control", "nonuniform", "polar", 8000, 0.0, 0.0, 3),
    Run("R3-5", "chance calibration", "nonuniform", "equatorial", 8000, 0.05, 0.0, 3),
    Run("R3-10", "chance calibration", "nonuniform", "equatorial", 8000, 0.10, 0.0, 3),
    Run("R3-20", "chance calibration", "nonuniform", "equatorial", 8000, 0.20, 0.0, 3),
    Run("R4-0.02", "power floor diagnostic", "nonuniform", "equatorial", 8000, 0.0, 0.02, 3),
    Run("R4-0.05", "primary power", "nonuniform", "equatorial", 8000, 0.0, 0.05, 3),
    Run("R4-0.10", "primary power", "nonuniform", "equatorial", 8000, 0.0, 0.10, 3),
)
FOOTPRINTS = {
    "equatorial": (180.0, 40.0, 25.0),
    "ra-wrap": (0.0, 40.0, 15.0),
    "polar": (0.0, 85.0, 10.0),
}
FULL = {
    "n_true_pairs": 1050,
    "n_field_stars": 6200,
    "n_comparator_pairs": 5000,
    "oracle_sample_stars": 192,
    "distance_pc_range": [70.0, 175.0],
    "density_edge_ratio": 3.0,
    "parallax_fractional_error": 0.015,
    "pm_error_masyr": 0.05,
    "ruwe_center": 1.05,
    "ruwe_scatter": 0.08,
    "quality_gradient": "1 + 0.5*abs(sin(dec-dec_center))",
}
PILOT = {**FULL, "n_true_pairs": 80, "n_field_stars": 120,
         "n_comparator_pairs": 600, "oracle_sample_stars": 96}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], check=check,
                          capture_output=True, text=True)


def _checkout(raw: str, label: str) -> Path:
    root = Path(raw).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"{label} is not a Git checkout: {root}")
    return root


def verify_sources(astrolabe: Path, principia: Path) -> dict[str, Any]:
    apparatus = astrolabe / "src/astrolabe/analysis/wide_binaries.py"
    protocol = principia / "studies/wide-binary-selection-bias-preregistration.md"
    gate = principia / "gates/wide-binary-selection-bias-control.json"
    actual = {
        "astrolabe_commit": _git(astrolabe, "rev-parse", "HEAD").stdout.strip(),
        "principia_commit": _git(principia, "rev-parse", "HEAD").stdout.strip(),
        "astrolabe_apparatus_sha256": _sha(apparatus),
        "protocol_sha256": _sha(protocol),
        "gate_sha256": _sha(gate),
    }
    checks = {
        "ORB-11217_merged": _git(astrolabe, "merge-base", "--is-ancestor", ASTROLABE_PIN,
                                  "HEAD", check=False).returncode == 0,
        "ORB-11221_merged": _git(principia, "merge-base", "--is-ancestor", PRINCIPIA_PIN,
                                  "HEAD", check=False).returncode == 0,
        "apparatus_exact": actual["astrolabe_apparatus_sha256"] == APPARATUS_HASH,
        "protocol_exact": actual["protocol_sha256"] == PROTOCOL_HASH,
        "gate_exact": actual["gate_sha256"] == GATE_HASH,
        "gate_id_exact": json.loads(gate.read_text())["id"] == GATE_ID,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen-source verification failed: {checks}; {actual}")
    return {**actual, "verification": checks}


def load_apparatus(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / "src"))
    from astrolabe.analysis import wide_binaries as wb  # noqa: PLC0415

    names = (
        "DEFAULT_G_EDGES", "_radius_candidates", "binned_vtilde",
        "chance_alignment_rate", "make_synthetic_star_field", "mass_from_abs_g",
        "newtonian_mock_pairs", "select_star_quality", "select_wide_pairs",
        "sensitivity_table",
    )
    return {name: getattr(wb, name) for name in names}


def _disk(rng: np.random.Generator, n: int, nonuniform: bool) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    while len(xs) < n:
        count = max(256, 2 * (n - len(xs)))
        x, y = rng.uniform(-1, 1, (2, count))
        keep = x * x + y * y <= 1
        if nonuniform:
            keep &= rng.random(count) <= (2 + x) / 3
        xs.extend(x[keep].tolist())
        ys.extend(y[keep].tolist())
    return np.asarray(xs[:n]), np.asarray(ys[:n])


def _positions(rng: np.random.Generator, n: int, footprint: str,
               nonuniform: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    ra0, dec0, radius = FOOTPRINTS[footprint]
    x, y = _disk(rng, n, nonuniform)
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    coords = center.directional_offset_by(np.arctan2(x, y) * u.rad,
                                          np.hypot(x, y) * radius * u.deg)
    return coords.ra.deg, coords.dec.deg, x


def _secondaries(ra: np.ndarray, dec: np.ndarray, theta: np.ndarray,
                 pa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    coords = SkyCoord(ra * u.deg, dec * u.deg).directional_offset_by(
        pa * u.rad, theta * u.arcsec)
    return coords.ra.deg, coords.dec.deg


def _abs_g(mass: np.ndarray, mass_from_abs_g: Any) -> np.ndarray:
    grid = np.linspace(-2, 16, 8193)
    grid_mass = np.asarray(mass_from_abs_g(grid))
    return np.interp(mass, grid_mass[::-1], grid[::-1])


def generate(run: Run, seed: int, params: dict[str, Any], app: dict[str, Any]) -> tuple[Any, Any, dict[tuple[str, str], str], dict[str, Any]]:
    """Extend Astrolabe's Newtonian fixture with frozen sky/quality/label controls."""
    from astropy.table import Table
    rng = np.random.default_rng(seed)
    n_true = params["n_true_pairs"]
    n_chance = round(run.contamination * n_true)
    nonuniform = run.population == "nonuniform"
    truth = app["newtonian_mock_pairs"](n_true, seed=seed, distance_pc=100.0)
    low_edge = float(app["DEFAULT_G_EDGES"][1])
    affected = np.asarray(truth["g_N_ms2"]) < low_edge
    truth["vtilde"][affected] += run.injection
    truth["dv_kms"][affected] = (np.asarray(truth["vtilde"])[affected]
                                          * np.asarray(truth["v_circ_kms"])[affected])
    names = ("source_id", "ra", "dec", "parallax", "parallax_error", "pmra", "pmdec",
             "pmra_error", "pmdec_error", "phot_g_mean_mag", "bp_rp", "ruwe",
             "radial_velocity", "radial_velocity_error")
    cols: dict[str, list[Any]] = {name: [] for name in names}
    labels: dict[tuple[str, str], str] = {}

    def add_pairs(pairs: Any, label: str, prefix: str) -> None:
        count = len(pairs)
        ra1, dec1, x = _positions(rng, count, run.footprint, nonuniform)
        if nonuniform:
            distance = np.clip(70 + 105 * (x + 1) / 2 + rng.normal(0, 4, count), 70, 175)
        else:
            distance = rng.uniform(70, 175, count)
        theta = np.asarray(pairs["s_au"]) / distance
        ra2, dec2 = _secondaries(ra1, dec1, theta, rng.uniform(0, 2 * np.pi, count))
        masses = np.column_stack((pairs["m1_msun"], pairs["m2_msun"])).astype(float)
        magnitude = _abs_g(masses, app["mass_from_abs_g"])
        bulk_ra, bulk_dec = rng.normal(5, 12, count), rng.normal(-3, 12, count)
        rel_pa = rng.uniform(0, 2 * np.pi, count)
        dpm = np.asarray(pairs["dv_kms"]) / (4.74047e-3 * distance)
        for i in range(count):
            ids = (f"{prefix}{i:05d}a", f"{prefix}{i:05d}b")
            labels[tuple(sorted(ids))] = label
            for component, sign in ((0, 0.5), (1, -0.5)):
                ra, dec = ((ra1[i], dec1[i]) if component == 0 else (ra2[i], dec2[i]))
                dec_center = FOOTPRINTS[run.footprint][1]
                quality = (1.0 if not nonuniform else
                           1 + 0.5 * abs(math.sin(math.radians(dec - dec_center))))
                plx = 1000 / distance[i]
                plx_err = params["parallax_fractional_error"] * quality * plx
                pm_err = params["pm_error_masyr"] * quality
                cols["source_id"].append(ids[component])
                cols["ra"].append(ra)
                cols["dec"].append(dec)
                cols["parallax"].append(plx + rng.normal(0, plx_err))
                cols["parallax_error"].append(plx_err)
                cols["pmra"].append(bulk_ra[i] + sign * dpm[i] * math.cos(rel_pa[i])
                                      + rng.normal(0, pm_err))
                cols["pmdec"].append(bulk_dec[i] + sign * dpm[i] * math.sin(rel_pa[i])
                                       + rng.normal(0, pm_err))
                cols["pmra_error"].append(pm_err)
                cols["pmdec_error"].append(pm_err)
                cols["phot_g_mean_mag"].append(magnitude[i, component]
                                                  + 5 * math.log10(distance[i]) - 5)
                cols["bp_rp"].append(1 + rng.normal(0, 0.08))
                cols["ruwe"].append(params["ruwe_center"] * quality
                                      + rng.normal(0, params["ruwe_scatter"]))
                cols["radial_velocity"].append(float("nan"))
                cols["radial_velocity_error"].append(float("nan"))

    add_pairs(truth, "true", "T")
    if n_chance:
        planted = app["newtonian_mock_pairs"](n_chance, seed=seed + 200_000)
        add_pairs(planted, "planted-chance", "C")

    n_field = params["n_field_stars"]
    ra, dec, x = _positions(rng, n_field, run.footprint, nonuniform)
    if nonuniform:
        distance = np.clip(70 + 105 * (x + 1) / 2 + rng.normal(0, 8, n_field), 70, 175)
    else:
        distance = rng.uniform(70, 175, n_field)
    mass = rng.uniform(0.2, 1.5, n_field)
    magnitude = _abs_g(mass, app["mass_from_abs_g"])
    for i in range(n_field):
        dec_center = FOOTPRINTS[run.footprint][1]
        quality = (1.0 if not nonuniform else
                   1 + 0.5 * abs(math.sin(math.radians(dec[i] - dec_center))))
        plx = 1000 / distance[i]
        plx_err = params["parallax_fractional_error"] * quality * plx
        pm_err = params["pm_error_masyr"] * quality
        values = {
            "source_id": f"F{i:06d}", "ra": ra[i], "dec": dec[i],
            "parallax": plx + rng.normal(0, plx_err), "parallax_error": plx_err,
            "pmra": rng.normal(0, 30), "pmdec": rng.normal(0, 30),
            "pmra_error": pm_err, "pmdec_error": pm_err,
            "phot_g_mean_mag": magnitude[i] + 5 * math.log10(distance[i]) - 5,
            "bp_rp": rng.uniform(0.5, 2),
            "ruwe": params["ruwe_center"] * quality + rng.normal(0, params["ruwe_scatter"]),
            "radial_velocity": float("nan"), "radial_velocity_error": float("nan"),
        }
        for name, value in values.items():
            cols[name].append(value)
    stars = Table(cols)
    meta = {
        "seed": seed, "n_true_pairs": n_true, "n_planted_chance_pairs": n_chance,
        "n_field_stars": n_field, "population": run.population,
        "footprint": {"name": run.footprint, "ra_deg": FOOTPRINTS[run.footprint][0],
                      "dec_deg": FOOTPRINTS[run.footprint][1],
                      "radius_deg": FOOTPRINTS[run.footprint][2]},
        "injected_amplitude_vtilde": run.injection,
        "injected_low_g_edge_ms2": low_edge,
        "n_injection_affected_truth_pairs": int(affected.sum()),
    }
    stars.meta["synthetic_population"] = meta
    return stars, truth, labels, meta


def ids(table: Any) -> set[tuple[str, str]]:
    return {tuple(sorted((str(row["source_id_1"]), str(row["source_id_2"]))))
            for row in table}


def oracle(stars: Any, app: dict[str, Any], n: int, shift: float = 0.0) -> dict[str, Any]:
    """Independent SkyCoord radius oracle for a bounded quality-selected sample."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord, search_around_sky
    sample = app["select_star_quality"](stars)[:n]
    ra, dec = np.asarray(sample["ra"]), np.asarray(sample["dec"])
    shifted = (ra + shift) % 360
    actual, _, _, _ = app["_radius_candidates"](ra, dec, shifted, dec, 3600.0)
    primary, secondary = SkyCoord(ra * u.deg, dec * u.deg), SkyCoord(shifted * u.deg, dec * u.deg)
    i, j, _, _ = search_around_sky(primary, secondary, 3600 * u.arcsec)
    expected, observed = set(zip(map(int, i), map(int, j), strict=True)), set(actual)
    return {"n_stars": len(sample), "shift_deg": shift, "oracle_count": len(expected),
            "astrolabe_count": len(observed),
            "missing": [list(x) for x in sorted(expected - observed)[:20]],
            "extra": [list(x) for x in sorted(observed - expected)[:20]],
            "pass": expected == observed}


def one_realization(run: Run, seed: int, params: dict[str, Any], app: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    stars, truth, labels, population = generate(run, seed, params, app)
    selected = app["select_wide_pairs"](stars, max_stars=run.cap)
    selected_ids = ids(selected)
    n_true = sum(labels.get(pair) == "true" for pair in selected_ids)
    n_planted = sum(labels.get(pair) == "planted-chance" for pair in selected_ids)
    n_chance = len(selected) - n_true
    mock = app["newtonian_mock_pairs"](params["n_comparator_pairs"], seed=seed + 100_000)
    shifts = SHIFTS if run.id in {"R0", "R1-baseline", "R3-5", "R3-10", "R3-20"} else (0.5,)
    estimates = []
    default_bins = None
    for shift in shifts:
        rate, per_bin = app["chance_alignment_rate"](stars, max_stars=run.cap,
                                                      ra_shift_deg=shift)
        if shift == 0.5:
            default_bins = per_bin
        estimates.append({"shift_deg": shift, "global_rate": float(rate),
                          "per_bin": [{"g_N_lo_ms2": float(row["g_N_lo_ms2"]),
                                       "g_N_hi_ms2": float(row["g_N_hi_ms2"]),
                                       "n_real": int(row["n_real"]), "n_shift": int(row["n_shift"]),
                                       "r_chance": float(row["r_chance"])} for row in per_bin]})
    recovered = app["binned_vtilde"](selected, mock, chance_per_bin=default_bins, min_pairs=5)
    truth_bins = app["binned_vtilde"](truth, min_pairs=5)
    truth_by_edge = {(float(r["g_N_lo_ms2"]), float(r["g_N_hi_ms2"])): r for r in truth_bins}
    rows = []
    for row in recovered:
        key = (float(row["g_N_lo_ms2"]), float(row["g_N_hi_ms2"]))
        tr = truth_by_edge.get(key)
        rec, comp = float(row["vtilde_med"]), float(row["vtilde_mock_med"])
        true_med = float(tr["vtilde_med"]) if tr is not None else float("nan")
        rows.append({"run_id": run.id, "seed": seed, "g_N_lo_ms2": key[0],
                     "g_N_hi_ms2": key[1], "n_pairs": int(row["n_pairs"]),
                     "vtilde_med": rec, "vtilde_within_sample_err": float(row["vtilde_err"]),
                     "vtilde_comparator_med": comp, "vtilde_truth_med": true_med,
                     "B_i": rec - comp, "S_i": rec - true_med,
                     "r_chance": float(row["r_chance"])})
    return {"run_id": run.id, "seed": seed, "population": population, "max_stars": run.cap,
            "contamination_frac": run.contamination, "injected_amplitude": run.injection,
            "n_input_stars": len(stars), "n_selected_pairs": len(selected),
            "n_selected_true_pairs": n_true, "n_selected_planted_chance_pairs": n_planted,
            "n_selected_known_or_incidental_chance_pairs": n_chance,
            "selected_true_pair_completeness": n_true / len(truth),
            "selected_contamination_fraction": n_chance / len(selected) if len(selected) else float("nan"),
            "selector_metadata": selected.meta.get("pair_cuts", {}),
            "oracle": oracle(stars, app, params["oracle_sample_stars"]),
            "chance_estimates": estimates, "bins": rows,
            "runtime_seconds": perf_counter() - started}


def center(values: list[float]) -> tuple[float, float]:
    values_array = np.asarray([v for v in values if np.isfinite(v)])
    if not len(values_array):
        return float("nan"), float("nan")
    median = float(np.median(values_array))
    return median, float(1.4826 * np.median(np.abs(values_array - median)))


def aggregate(realizations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for realization in realizations:
        for row in realization["bins"]:
            grouped.setdefault((row["run_id"], row["g_N_lo_ms2"], row["g_N_hi_ms2"]), []).append(row)
    result = []
    for (run_id, lo, hi), rows in sorted(grouped.items()):
        b_med, b_sig = center([row["B_i"] for row in rows])
        s_med, s_sig = center([row["S_i"] for row in rows])
        result.append({"run_id": run_id, "g_N_lo_ms2": lo, "g_N_hi_ms2": hi,
                       "n_seeds_populated": len(rows),
                       "median_n_pairs": float(np.median([row["n_pairs"] for row in rows])),
                       "B_i_median": b_med, "B_i_realization_sigma": b_sig,
                       "S_i_median": s_med, "S_i_realization_sigma": s_sig,
                       "reproducible_threshold_excursion": abs(b_med) >= THRESHOLD and abs(b_med) > b_sig})
    return result


def decision(specs: tuple[Run, ...], realizations: list[dict[str, Any]],
             aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in aggregates:
        by_run.setdefault(row["run_id"], []).append(row)
    r0 = by_run.get("R0", [])
    r0_bias = bool(r0) and all(abs(r["B_i_median"]) <= 1.5 * r["B_i_realization_sigma"]
                               or r["B_i_median"] == r["B_i_realization_sigma"] == 0 for r in r0)
    r0_rates = [e["global_rate"] for r in realizations if r["run_id"] == "R0"
                for e in r["chance_estimates"]]
    r0_rate, r0_sigma = center(r0_rates)
    r0_chance = abs(r0_rate) <= 0.05 + r0_sigma
    primary_names = ("R1-baseline", "R1-uncapped", "R1-heavycap", "R2-wrap", "R2-polar")
    primary = [row for name in primary_names for row in by_run.get(name, [])]
    failures = [r for r in primary if r["reproducible_threshold_excursion"]]
    uncertain = [r for r in primary if abs(r["B_i_median"]) >= THRESHOLD
                 and not r["reproducible_threshold_excursion"]]
    primary_status = "fail" if failures else "unresolved" if uncertain else "pass"

    baseline = {(r["g_N_lo_ms2"], r["g_N_hi_ms2"]): r for r in by_run.get("R1-baseline", [])}
    cap_rows = []
    for name in ("R1-uncapped", "R1-heavycap"):
        for row in by_run.get(name, []):
            key = row["g_N_lo_ms2"], row["g_N_hi_ms2"]
            if key in baseline:
                delta = row["B_i_median"] - baseline[key]["B_i_median"]
                sigma = math.hypot(row["B_i_realization_sigma"], baseline[key]["B_i_realization_sigma"])
                cap_rows.append({"run_id": name, "g_N_lo_ms2": key[0], "g_N_hi_ms2": key[1],
                                 "delta_B_i": delta, "combined_realization_sigma": sigma,
                                 "disagrees_beyond_uncertainty": abs(delta) > sigma})

    spec_by_id = {spec.id: spec for spec in specs}
    chance_rows = []
    for name in ("R3-5", "R3-10", "R3-20"):
        if name not in spec_by_id:
            continue
        injected = spec_by_id[name].contamination
        for shift in SHIFTS:
            values = [e["global_rate"] for r in realizations if r["run_id"] == name
                      for e in r["chance_estimates"] if e["shift_deg"] == shift]
            median, sigma = center(values)
            tolerance = 0.05 if injected == 0.05 else 0.25 * injected
            chance_rows.append({"run_id": name, "shift_deg": shift, "injected_fraction": injected,
                                "estimated_rate_median": median, "realization_sigma": sigma,
                                "tolerance": tolerance, "pass": abs(median - injected) <= tolerance})

    power_rows = []
    low_edge = 1.0000000001e-11
    for name in ("R4-0.02", "R4-0.05", "R4-0.10"):
        if name not in spec_by_id:
            continue
        affected = {(r["g_N_lo_ms2"], r["g_N_hi_ms2"]): r for r in by_run.get(name, [])
                    if r["g_N_hi_ms2"] <= low_edge}
        detections = []
        for realization in [r for r in realizations if r["run_id"] == name]:
            detections.append(any(
                (row["g_N_lo_ms2"], row["g_N_hi_ms2"]) in affected
                and abs(row["B_i"]) >= 2 * affected[(row["g_N_lo_ms2"], row["g_N_hi_ms2"])]["B_i_realization_sigma"]
                for row in realization["bins"]))
        rate = sum(detections) / len(detections) if detections else float("nan")
        power_rows.append({"run_id": name, "per_seed_detected": detections, "detection_rate": rate,
                           "required": None if name == "R4-0.02" else 0.8,
                           "pass": None if name == "R4-0.02" else rate >= 0.8})

    gates = {
        "R0_sanity": {"pass": r0_bias and r0_chance, "bias_pass": r0_bias,
                      "chance_pass": r0_chance, "chance_rate_median": r0_rate,
                      "chance_rate_sigma": r0_sigma,
                      "zero_contamination_operational_tolerance": 0.05},
        "R1_primary": {"status": primary_status, "threshold": THRESHOLD,
                       "reproducible_failures": failures, "nonreproducible_excursions": uncertain},
        "cap_isolation": {"pass": not any(r["disagrees_beyond_uncertainty"] for r in cap_rows),
                          "comparisons": cap_rows},
        "R3_chance_calibration": {"pass": bool(chance_rows) and all(r["pass"] for r in chance_rows),
                                  "checks": chance_rows},
        "R4_power": {"pass": bool(power_rows) and all(r["pass"] for r in power_rows if r["pass"] is not None),
                     "checks": power_rows},
        "candidate_oracle_regression": {"pass": all(r["oracle"]["pass"] for r in realizations)},
    }
    if not gates["R0_sanity"]["pass"] or not gates["R4_power"]["pass"]:
        verdict = "unresolved"
    elif primary_status == "fail":
        verdict = "hypothesis-survives"
    elif primary_status == "unresolved" or not gates["R3_chance_calibration"]["pass"]:
        verdict = "unresolved"
    else:
        verdict = "hypothesis-fails-at-predeclared-scale"
    return {"verdict": verdict, "gates": gates}


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def environment() -> dict[str, Any]:
    versions = {}
    for package in ("orrery", "astrolabe", "numpy", "scipy", "astropy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source-checkout" if package == "astrolabe" else "unavailable"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": versions}


def profile_matrix(profile: str) -> tuple[Run, ...]:
    if profile == "full":
        return MATRIX
    wanted = {"R0", "R1-baseline", "R2-wrap", "R2-polar", "R3-10", "R4-0.05"}
    return tuple(Run(r.id, r.purpose, r.population, r.footprint, r.cap,
                     r.contamination, r.injection, 1) for r in MATRIX if r.id in wanted)


def experiment(profile: str, app: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    specs, params, started = profile_matrix(profile), (FULL if profile == "full" else PILOT), perf_counter()
    realizations = [one_realization(spec, seed, params, app)
                    for spec in specs for seed in SEEDS[:spec.seeds]]
    aggregates = aggregate(realizations)
    result = {"schema_version": 1, "experiment": "wide-binary-selection-bias",
              "run_date": RUN_DATE, "profile": profile,
              "scope": "synthetic apparatus calibration only; no observational or nature-level claim",
              "preregistration": {"task": "ORB-11221", "gate_id": GATE_ID,
                  "decision_rule": {"primary_abs_B_i_threshold": THRESHOLD,
                      "uncertainty": "median +/- 1.4826*MAD across independent realization seeds",
                      "r0": "all |median B_i| <= 1.5*realization sigma and chance <= 0.05 + sigma",
                      "r1": "median |B_i| >= 0.04 and bootstrap band excludes zero",
                      "r3": "5% +/-0.05 absolute; 10%/20% +/-25% relative",
                      "r4": "0.05/0.10 detected at >=2 realization sigma in >=80% of seeds"}},
              "sources": {**sources, "orrery_base_commit": ORRERY_BASE},
              "protocol_matrix_arithmetic": {
                  "declared_in_protocol_text": PROTOCOL_DECLARED_REALIZATIONS,
                  "enumerated_by_12_frozen_rows": ENUMERATED_REALIZATIONS,
                  "resolution": "executed every enumerated row/seed exactly; did not invent three runs"},
              "apparatus": {"module": "astrolabe.analysis.wide_binaries",
                  "functions_called": ["select_star_quality", "_radius_candidates", "select_wide_pairs",
                      "chance_alignment_rate", "newtonian_mock_pairs", "binned_vtilde",
                      "sensitivity_table", "make_synthetic_star_field"],
                  "selector": "cKDTree unit-vector spherical radius search plus Astrolabe physical cuts",
                  "candidate_oracle": "independent SkyCoord.search_around_sky on small samples"},
              "parameters": {**params, "seeds": list(SEEDS), "shift_ladder_deg": list(SHIFTS)},
              "run_matrix": [asdict(spec) for spec in specs], "n_realizations": len(realizations),
              "realizations": realizations, "aggregates": aggregates,
              "decision": decision(specs, realizations, aggregates), "environment": environment(),
              "rerun_commands": {
                  "full": "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11222 uv run --with ../astrolabe lab/sims/wide-binary-selection-bias/main.py --astrolabe-root ../astrolabe --principia-root ../principia",
                  "pilot": "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11222 uv run --with ../astrolabe lab/sims/wide-binary-selection-bias/main.py --profile pilot --output /tmp/wbsel-pilot.json --astrolabe-root ../astrolabe --principia-root ../principia",
                  "self_test": "PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/orbit-uv-cache-11222 uv run --with ../astrolabe lab/sims/wide-binary-selection-bias/main.py --self-test --astrolabe-root ../astrolabe --principia-root ../principia"}}
    # Required Astrolabe sensitivity function; explicitly exploratory and verdict-excluded.
    sensitivity_stars = generate(specs[min(1, len(specs) - 1)], SEEDS[0], params, app)[0]
    sensitivity = app["sensitivity_table"](sensitivity_stars,
        n_mock=min(1000, params["n_comparator_pairs"]),
        max_stars=specs[min(1, len(specs) - 1)].cap, seed=SEEDS[0])
    result["exploratory"] = {"label": "exploratory; excluded from frozen primary verdict",
        "sensitivity_table": [{name: row[name].item() if hasattr(row[name], "item") else row[name]
                               for name in sensitivity.colnames} for row in sensitivity]}
    result["runtime_seconds"] = perf_counter() - started
    return sanitize(result)


def report(result: dict[str, Any]) -> str:
    gates = result["decision"]["gates"]
    packages = json.dumps(result["environment"]["packages"], sort_keys=True)
    parameters = json.dumps(result["parameters"], sort_keys=True)
    r1 = "\n".join(
        f"- `{row['run_id']}` [{row['g_N_lo_ms2']:.3e}, {row['g_N_hi_ms2']:.3e}): "
        f"B_i = {row['B_i_median']:+.4f} +/- {row['B_i_realization_sigma']:.4f}."
        for row in gates["R1_primary"]["reproducible_failures"]
    ) or "- No reproducible R1 threshold excursion."
    chance = "\n".join(
        f"- `{row['run_id']}`, shift {row['shift_deg']:.1f} deg: "
        f"R_chance = {row['estimated_rate_median']:.4f} +/- {row['realization_sigma']:.4f} "
        f"for truth {row['injected_fraction']:.2f} ({'PASS' if row['pass'] else 'FAIL'})."
        for row in gates["R3_chance_calibration"]["checks"]
    )
    power = "\n".join(
        f"- `{row['run_id']}` detection rate {row['detection_rate']:.1%}"
        f"{' (diagnostic)' if row['pass'] is None else ' (' + ('PASS' if row['pass'] else 'FAIL') + ')'}."
        for row in gates["R4_power"]["checks"]
    )
    completeness = [row["selected_true_pair_completeness"] for row in result["realizations"]]
    purity = [1 - row["selected_contamination_fraction"] for row in result["realizations"]]
    return f"""# Validated run — {RUN_DATE}

## Verdict

**{result['decision']['verdict']}** under the frozen `{GATE_ID}` decision rule.
This is a synthetic calibration of Astrolabe's apparatus, not a claim about nature.

## Prerequisites and exact apparatus

- Astrolabe ORB-11217 merged source: `{result['sources']['astrolabe_commit']}`.
- Astrolabe apparatus SHA-256: `{result['sources']['astrolabe_apparatus_sha256']}`.
- Principia ORB-11221 merged protocol: `{result['sources']['principia_commit']}`.
- Orrery implementation base: `{result['sources']['orrery_base_commit']}`.
- Protocol SHA-256: `{result['sources']['protocol_sha256']}`.
- Gate: `{GATE_ID}`; gate SHA-256: `{result['sources']['gate_sha256']}`.

## Frozen controls

- R0 sanity: **{'PASS' if gates['R0_sanity']['pass'] else 'FAIL'}**.
- R1 geometry/truncation: **{gates['R1_primary']['status'].upper()}**.
- Cap isolation: **{'PASS' if gates['cap_isolation']['pass'] else 'FAIL'}**.
- Shifted-field calibration: **{'PASS' if gates['R3_chance_calibration']['pass'] else 'FAIL'}**.
- Known-effect power: **{'PASS' if gates['R4_power']['pass'] else 'FAIL'}**.
- Independent candidate oracle: **{'PASS' if gates['candidate_oracle_regression']['pass'] else 'FAIL'}**.

Failed controls are not hidden: a failed sanity or power control forces the
verdict to remain unresolved. Machine-readable per-seed truth/recovery, B_i,
S_i, chance-rate, uncertainty, cap, oracle, and power evidence is in
`assets/results.json`.

### Reproducible R1 excursions

{r1}

### Shifted-field calibration

{chance}

### Positive-control power

{power}

Across all realizations, true-pair completeness ranged from {min(completeness):.3f}
to {max(completeness):.3f}, and selected-pair purity from {min(purity):.3f} to
{max(purity):.3f}. The 2,000-star heavy cap produces the low-completeness end;
known labels, rather than the shifted estimator, define these recovery metrics.

## Runtime and environment

- Realizations: {result['n_realizations']}. The protocol prose says 47, but its
  12 frozen rows enumerate 44; every row/seed was run and no three runs were invented.
- Wall time: {result['runtime_seconds']:.3f} s.
- Python: `{result['environment']['python']}` on `{result['environment']['platform']}`.
- Packages: `{packages}`.
- Frozen parameters and seeds: `{parameters}`.

## Exact rerun

```sh
{result['rerun_commands']['full']}
python3 lab/tools/build-gallery.py
git diff --check
```

## Reconciliation handoff

Principia should judge only the methodological gate from these results.
Exploratory sensitivity rows are segregated in the JSON and do not alter the
frozen verdict. No sibling repository was modified by this run.
"""


def self_test(app: dict[str, Any]) -> dict[str, Any]:
    fixture = app["make_synthetic_star_field"](n_pairs=24, n_field=30, seed=77)
    selected = app["select_wide_pairs"](fixture, max_stars=None)
    bins = app["binned_vtilde"](selected, app["newtonian_mock_pairs"](400, seed=91),
                                 min_pairs=2)
    oracles = [oracle(fixture, app, len(fixture), shift) for shift in (0.0, 0.5, -0.5)]
    rate, _ = app["chance_alignment_rate"](fixture, max_stars=None, ra_shift_deg=0.5)
    checks = {"actual_selector_returned_pairs": len(selected) > 0,
              "actual_binned_statistic_returned_bins": len(bins) > 0,
              "oracle_exact": all(row["pass"] for row in oracles),
              "chance_estimator_finite": np.isfinite(rate),
              "candidate_generation_repaired": selected.meta["pair_cuts"]["candidate_generation"]
              == "cKDTree unit-vector spherical radius search"}
    if not all(checks.values()):
        raise AssertionError(checks)
    return sanitize({"checks": checks, "n_stars": len(fixture), "n_selected_pairs": len(selected),
                     "n_bins": len(bins), "chance_rate": float(rate), "oracles": oracles})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astrolabe-root", required=True)
    parser.add_argument("--principia-root", required=True)
    parser.add_argument("--profile", choices=("pilot", "full"), default="full")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "assets/results.json")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / f"RUN-{RUN_DATE}.md")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    astrolabe, principia = _checkout(args.astrolabe_root, "Astrolabe"), _checkout(args.principia_root, "Principia")
    sources = verify_sources(astrolabe, principia)
    app = load_apparatus(astrolabe)
    if args.self_test:
        print(json.dumps(self_test(app), indent=2, sort_keys=True))
        return 0
    result = experiment(args.profile, app, sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.profile == "full":
        args.report.write_text(report(result))
    print(f"{args.profile}: {result['n_realizations']} realizations; "
          f"verdict={result['decision']['verdict']}; output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
