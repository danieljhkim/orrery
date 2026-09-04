"""Measure the imported frame-drag toy's shape against Lense--Thirring.

This deterministic fixture transcribes the three ``flow(r)`` branches from
``frame-drag-swirl/index.html`` and measures angular velocity falloff,
linearity in the toy's ``u0`` spin surrogate, sign reversal, and the absence
of any coupling to its ``GM`` mass parameter.  It tests a hand-set 2-D flow
law, not a rotating-source dynamics or a physical-unit prediction.

Usage:
    uv run lab/sims/frame-drag-swirl-lense-thirring-check/main.py
    uv run lab/sims/frame-drag-swirl-lense-thirring-check/main.py --check-determinism
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SLUG = "frame-drag-swirl-lense-thirring-check"
TASK_ID = "ORB-11172"
RUN_DATE = "2026-09-04"
ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "frame-drag-swirl" / "index.html"
RESULT_PATH = ROOT / "results.json"
RUN_PATH = ROOT / "runs" / f"{RUN_DATE}.json"

# Verbatim source fragment at frame-drag-swirl/index.html:35-39.  The runtime
# check makes source drift explicit rather than silently testing a new law.
SOURCE_LAW = """function flow(r){
  if(prof==='flat')return u0*Math.min(r/80,1);
  if(prof==='solid')return u0*r/RMAXV*1.6;
  return u0*80/Math.max(r,24);
}"""
SOURCE_OMEGA = """    var om=flow(r)/r;
    phases[i]=phases[i].map(function(ph){return ph+om*1.0;});"""

RMAXV = 380.0
ROUNDOFF_FLOOR = 32.0 * sys.float_info.epsilon


def verify_source() -> str:
    source = SOURCE_PATH.read_text()
    if SOURCE_LAW not in source or SOURCE_OMEGA not in source:
        raise RuntimeError(
            "frame-drag-swirl source law changed; update the transcription and provenance"
        )
    return hashlib.sha256(source.encode()).hexdigest()


def toy_flow(profile: str, radius: float, u0: float) -> float:
    """Literal Python transcription of the JavaScript ``flow(r)`` branches."""
    if profile == "flat":
        return u0 * min(radius / 80.0, 1.0)
    if profile == "solid":
        return u0 * radius / RMAXV * 1.6
    if profile == "whirl":
        return u0 * 80.0 / max(radius, 24.0)
    raise ValueError(f"unknown profile: {profile}")


def omega(profile: str, radius: float, u0: float) -> float:
    return toy_flow(profile, radius, u0) / radius


def geometric_ladder(start: float, stop: float, count: int) -> list[float]:
    ratio = stop / start
    return [start * ratio ** (index / (count - 1)) for index in range(count)]


def log_log_slope(xs: list[float], ys: list[float]) -> float:
    log_x = [math.log(value) for value in xs]
    log_y = [math.log(abs(value)) for value in ys]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in log_x)
    return numerator / denominator


def slope_error(xs: list[float], ys: list[float], fitted: float) -> float:
    local = [
        math.log(abs(ys[index + 1] / ys[index]))
        / math.log(xs[index + 1] / xs[index])
        for index in range(len(xs) - 1)
    ]
    return max(ROUNDOFF_FLOOR, *(abs(value - fitted) for value in local))


def falloff_check() -> dict:
    sample_counts = [5, 9, 17, 33]
    profile_results = []
    for profile in ("flat", "solid", "whirl"):
        ladder = []
        for count in sample_counts:
            radii = geometric_ladder(80.0, 320.0, count)
            values = [omega(profile, radius, 1.2) for radius in radii]
            exponent = log_log_slope(radii, values)
            ladder.append({"sample_count": count, "measured_exponent": exponent})
        fine_radii = geometric_ladder(80.0, 320.0, sample_counts[-1])
        fine_values = [omega(profile, radius, 1.2) for radius in fine_radii]
        exponent = ladder[-1]["measured_exponent"]
        error = max(
            ROUNDOFF_FLOOR,
            abs(exponent - ladder[-2]["measured_exponent"]),
            slope_error(fine_radii, fine_values, exponent),
        )
        matches = abs(exponent - (-3.0)) <= error
        profile_results.append(
            {
                "profile": profile,
                "radial_domain": [80.0, 320.0],
                "u0": 1.2,
                "ladder": ladder,
                "measured_exponent": exponent,
                "estimated_ladder_error": error,
                "lense_thirring_target_exponent": -3.0,
                "target_distance": abs(exponent + 3.0),
                "matches_lense_thirring": matches,
                "verdict": "PASS" if matches else "KILL",
            }
        )
    primary = profile_results[0]
    return {
        "id": "omega_radial_falloff",
        "primary_profile": "flat (the UI default)",
        "method": "nested log-spaced radial ladders; least-squares log Omega versus log r",
        "predeclared_pass": "the default flat profile exponent differs from -3 by no more than its estimated ladder error",
        "predeclared_kill": "the default flat profile exponent differs from -3 by more than its estimated ladder error",
        "profile_results": profile_results,
        "available_profile_matches_lense_thirring": any(
            item["matches_lense_thirring"] for item in profile_results
        ),
        "passed": primary["matches_lense_thirring"],
        "verdict": primary["verdict"],
    }


def amplitude_check() -> dict:
    amplitudes = [0.15, 0.3, 0.6, 1.2, 2.4]
    values = [omega("flat", 160.0, amplitude) for amplitude in amplitudes]
    exponent = log_log_slope(amplitudes, values)
    error = slope_error(amplitudes, values, exponent)
    matches = abs(exponent - 1.0) <= error
    return {
        "id": "spin_surrogate_scaling",
        "method": "log-log fit of Omega versus u0 at fixed r=160",
        "parameter_interpretation": "u0 is the toy's free swirl-amplitude/spin surrogate; the toy defines no angular momentum J",
        "amplitude_ladder": [
            {"u0": amplitude, "omega": value}
            for amplitude, value in zip(amplitudes, values, strict=True)
        ],
        "measured_exponent": exponent,
        "estimated_ladder_error": error,
        "lense_thirring_target_exponent": 1.0,
        "predeclared_pass": "Omega is linear in the J surrogate u0 within the amplitude-ladder error",
        "predeclared_kill": "Omega is not linear in u0 within the amplitude-ladder error",
        "passed": matches,
        "verdict": "PASS" if matches else "KILL",
    }


def sign_check() -> dict:
    samples = [
        {"u0": amplitude, "omega": omega("flat", 160.0, amplitude)}
        for amplitude in (-1.2, 0.0, 1.2)
    ]
    negative, zero, positive = (item["omega"] for item in samples)
    odd_residual = abs(positive + negative)
    tolerance = ROUNDOFF_FLOOR * max(abs(positive), 1.0)
    matches = negative < 0.0 and zero == 0.0 and positive > 0.0 and odd_residual <= tolerance
    return {
        "id": "spin_sign_reversal",
        "method": "evaluate the transcribed law at signed u0 with fixed r=160",
        "samples": samples,
        "odd_symmetry_residual": odd_residual,
        "tolerance": tolerance,
        "predeclared_pass": "Omega(-u0)=-Omega(u0), Omega(0)=0, and nonzero signs follow u0",
        "predeclared_kill": "Omega fails the odd-in-u0 sign-source test",
        "passed": matches,
        "verdict": "PASS" if matches else "KILL",
    }


def mass_diagnostic() -> dict:
    masses = [56.25, 112.5, 225.0, 450.0, 900.0]
    values = [omega("flat", 160.0, 1.2) for _gm in masses]
    exponent = log_log_slope(masses, values)
    return {
        "id": "mass_parameter_decoupling",
        "method": "vary the toy's GM parameter while holding the separately imposed flow law fixed",
        "gm_ladder": [
            {"GM": gm, "omega": value}
            for gm, value in zip(masses, values, strict=True)
        ],
        "measured_exponent": exponent,
        "interpretation": "GM affects tracer gravity but is absent from flow(r); the toy supplies no rotating-source map from mass and spin to u0",
        "verdict": "NO_SOURCE_COUPLING",
    }


def run_checks() -> dict:
    source_hash = verify_source()
    falloff = falloff_check()
    amplitude = amplitude_check()
    sign = sign_check()
    diagnostics = [mass_diagnostic()]
    checks = [falloff, amplitude, sign]
    return {
        "schema_version": 1,
        "sim": SLUG,
        "task": TASK_ID,
        "run_date": RUN_DATE,
        "source": {
            "path": "../frame-drag-swirl/index.html",
            "sha256": source_hash,
            "flow_law_lines": "35-39",
            "omega_use_lines": "74-75",
        },
        "model": {
            "measured_quantity": "Omega(r) = flow(r)/r",
            "comparator": "equatorial Omega_LT = 2 G J / (c^2 r^3)",
            "comparator_shape_tests": ["r^-3", "linear in J", "odd under J sign reversal"],
            "toy_geometry": "2-D equatorial visualization with no polar dependence",
            "units": "dimensionless browser coordinates; no lattice-to-physical normalization",
        },
        "checks": checks,
        "diagnostics": diagnostics,
        "shape_tests_passed": sum(check["passed"] for check in checks),
        "shape_tests_total": len(checks),
        "full_lense_thirring_shape_reproduced": all(check["passed"] for check in checks),
        "overall_verdict": "PASS" if all(check["passed"] for check in checks) else "KILL",
        "interpretation": "the toy is linear and odd in its free swirl amplitude, but its default Omega falls as r^-1 (and no selectable profile falls as r^-3); it does not reproduce the full Lense-Thirring shape",
    }


def canonical_bytes(result: dict) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="execute the complete fixture twice and require byte-identical canonical JSON",
    )
    parser.add_argument("--no-write", action="store_true", help="run without updating tracked artifacts")
    args = parser.parse_args()

    result = run_checks()
    payload = canonical_bytes(result)
    if args.check_determinism:
        rerun = canonical_bytes(run_checks())
        if payload != rerun:
            raise RuntimeError("determinism check failed: in-process rerun was not byte-identical")

    if not args.no_write:
        RESULT_PATH.write_bytes(payload)
        RUN_PATH.parent.mkdir(exist_ok=True)
        RUN_PATH.write_bytes(payload)

    for check in result["checks"]:
        print(f"{check['verdict']:4s}  {check['id']}")
    print(f"{result['overall_verdict']:4s}  full_lense_thirring_shape")
    if args.check_determinism:
        print("PASS  byte-identical in-process rerun")
    if not args.no_write:
        print(f"wrote {RESULT_PATH.relative_to(Path.cwd())}")
        print(f"wrote {RUN_PATH.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
