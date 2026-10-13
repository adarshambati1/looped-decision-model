"""Paired statistics and publication gates. Official harnesses own their metrics."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedResult:
    gap: float
    lower: float
    upper: float
    n: int
    resamples: int
    seed: int


def paired_bootstrap(
    treatment: Mapping[str, object],
    control: Mapping[str, object],
    metric: Callable = np.mean,
    resamples: int = 10000,
    seed: int = 238,
) -> PairedResult:
    """Resample identical item indices; recompute EACH system's full metric.

    For DecisionBench accuracy use per-item booleans and mean. JevBench callers
    must supply its official aggregate capability function, not average a made-up
    per-item capability (its calibration aggregate can be nonlinear).
    """
    if not treatment or treatment.keys() != control.keys():
        raise ValueError("Paired bootstrap requires exactly matching nonempty item IDs")
    if resamples < 100:
        raise ValueError("Use at least 100 resamples")
    ids = sorted(treatment)
    a = np.asarray([treatment[i] for i in ids])
    b = np.asarray([control[i] for i in ids])
    n = len(ids)
    rng = np.random.default_rng(seed)
    gaps = np.empty(resamples)
    for j in range(resamples):
        idx = rng.integers(0, n, n)
        gaps[j] = float(metric(a[idx])) - float(metric(b[idx]))
    if not np.isfinite(gaps).all():
        raise ValueError("Metric produced a nonfinite value")
    gap = float(metric(a)) - float(metric(b))
    if not np.isfinite(gap):
        raise ValueError("Nonfinite observed gap")
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return PairedResult(gap, float(lo), float(hi), n, resamples, seed)


def choose_stage_c(b_extra_passes: PairedResult):
    return "B" if b_extra_passes.lower > 0 else "A"


def c_preview_passes(capability_gap: PairedResult, c_p50_ms: float, h2o_p50_ms: float):
    if not all(np.isfinite(x) and x > 0 for x in (c_p50_ms, h2o_p50_ms)):
        raise ValueError("Need positive measured p50 latencies on identical hardware")
    # The user's preview gate is not a claim of statistical superiority.
    return c_p50_ms < h2o_p50_ms and capability_gap.upper >= 0


def latency_summary(request_ms, decisions_per_request, gpu_dollars_per_hour):
    ms = np.asarray(request_ms, dtype=float)
    count = np.asarray(decisions_per_request, dtype=int)
    if ms.ndim != 1 or not len(ms) or count.shape != ms.shape:
        raise ValueError("Need paired requests and question counts")
    if not np.isfinite(ms).all() or np.any(ms <= 0) or np.any(count < 1):
        raise ValueError("Invalid latency or decision count")
    if not np.isfinite(gpu_dollars_per_hour) or gpu_dollars_per_hour <= 0:
        raise ValueError("Need actual positive hourly GPU rate")
    return {
        "p50_request_ms": float(np.median(ms)),
        "p50_amortized_decision_ms": float(np.median(ms / count)),
        "mean_amortized_decision_ms": float(ms.sum() / count.sum()),
        # Cost uses total walltime / total decisions, NEVER p50 * GPU rate.
        "steady_state_dollars_per_1000": float(
            ms.sum() / count.sum() * gpu_dollars_per_hour / 3600
        ),
        "requests": len(ms),
        "decisions": int(count.sum()),
        "note": "Warm serial-request serving only; excludes idle, setup, training, storage.",
    }
