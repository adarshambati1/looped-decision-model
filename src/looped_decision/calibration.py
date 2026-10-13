"""Offline calibration and stopping; never fit this module on benchmark outputs.

Each trajectory contains all pass logits for one held-out decision. Temperature
refitting uses the selected stopping population once; tau remains fixed afterward.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

KINDS = {"choice", "score", "noul"}


def softmax(logits, temperature=1.0):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite logits over at least two options")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    x = x / temperature
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def confidence(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not 2 <= len(p) <= 64 or not np.isfinite(p).all():
        raise ValueError("Expected 2-64 finite probabilities")
    if np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Invalid probability distribution")
    return float(np.clip((len(p) * p.max() - 1) / (len(p) - 1), 0, 1))


@dataclass(frozen=True)
class Trajectory:
    item_id: str
    kind: str
    logits: np.ndarray  # [pass, option], option count can differ across items
    target: np.ndarray  # normalized SOFT labels, never replaced with argmax for NLL

    def __post_init__(self):
        x = np.asarray(self.logits, dtype=float).copy()
        t = np.asarray(self.target, dtype=float).copy()
        if self.kind not in KINDS or x.ndim != 2 or x.shape[0] < 1:
            raise ValueError("Invalid trajectory kind or shape")
        if not 2 <= x.shape[1] <= 64 or t.shape != (x.shape[1],):
            raise ValueError("Target must match 2-64 options")
        if self.kind == "noul" and len(t) != 2:
            raise ValueError("Noul has exactly two options")
        if not np.isfinite(x).all() or not np.isfinite(t).all():
            raise ValueError("Nonfinite logits/targets")
        if np.any(t < 0) or not np.isclose(t.sum(), 1):
            raise ValueError("Invalid target distribution")
        x.setflags(write=False)
        t.setflags(write=False)
        object.__setattr__(self, "logits", x)
        object.__setattr__(self, "target", t)


def fit_temperature(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray]):
    """Bounded log-temperature minimization of mean soft-label cross entropy."""
    if not logits or len(logits) != len(targets):
        raise ValueError("Need paired nonempty logits and targets")
    # Validate using the same shape/target checks as stored trajectories.
    rows = [
        Trajectory(str(i), "choice", np.asarray(x)[None, :], t)
        for i, (x, t) in enumerate(zip(logits, targets))
    ]

    def nll(log_t):
        losses = []
        for row in rows:
            z = row.logits[0] / math.exp(log_t)
            z = z - z.max()
            log_p = z - np.log(np.exp(z).sum())
            losses.append(-float(row.target @ log_p))
        return float(np.mean(losses))

    lo, hi = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fc, fd = nll(c), nll(d)
    for _ in range(90):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - ratio * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + ratio * (hi - lo)
            fd = nll(d)
    candidates = [math.log(0.05), math.log(20), 0.0, (lo + hi) / 2]
    return math.exp(min(candidates, key=nll))


def fit_per_pass(rows: Sequence[Trajectory]):
    groups = {}
    for row in rows:
        for k, logits in enumerate(row.logits, start=1):
            groups.setdefault((row.kind, k), []).append((logits, row.target))
    return {
        key: fit_temperature([x for x, _ in pairs], [t for _, t in pairs])
        for key, pairs in groups.items()
    }


def stop(row: Trajectory, temperatures: Mapping, tau: float, max_passes: int):
    if not 0 <= tau <= 1 or not 1 <= max_passes <= len(row.logits):
        raise ValueError("Invalid threshold/pass cap")
    for k in range(1, max_passes + 1):
        p = softmax(row.logits[k - 1], temperatures[(row.kind, k)])
        if p.max() >= tau or k == max_passes:
            return k, p
    raise AssertionError("Unreachable")


def policy_summary(rows, temperatures, tau, max_passes):
    selected = [stop(row, temperatures, tau, max_passes) for row in rows]
    if not selected:
        raise ValueError("No held-out rows")
    return {
        "accuracy": float(
            np.mean([p.argmax() == row.target.argmax() for row, (_, p) in zip(rows, selected)])
        ),
        "mean_passes": float(np.mean([k for k, _ in selected])),
    }


def select_threshold(rows, temperatures, max_passes, tolerance=0.01):
    """Exact heldout empirical candidates; fastest means lowest average passes.

    This is a compute-depth proxy, NOT measured latency. Actual GPU timing must
    be measured separately. Ties prefer accuracy then the larger threshold.
    """
    if not rows or not 0 <= tolerance <= 1:
        raise ValueError("Need held-out rows and valid accuracy tolerance")
    candidates = {0.0, 1.0}
    maximum_correct = []
    for row in rows:
        if len(row.logits) < max_passes:
            raise ValueError("Missing maximum-pass output")
        for k in range(1, max_passes + 1):
            p = softmax(row.logits[k - 1], temperatures[(row.kind, k)])
            if k < max_passes:
                c = float(p.max())
                candidates.add(c)
                candidates.add(min(1.0, float(np.nextafter(c, np.inf))))
        maximum_correct.append(p.argmax() == row.target.argmax())
    baseline = float(np.mean(maximum_correct))
    feasible = []
    for tau in sorted(candidates):
        summary = policy_summary(rows, temperatures, tau, max_passes)
        if summary["accuracy"] + 1e-12 >= baseline - tolerance:
            feasible.append((summary["mean_passes"], -summary["accuracy"], -tau, tau))
    if not feasible:
        raise ValueError("No feasible threshold; numerical saturation may force early exits")
    return min(feasible)[-1]


def calibrate(rows: Sequence[Trajectory], max_passes: int):
    """Required order: per-pass fit -> tau -> one selected-population refit."""
    first = fit_per_pass(rows)
    tau = select_threshold(rows, first, max_passes)
    selected_groups = {}
    for row in rows:
        k, _ = stop(row, first, tau, max_passes)
        selected_groups.setdefault((row.kind, k), []).append(row)
    final = dict(first)
    for (kind, k), group in selected_groups.items():
        final[(kind, k)] = fit_temperature(
            [row.logits[k - 1] for row in group], [row.target for row in group]
        )
    return {
        "tau": tau,
        "temperatures": final,
        "selected_counts": {key: len(group) for key, group in selected_groups.items()},
        "before_refit": policy_summary(rows, first, tau, max_passes),
        "after_refit": policy_summary(rows, final, tau, max_passes),
    }
