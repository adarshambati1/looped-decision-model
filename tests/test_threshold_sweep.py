import numpy as np
import pytest

from looped_decision.calibration import Trajectory, policy_summary, select_threshold, softmax


@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_sweep_exactly_matches_brute_force(cap):
    for seed in range(5):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(5):
            options = int(rng.integers(2, 7))
            # Discrete values deliberately produce equal-confidence ties.
            logits = rng.integers(-4, 5, size=(cap, options)).astype(float)
            target = np.eye(options)[int(rng.integers(options))]
            rows.append(Trajectory(str(i), "choice", logits, target))
        temperatures = {
            ("choice", k): float(rng.choice([0.05, 0.5, 1, 3])) for k in range(1, cap + 1)
        }
        candidates = {0.0, 1.0}
        for row in rows:
            for k in range(1, cap):
                c = float(softmax(row.logits[k - 1], temperatures[("choice", k)]).max())
                candidates.update([c, min(1.0, float(np.nextafter(c, np.inf)))])
        baseline = float(
            np.mean(
                [
                    softmax(r.logits[cap - 1], temperatures[("choice", cap)]).argmax()
                    == r.target.argmax()
                    for r in rows
                ]
            )
        )
        feasible = []
        for tau in candidates:
            summary = policy_summary(rows, temperatures, tau, cap)
            if summary["accuracy"] + 1e-12 >= baseline - 0.01:
                feasible.append((summary["mean_passes"], -summary["accuracy"], -tau, tau))
        if feasible:
            assert select_threshold(rows, temperatures, cap) == min(feasible)[-1]
        else:
            with pytest.raises(ValueError, match="No feasible"):
                select_threshold(rows, temperatures, cap)
