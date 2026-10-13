import numpy as np
import pytest

from looped_decision.calibration import (
    Trajectory,
    calibrate,
    confidence,
    fit_temperature,
    softmax,
    stop,
)
from looped_decision.evaluation import (
    PairedResult,
    c_preview_passes,
    choose_stage_c,
    latency_summary,
    paired_bootstrap,
)


def test_confidence_and_stability():
    assert confidence([0.5, 0.5]) == 0
    assert confidence([1, 0]) == 1
    assert confidence([0.91, 0.09]) == pytest.approx(0.82)
    assert np.allclose(softmax([10000, 10000]), [0.5, 0.5])
    with pytest.raises(ValueError):
        confidence([0.2, 0.2])


def test_soft_labels_determine_temperature():
    t = fit_temperature([np.array([2.0, 0.0])] * 5, [np.array([0.6, 0.4])] * 5)
    assert softmax([2.0, 0.0], t)[0] == pytest.approx(0.6, abs=1e-5)


def test_adaptive_stop_uses_pmax_not_normalized_confidence():
    row = Trajectory("a", "choice", [[2.0, 0.0], [3.0, 0.0]], [1.0, 0.0])
    k, _ = stop(row, {("choice", 1): 1.0, ("choice", 2): 1.0}, 0.85, 2)
    assert k == 1  # pmax~.881; normalized confidence~.762
    assert stop(row, {("choice", 1): 1.0, ("choice", 2): 1.0}, 0.99, 2)[0] == 2


def test_three_step_calibration():
    rows = [Trajectory(str(i), "choice", [[2.0, 0.0], [3.0, 0.0]], [0.8, 0.2]) for i in range(6)]
    fitted = calibrate(rows, 2)
    assert 0 <= fitted["tau"] <= 1
    assert sum(fitted["selected_counts"].values()) == len(rows)
    assert set(fitted["temperatures"]) == {("choice", 1), ("choice", 2)}


def test_paired_bootstrap_and_gates():
    result = paired_bootstrap({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}, resamples=100)
    assert result.lower == result.upper == result.gap == 1
    assert choose_stage_c(result) == "B"
    zero = PairedResult(0, 0, 0, 231, 100, 0)
    assert choose_stage_c(zero) == "A"
    assert c_preview_passes(zero, 5, 10)  # preview allows inconclusive quality gap
    assert not c_preview_passes(zero, 10, 5)
    with pytest.raises(ValueError):
        paired_bootstrap({"a": 1}, {"b": 1})


def test_bootstrap_recomputes_nonlinear_metric():
    result = paired_bootstrap(
        {"a": 0.0, "b": 2.0}, {"a": 1.0, "b": 1.0}, metric=np.var, resamples=100
    )
    assert result.gap == 1


def test_cost_uses_mean_not_median():
    summary = latency_summary([100, 100, 1000], [1, 1, 1], 3.6)
    assert summary["p50_request_ms"] == 100
    assert summary["steady_state_dollars_per_1000"] == pytest.approx(0.4)
