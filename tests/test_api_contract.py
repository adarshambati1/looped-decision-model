import pytest
from pydantic import ValidationError

from looped_decision.api_contract import SystemOneRequest, make_answer


def request(kind, criteria=None):
    q = {"type": kind, "instructions": "judge"}
    if criteria is not None:
        q["criteria"] = criteria
    return SystemOneRequest(model="looped-decision-1.4b", state={"x": 1}, questions={"q": q})


def test_exact_choice_wire():
    q = request("choice", {"a": None, "b": {"text": "B"}}).questions["q"]
    a = make_answer(q, [0.91, 0.09]).model_dump()
    assert set(a) == {"type", "choice", "probabilities", "confidence"}
    assert a["choice"] == "a"
    assert a["confidence"] == pytest.approx(0.82)


def test_score_dict_legend_and_expectation():
    q = request("score", ["low", "mid", "high"]).questions["q"]
    a = make_answer(q, [0.1, 0.2, 0.7]).model_dump()
    assert a["score"] == pytest.approx(1.6)
    assert a["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert a["probabilities"] == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert "confidence" in a


def test_noul_only_true_probability():
    q = request("noul", {"true": "supported", "false": "unsupported"}).questions["q"]
    assert make_answer(q, [0.07, 0.93]).model_dump() == {"type": "noul", "noul": 0.93}


def test_limits_and_invalid_distribution():
    with pytest.raises(ValidationError):
        request("choice", {"only": None})
    with pytest.raises(ValidationError):
        request("choice", {str(i): None for i in range(65)})
    with pytest.raises(ValueError):
        make_answer(request("noul").questions["q"], [0.1, 0.1])
    with pytest.raises(ValidationError):
        SystemOneRequest(model="x", state="", questions={})
