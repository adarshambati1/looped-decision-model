"""Jev HTTP shapes verified against stiermid/laya-serve; see docs/contract.md.

Independent schema implementation. Project-specific limits are deliberate.
"""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .calibration import confidence

StructuredText = str | dict[str, Any] | list[Any]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class NoulCriteria(StrictModel):
    true: StructuredText | None = None
    false: StructuredText | None = None


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: StructuredText
    criteria: NoulCriteria | None = None


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: StructuredText
    criteria: dict[str, StructuredText | None] = Field(min_length=2, max_length=64)


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: StructuredText
    criteria: list[StructuredText] = Field(min_length=2, max_length=10)


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class SystemOneRequest(StrictModel):
    model: str = Field(min_length=1)
    state: StructuredText
    questions: dict[str, Question] = Field(min_length=1, max_length=64)


Probability = Annotated[float, Field(ge=0, le=1)]


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, Probability]
    legend: dict[str, StructuredText]
    confidence: Probability


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: Probability


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class SystemOneResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def make_answer(question: Question, probabilities):
    """Probabilities must already be calibrated and in ORIGINAL request order.

    Noul's canonical internal order is [false, true], including after shuffled
    training. Timing/pass counts do not belong in Jev's usage object.
    """
    p = np.asarray(probabilities, dtype=float)
    conf = confidence(p)  # validates normalization/finiteness
    expected = 2 if question.type == "noul" else len(question.criteria)
    if len(p) != expected:
        raise ValueError("Probability count does not match question")
    if question.type == "noul":
        return NoulAnswer(noul=float(p[1]))
    if question.type == "choice":
        keys = list(question.criteria)
        return ChoiceAnswer(
            choice=keys[int(p.argmax())], probabilities=dict(zip(keys, p.tolist())), confidence=conf
        )
    keys = [str(i) for i in range(len(p))]
    return ScoreAnswer(
        score=float(np.arange(len(p)) @ p),
        probabilities=dict(zip(keys, p.tolist())),
        legend=dict(zip(keys, question.criteria)),
        confidence=conf,
    )
