"""Pure dataset adapters. No network, evaluation access, or inferred hard labels."""

import json
import math

from .data import DecisionExample, canonical_group
from .prompting import render


def typed_decisions(row):
    state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
    questions = (
        json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
    )
    gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
    # Canonical JSON key order avoids group leakage from equivalent objects.
    text = (
        state
        if isinstance(state, str)
        else json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    conversation = isinstance(state, list) and all(
        isinstance(t, dict) and "role" in t for t in state
    )
    for qid, q in questions.items():
        kind = q["type"]
        criteria = q.get("criteria")
        if kind == "score":
            options = tuple(map(str, range(len(criteria))))
            descriptions = tuple(map(render, criteria))
        elif kind == "noul":
            options = ("false", "true")
            descriptions = tuple((criteria or {}).get(k) for k in options)
        else:
            options = tuple(criteria)
            descriptions = tuple(
                render(criteria[k]) if criteria[k] is not None else None for k in options
            )
        probabilities = gold[qid]["probabilities"]
        if set(probabilities) != set(options):
            raise ValueError("Gold probabilities and options differ")
        targets = tuple(float(probabilities[k]) for k in options)
        total = sum(targets)
        # Public labels are rounded to six decimals; normalize that rounding only.
        if not math.isclose(total, 1, abs_tol=5e-6):
            raise ValueError("Invalid source probability sum")
        yield DecisionExample(
            id=f"typed/{row['id']}/{qid}",
            state=text,
            instructions=render(q["instructions"]),
            kind=kind,
            options=options,
            targets=tuple(p / total for p in targets),
            source="LocalLLaMA/typed-decisions",
            group_id=canonical_group(text),
            descriptions=descriptions,
            conversation=conversation,
        )


def multiple_choice(row, source):
    labels = tuple(row["choices"]["label"])
    if row["answerKey"] not in labels:
        raise ValueError("Answer key not in source choice labels")
    state = row["question"]
    return DecisionExample(
        id=f"{source}/{row['id']}",
        state=state,
        instructions="Select the best answer to the question.",
        kind="choice",
        options=labels,
        descriptions=tuple(row["choices"]["text"]),
        targets=tuple(float(k == row["answerKey"]) for k in labels),
        source=source,
        group_id=canonical_group(state),
    )


def boolq(row, index):
    if not isinstance(row["answer"], bool):
        raise TypeError("BoolQ answer must be boolean")
    return DecisionExample(
        id=f"google/boolq/{index}",
        state=row["passage"],
        instructions=row["question"],
        kind="noul",
        options=("false", "true"),
        targets=(float(not row["answer"]), float(row["answer"])),
        source="google/boolq",
        group_id=canonical_group(row["passage"]),
        descriptions=("The answer to the question is no.", "The answer to the question is yes."),
    )


def sst5(row, index):
    """Adapter only; ingest excludes SST5 until original data rights are resolved."""
    label = row["label"]
    if not isinstance(label, int) or not 0 <= label < 5:
        raise ValueError("SST5 needs a 0..4 label")
    return DecisionExample(
        id=f"SetFit/sst5/{index}",
        state=row["text"],
        instructions="Rate the sentiment of this text.",
        kind="score",
        options=tuple(map(str, range(5))),
        descriptions=("very negative", "negative", "neutral", "positive", "very positive"),
        targets=tuple(float(i == label) for i in range(5)),
        source="SetFit/sst5",
        group_id=canonical_group(row["text"]),
    )
