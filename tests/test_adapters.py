import json
import random

import pytest

from looped_decision.adapters import boolq, multiple_choice, sst5, typed_decisions
from looped_decision.api_contract import ChoiceQuestion
from looped_decision.data import permute_options
from looped_decision.prompting import prepare_example, question_suffix


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


def test_soft_targets_descriptions_and_shared_group():
    row = {
        "id": "example",
        "state": json.dumps({"b": 2, "a": 1}),
        "questions": json.dumps(
            {
                "q": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": "first", "b": "second"},
                },
                "s": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
            }
        ),
        "gold": json.dumps(
            {
                "q": {"probabilities": {"a": 0.333333, "b": 0.666666}},
                "s": {"probabilities": {"0": 0.2, "1": 0.8}},
            }
        ),
    }
    choice, score = list(typed_decisions(row))
    assert choice.group_id == score.group_id
    assert choice.targets == pytest.approx((1 / 3, 2 / 3))
    assert choice.descriptions == ("first", "second")
    shuffled = permute_options(choice, random.Random(1))
    assert dict(zip(shuffled.options, shuffled.descriptions)) == {"a": "first", "b": "second"}
    assert dict(zip(shuffled.options, shuffled.targets)) == pytest.approx({"a": 1 / 3, "b": 2 / 3})
    assert permute_options(score, random.Random(1)) is score
    _, suffix = prepare_example(CharacterTokenizer(), choice)
    q = ChoiceQuestion(type="choice", instructions="Choose", criteria={"a": "first", "b": "second"})
    assert "".join(map(chr, suffix)) == question_suffix(q)


def test_choice_and_boolq_mapping():
    choice = multiple_choice(
        {
            "id": "1",
            "question": "Which?",
            "choices": {"label": ["A", "B"], "text": ["same", "same"]},
            "answerKey": "B",
        },
        "test",
    )
    assert choice.targets == (0, 1)
    assert choice.descriptions == ("same", "same")
    yes = boolq({"passage": "Context", "question": "True?", "answer": True}, 0)
    assert yes.targets == (0, 1)
    assert dict(
        zip(
            permute_options(yes, random.Random(1)).options,
            permute_options(yes, random.Random(1)).targets,
        )
    ) == {"false": 0, "true": 1}
    with pytest.raises(TypeError):
        boolq({"passage": "Context", "question": "True?", "answer": "yes"}, 0)


def test_sst_mapping_does_not_imply_license_clearance():
    item = sst5({"text": "great", "label": 4}, 0)
    assert item.options == ("0", "1", "2", "3", "4")
    assert item.targets == (0, 0, 0, 0, 1)
    assert item.descriptions[-1] == "very positive"


def test_conversation_whole_turn_truncation():
    from dataclasses import replace

    item = boolq({"passage": "Context", "question": "True?", "answer": True}, 0)
    turns = [{"role": "user", "content": "old" * 100}, {"role": "assistant", "content": "new"}]
    state = json.dumps(turns, ensure_ascii=False, separators=(",", ":"))
    item = replace(item, state=state, conversation=True)
    tokenizer = CharacterTokenizer()
    _, suffix = prepare_example(tokenizer, item)
    latest = json.dumps(turns[-1:], separators=(",", ":"))
    state_ids, _ = prepare_example(tokenizer, item, max_tokens=len(suffix) + len(latest))
    assert "".join(map(chr, state_ids)) == latest
