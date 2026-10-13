import json
import random
from dataclasses import replace

from looped_decision.api_contract import ChoiceQuestion
from looped_decision.data import (
    canonical_group,
    decontaminate,
    group_split,
    ngram_hashes,
    permute_options,
    synthetic_examples,
)
from looped_decision.prompting import prepare_request


def test_synthetic_oracles():
    examples = list(synthetic_examples(40))
    assert examples == list(synthetic_examples(40))
    for item in examples:
        state = json.loads(item.state)
        if item.source.endswith("balance"):
            answer = str(state["initial_balance"] + sum(state["events"]))
            assert item.steps == len(state["events"])
        else:
            answer = state["start"]
            for _ in range(item.steps):
                answer = state["records"][answer]["next"]
        assert item.options[item.targets.index(1.0)] == answer


def test_permutation_and_score_order():
    item = next(synthetic_examples(1))
    soft = replace(item, targets=(0.1, 0.2, 0.3, 0.4))
    perm = permute_options(soft, random.Random(1))
    assert dict(zip(soft.options, soft.targets)) == dict(zip(perm.options, perm.targets))
    score = replace(soft, kind="score")
    assert permute_options(score, random.Random(1)) is score


def test_group_split_and_13grams():
    examples = list(synthetic_examples(100))
    examples += [replace(examples[0], id="duplicate-question")]
    train, heldout = group_split(examples)
    assert not {x.group_id for x in train} & {x.group_id for x in heldout}
    assert ngram_hashes("one two three") == set()
    sentence = " ".join(f"word{i}" for i in range(13))
    item = replace(examples[0], instructions=sentence)
    keep, dropped = decontaminate([item], ngram_hashes(sentence.upper()))
    assert not keep and dropped == [item.id]
    assert canonical_group("Ａ B") == canonical_group("a b")


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


def test_truncate_only_shared_state():
    tokenizer = CharacterTokenizer()
    q = ChoiceQuestion(type="choice", instructions="Pick", criteria={"a": None, "b": None})
    prepared = prepare_request(tokenizer, "X" * 200, {"q": q}, max_tokens=100)
    assert prepared.state_was_truncated
    assert len(prepared.state_ids) + len(prepared.question_ids["q"]) == 100
    assert "".join(map(chr, prepared.question_ids["q"])).endswith("Answer: (")
