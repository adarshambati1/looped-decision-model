"""Typed examples, leakage controls, and reproducible algorithmic synthetic data."""

import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DecisionExample:
    id: str
    state: str
    instructions: str
    kind: str
    options: tuple[str, ...]
    targets: tuple[float, ...]
    source: str
    group_id: str
    steps: int | None = None

    def __post_init__(self):
        if self.kind not in {"choice", "score", "noul"}:
            raise ValueError("Invalid decision kind")
        if not 2 <= len(self.options) <= 64 or len(self.targets) != len(self.options):
            raise ValueError("Invalid option/label shape")
        if len(set(self.options)) != len(self.options):
            raise ValueError("Duplicate options")
        if self.kind == "noul" and set(self.options) != {"false", "true"}:
            raise ValueError("Noul requires false/true options")
        if not all(math.isfinite(p) and p >= 0 for p in self.targets):
            raise ValueError("Invalid target probabilities")
        if not math.isclose(sum(self.targets), 1.0, abs_tol=1e-8):
            raise ValueError("Targets must sum to one")
        if not self.id or not self.group_id:
            raise ValueError("Explicit item and group identities are required")


def permute_options(example, rng: random.Random):
    if example.kind == "score":
        return example
    order = list(range(len(example.options)))
    rng.shuffle(order)
    return replace(
        example,
        options=tuple(example.options[i] for i in order),
        targets=tuple(example.targets[i] for i in order),
    )


def group_split(examples, heldout_fraction=0.05, seed=238):
    if not 0 < heldout_fraction < 1:
        raise ValueError("Invalid heldout fraction")
    train, heldout = [], []
    # Caller MUST canonicalize group IDs across datasets/duplicate states first.
    for item in examples:
        digest = hashlib.sha256(f"{seed}:{item.group_id}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        (heldout if fraction < heldout_fraction else train).append(item)
    return train, heldout


def canonical_group(state):
    return hashlib.sha256(normalize(state).encode()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngram_hashes(text, n=13):
    if n < 1:
        raise ValueError("n must be positive")
    words = re.findall(r"\w+", normalize(text), flags=re.UNICODE)
    return {
        hashlib.sha256(" ".join(words[i : i + n]).encode()).digest()
        for i in range(len(words) - n + 1)
    }


def decontaminate(examples, evaluation_hashes):
    kept, dropped_ids = [], []
    for item in examples:
        text = "\n".join([item.state, item.instructions, *item.options])
        if ngram_hashes(text) & evaluation_hashes:
            dropped_ids.append(item.id)
        else:
            kept.append(item)
    return kept, dropped_ids


def synthetic_examples(count, seed=238, max_steps=8):
    """Half pointer lookups, half running balances. Known operations, not BBH."""
    if count < 0 or max_steps < 1:
        raise ValueError("Invalid generation size")
    rng = random.Random(seed)
    for index in range(count):
        steps = rng.randint(1, max_steps)
        if index % 2 == 0:
            labels = rng.sample(range(10000, 99999), steps + 4)
            names = [f"record_{x}" for x in labels]
            records = {names[i]: {"next": names[i + 1]} for i in range(steps)}
            # Three irrelevant records prevent a sole-terminal shortcut.
            for name in names[steps:]:
                records[name] = {"next": name}
            keys = list(records)
            rng.shuffle(keys)
            state = json.dumps({"start": names[0], "records": {k: records[k] for k in keys}})
            answer = names[steps]
            wrong = rng.sample([x for x in names if x != answer], 3)
            instructions = f"Follow exactly {steps} next links from start. Which record is reached?"
            source = "synthetic/multihop"
        else:
            initial = rng.randint(100, 1000)
            events = [rng.choice([-1, 1]) * rng.randint(1, 50) for _ in range(steps)]
            answer_value = initial + sum(events)
            state = json.dumps({"initial_balance": initial, "events": events})
            answer = str(answer_value)
            offsets = rng.sample([x for x in range(-40, 41) if x], 3)
            wrong = [str(answer_value + x) for x in offsets]
            instructions = "Apply every signed event in order. What is the final balance?"
            source = "synthetic/balance"
        options = [answer, *wrong]
        rng.shuffle(options)
        yield DecisionExample(
            id=f"{source}/{seed}/{index}",
            state=state,
            instructions=instructions,
            kind="choice",
            options=tuple(options),
            targets=tuple(float(x == answer) for x in options),
            source=source,
            group_id=canonical_group(state),
            steps=steps,
        )
