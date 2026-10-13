"""Prompt construction with auditable boundary tokenization and state-only truncation."""

import json
import string
from dataclasses import dataclass

from .api_contract import Question

CODE_CANDIDATES = string.ascii_uppercase + string.ascii_lowercase + string.digits + "!?"


def render(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )


def verify_codes(tokenizer, codes=CODE_CANDIDATES):
    """Require each code to be a one-token continuation, not a merged '(A'.

    This fails closed: silently appending a separator would change the prompt.
    Audit on the actual tokenizer before initializing the output-embedding head.
    """
    prefix = tokenizer.encode("Answer: (", add_special_tokens=False)
    ids = []
    for code in codes:
        full = tokenizer.encode("Answer: (" + code, add_special_tokens=False)
        if full[:-1] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f"Code {code!r} is not a single prefix-stable token after '('")
        ids.append(full[-1])
    if len(codes) != 64 or len(set(ids)) != 64:
        raise ValueError("Need 64 distinct single-token codes")
    return tuple(ids)


def question_suffix(question: Question, codes=CODE_CANDIDATES):
    if question.type == "choice":
        options = [(key, value) for key, value in question.criteria.items()]
    elif question.type == "score":
        options = [(str(i), value) for i, value in enumerate(question.criteria)]
    else:
        options = [
            ("false", question.criteria.false if question.criteria else None),
            ("true", question.criteria.true if question.criteria else None),
        ]
    return option_suffix(question.instructions, options, codes)


def option_suffix(instructions, options, codes=CODE_CANDIDATES):
    lines = ["\n\nQuestion: " + render(instructions)]
    for code, (name, description) in zip(codes, options):
        detail = "" if description is None else ": " + render(description)
        lines.append(f"({code}) {name}{detail}")
    lines.append("Answer: (")
    return "\n".join(lines)


def prepare_example(tokenizer, item, max_tokens=4096):
    """Training suffix uses the exact serving renderer, including descriptions."""
    descriptions = item.descriptions or (None,) * len(item.options)
    suffix = option_suffix(item.instructions, zip(item.options, descriptions))
    question_ids = tokenizer.encode(suffix, add_special_tokens=False)
    budget = max_tokens - len(question_ids)
    if budget < 1:
        raise ValueError(f"Question too long: {item.id}")
    state_ids = tokenizer.encode(item.state, add_special_tokens=False)
    if item.conversation and len(state_ids) > budget:
        turns = json.loads(item.state)
        if not isinstance(turns, list):
            raise ValueError("Conversation must be serialized turns")
        while len(turns) > 1 and len(state_ids) > budget:
            turns.pop(0)
            state_ids = tokenizer.encode(render(turns), add_special_tokens=False)
    state_ids = state_ids[-budget:] if item.conversation else state_ids[:budget]
    return state_ids, question_ids


@dataclass(frozen=True)
class PreparedRequest:
    state_ids: tuple[int, ...]
    question_ids: dict[str, tuple[int, ...]]
    state_was_truncated: bool


def prepare_request(tokenizer, state, questions, max_tokens=4096, conversation=False):
    """A common tokenized state prefix sized to fit the longest question.

    Tokenize prefix and suffix separately so all questions share EXACT state IDs.
    These exact concatenated IDs, not retokenized strings, must be used in BOTH
    training and serving. For conversation lists, retain whole latest turns when
    possible, then the tail of the last remaining turn if necessary.
    """
    if not questions:
        raise ValueError("Need at least one question")
    suffixes = {
        qid: tuple(tokenizer.encode(question_suffix(q), add_special_tokens=False))
        for qid, q in questions.items()
    }
    budget = max_tokens - max(map(len, suffixes.values()))
    if budget < 1:
        raise ValueError("Question/options exceed context; cannot truncate them")
    text = render(state)
    state_ids = tokenizer.encode(text, add_special_tokens=False)
    truncated = len(state_ids) > budget
    if truncated and conversation and isinstance(state, list):
        turns = list(state)
        while len(turns) > 1 and len(state_ids) > budget:
            turns.pop(0)
            state_ids = tokenizer.encode(render(turns), add_special_tokens=False)
    if len(state_ids) > budget:
        state_ids = state_ids[-budget:] if conversation else state_ids[:budget]
    return PreparedRequest(tuple(state_ids), suffixes, truncated)
