"""GPU pilot trainer for A/B; requires a separately audited, decontaminated manifest.

Correctness-first, per-example gradient accumulation. Not final throughput code.
A full run must record data mixing, sample/group splits, and convergence curves.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from looped_decision.data import DecisionExample, permute_options
from looped_decision.ouro import (
    OURO_ID,
    OURO_REVISION,
    OuroDecisionModel,
    attach_lora,
    weighted_pass_loss,
)
from looped_decision.prompting import CODE_CANDIDATES, verify_codes


def prompt_ids(tokenizer, item):
    suffix = "\n\nQuestion: " + item.instructions + "\n"
    suffix += "\n".join(f"({c}) {option}" for c, option in zip(CODE_CANDIDATES, item.options))
    suffix += "\nAnswer: ("
    question = tokenizer.encode(suffix, add_special_tokens=False)
    budget = 4096 - len(question)
    if budget < 1:
        raise ValueError(f"Question too long: {item.id}")
    # Normalized data must already apply conversation truncation at whole turns.
    state = tokenizer.encode(item.state, add_special_tokens=False)[:budget]
    return state, question


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=238)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Real model training is GPU-only; do not load weights on laptop.")
    if args.steps < 1 or args.accumulate < 1 or args.lr <= 0:
        raise SystemExit("Invalid training hyperparameters")
    manifest = json.loads(args.manifest.read_text())
    required = ["licenses_reviewed", "group_split_verified", "evaluation_13grams_checked"]
    if any(manifest.get(key) is not True for key in required):
        raise SystemExit("Dataset audit gates incomplete; no weights loaded.")
    if manifest.get("train_sha256") != hashlib.sha256(args.data.read_bytes()).hexdigest():
        raise SystemExit("Manifest does not match this training file")
    if manifest.get("split") != "train":
        raise SystemExit("Only the train split can update weights")
    if args.stage == "B" and args.init is None:
        raise SystemExit("B MUST initialize from a trained A checkpoint")
    if args.output.exists():
        raise SystemExit("Use a fresh output directory; never overwrite a checkpoint")
    rows = []
    for line in args.data.read_text().splitlines():
        data = json.loads(line)
        data["options"], data["targets"] = tuple(data["options"]), tuple(data["targets"])
        rows.append(DecisionExample(**data))
    if not rows:
        raise SystemExit("Empty training set")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(OURO_ID, revision=OURO_REVISION)
    code_ids = verify_codes(tokenizer)
    config = AutoConfig.from_pretrained(OURO_ID, revision=OURO_REVISION, trust_remote_code=True)
    config.total_ut_steps = 4
    config.early_exit_threshold = 1.0
    lm = AutoModelForCausalLM.from_pretrained(
        OURO_ID,
        revision=OURO_REVISION,
        code_revision=OURO_REVISION,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    if args.init:
        metadata = json.loads((args.init / "checkpoint.json").read_text())
        if args.stage == "B" and metadata["stage"] != "A":
            raise SystemExit("B initialization must have stage=A provenance")
        if metadata["code_ids"] != list(code_ids) or metadata["base_revision"] != OURO_REVISION:
            raise SystemExit("Checkpoint/tokenizer/base provenance mismatch")
        adapted = PeftModel.from_pretrained(lm, args.init / "adapter", is_trainable=True)
    else:
        adapted = attach_lora(lm)
    model = OuroDecisionModel(adapted.get_base_model(), code_ids).train()
    if args.init:
        model.head.load_state_dict(load_file(args.init / "head.safetensors"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    args.output.mkdir(parents=True)
    started = perf_counter()
    max_passes = 4 if args.stage == "A" else 8
    with (args.output / "train.jsonl").open("w") as log:
        for step in range(args.steps):
            # K is sampled once per accumulated batch, not per pass/example.
            K = rng.randint(1, max_passes)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for _ in range(args.accumulate):
                item = permute_options(rng.choice(rows), rng)
                state_ids, question_ids = prompt_ids(tokenizer, item)
                passes = model.shared_passes(state_ids, [question_ids], K, args.stage)
                logits = torch.stack([p.logits[0][:, : len(item.options)] for p in passes])
                targets = torch.tensor([item.targets], device=logits.device, dtype=torch.float32)
                loss = weighted_pass_loss(logits, targets) / args.accumulate
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Nonfinite loss; inspect B hidden states before normalization changes"
                    )
                loss.backward()
                batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            record = {
                "step": step + 1,
                "K": K,
                "loss": batch_loss,
                "elapsed_s": perf_counter() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    adapted.save_pretrained(args.output / "adapter")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
        args.output / "head.safetensors",
    )
    tokenizer.save_pretrained(args.output / "tokenizer")
    metadata = {
        "stage": args.stage,
        "base_model": OURO_ID,
        "base_revision": OURO_REVISION,
        "code_ids": list(code_ids),
        "codes": CODE_CANDIDATES,
        "calibrated": False,
        "training_manifest": manifest,
        "steps": args.steps,
        "seed": args.seed,
        "learning_rate": args.lr,
        "accumulate": args.accumulate,
        "init": str(args.init) if args.init else None,
        "elapsed_s": perf_counter() - started,
        "note": "Pilot trainer; not a completed A/B/C experiment.",
    }
    (args.output / "checkpoint.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
