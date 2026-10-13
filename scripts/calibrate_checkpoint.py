"""Collect ONLY grouped heldout logits and calibrate a checkpoint; no benchmarks.

Raw trajectories/targets stay on the private RunPod volume. Public summary metrics
are calibration-set diagnostics, not independent validation or quality claims.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from looped_decision.calibration import Trajectory, calibrate, softmax
from looped_decision.data import DecisionExample
from looped_decision.ouro import OURO_ID, OURO_REVISION, OuroDecisionModel
from looped_decision.prompting import prepare_example, verify_codes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Real weights are GPU-only")
    if args.output.exists():
        raise SystemExit("Use a fresh calibration output")
    metadata = json.loads((args.checkpoint / "checkpoint.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") != "heldout" or manifest.get("group_split_verified") is not True:
        raise SystemExit("Require grouped heldout split")
    if hashlib.sha256(args.heldout.read_bytes()).hexdigest() != manifest["data_sha256"]:
        raise SystemExit("Heldout checksum mismatch")
    if (
        metadata["training_manifest"]["evaluation_index"]["index_sha256"]
        != manifest["evaluation_index"]["index_sha256"]
    ):
        raise SystemExit("Train/heldout provenance differs")
    if metadata["base_revision"] != OURO_REVISION:
        raise SystemExit("Wrong base revision")
    tokenizer = AutoTokenizer.from_pretrained(OURO_ID, revision=OURO_REVISION)
    codes = verify_codes(tokenizer)
    if list(codes) != metadata["code_ids"]:
        raise SystemExit("Tokenizer codes differ")
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
    adapted = PeftModel.from_pretrained(lm, args.checkpoint / "adapter", is_trainable=False)
    model = OuroDecisionModel(adapted.get_base_model(), codes).eval()
    model.head.load_state_dict(load_file(args.checkpoint / "head.safetensors"))
    cap = 4 if metadata["stage"] == "A" else 8
    args.output.mkdir(parents=True)
    rows = []
    started = time.perf_counter()
    with torch.inference_mode(), (args.output / "private-trajectories.jsonl").open("w") as out:
        for line in args.heldout.read_text().splitlines():
            raw = json.loads(line)
            for key in ["options", "targets", "descriptions"]:
                raw[key] = tuple(raw.get(key, ()))
            item = DecisionExample(**raw)
            state, question = prepare_example(tokenizer, item)
            logits = []
            for step in model.shared_passes(
                state, [question], max_passes=cap, stage=metadata["stage"]
            ):
                logits.append(step.logits[0][0, : len(item.options)].float().cpu().numpy())
            trajectory = Trajectory(item.id, item.kind, np.stack(logits), np.array(item.targets))
            rows.append(trajectory)
            out.write(
                json.dumps(
                    {
                        "id": item.id,
                        "group_id": item.group_id,
                        "kind": item.kind,
                        "logits": trajectory.logits.tolist(),
                        "target": item.targets,
                    }
                )
                + "\n"
            )
            if len(rows) % 100 == 0:
                out.flush()
                print(
                    json.dumps(
                        {
                            "heldout_rows_collected": len(rows),
                            "elapsed_s": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    profile = calibrate(rows, cap)
    profile["temperatures"] = {f"{kind}:{k}": v for (kind, k), v in profile["temperatures"].items()}
    profile["selected_counts"] = {
        f"{kind}:{k}": v for (kind, k), v in profile["selected_counts"].items()
    }
    full_accuracy = float(
        np.mean([softmax(r.logits[cap - 1]).argmax() == r.target.argmax() for r in rows])
    )
    report = {
        "kind": "Noncommercial PILOT calibration-set diagnostics; NOT independent evaluation",
        "checkpoint": args.checkpoint.name,
        "base_revision": OURO_REVISION,
        "stage": metadata["stage"],
        "optimizer_steps": metadata["steps"],
        "heldout_rows": len(rows),
        "heldout_sha256": manifest["data_sha256"],
        "full_pass_accuracy": full_accuracy,
        "meets_1pp_after_refit": profile["after_refit"]["accuracy"] + 1e-12 >= full_accuracy - 0.01,
        "calibration": profile,
        "elapsed_s": time.perf_counter() - started,
        "note": "Threshold minimizes average passes, not measured latency. No speed claim. Temperatures/threshold fitted on these same heldout rows; benchmark untouched.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PILOT_CALIBRATION_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
