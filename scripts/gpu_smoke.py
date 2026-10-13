"""Real pretrained Ouro GPU correctness smoke. No benchmark and no optimizer steps."""

import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from looped_decision.ouro import (
    OURO_ID,
    OURO_REVISION,
    OuroDecisionModel,
    attach_lora,
    weighted_pass_loss,
)
from looped_decision.prompting import verify_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU required; never load pretrained weights on laptop")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(OURO_ID, revision=OURO_REVISION)
    code_ids = verify_codes(tokenizer)
    config = AutoConfig.from_pretrained(OURO_ID, revision=OURO_REVISION, trust_remote_code=True)
    config.total_ut_steps = 4
    config.early_exit_threshold = 1.0
    print("LOADING_PRETRAINED", flush=True)
    lm = AutoModelForCausalLM.from_pretrained(
        OURO_ID,
        revision=OURO_REVISION,
        code_revision=OURO_REVISION,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()
    model = OuroDecisionModel(lm, code_ids).eval()
    # Synthetic only: no evaluation-set content.
    state = tokenizer.encode(
        "Inventory: apples=8. Sold 3 apples. Received 2 apples.", add_special_tokens=False
    )
    questions = [
        tokenizer.encode(q, add_special_tokens=False)
        for q in [
            "\n\nQuestion: How many apples remain?\n(A) 7\n(B) 5\nAnswer: (",
            "\n\nQuestion: Are there more than six apples?\n(A) false\n(B) true\nAnswer: (",
        ]
    ]
    result = {
        "kind": "pretrained BF16 correctness smoke, NOT benchmark/quality/speed evaluation",
        "base_revision": OURO_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "code_ids_verified": len(code_ids),
    }
    with torch.no_grad():
        ids = torch.tensor([state + questions[0]], device="cuda")
        _, official, _ = lm.model(input_ids=ids, use_cache=False)
        full = list(model.full_passes(state + questions[0]))
        result["full_upstream_max_abs"] = []
        for k in range(4):
            delta = float((official[k].float() - full[k][0].float()).abs().max())
            result["full_upstream_max_abs"].append(delta)
            # Native SDPA may choose causal fastpath vs explicit-mask kernels.
            a = model.head(full[k][0][:, -1]).float().softmax(-1)
            b = model.head(official[k][:, -1]).float().softmax(-1)
            torch.testing.assert_close(a, b, atol=0.01, rtol=0.02)
        shared = list(model.shared_passes(state, questions, max_passes=4))
        independent = [full, list(model.full_passes(state + questions[1]))]
        result["prefix_probability_max_abs"] = []
        for k in range(4):
            for j in range(2):
                a = shared[k].logits[j][:, :2].float().softmax(-1)
                b = independent[j][k][1][:, :2].float().softmax(-1)
                result["prefix_probability_max_abs"].append(float((a - b).abs().max()))
                # Predeclared numerical tolerance for changed BF16 GEMM shapes.
                torch.testing.assert_close(a, b, atol=0.01, rtol=0.02)
        extended = list(model.shared_passes(state, questions, max_passes=8, stage="B"))
        result["B_hidden_rms"] = []
        for k, step in enumerate(extended):
            result["B_hidden_rms"].append(
                [float(q.float().square().mean().sqrt()) for q in step.question_hidden]
            )
            assert all(torch.isfinite(q).all() for q in step.question_hidden)
            if k < 4:
                for a, b in zip(step.logits, shared[k].logits):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            else:
                assert step.prefix_cache is extended[3].prefix_cache
                assert step.state_hidden is extended[3].state_hidden
        result["B_first4_equal_A"] = True
        result["B_frozen_prefix4"] = True
    del official, full, shared, independent, extended, model
    torch.cuda.empty_cache()
    print("FORWARD_CHECKS_PASSED; CHECKING_LORA_GRADIENTS", flush=True)
    adapted = attach_lora(lm)
    model = OuroDecisionModel(adapted.get_base_model(), code_ids).train()
    outputs = list(model.shared_passes(state, questions[:1], max_passes=4, stage="A"))
    logits = torch.stack([step.logits[0][:, :2] for step in outputs])
    loss = weighted_pass_loss(logits, torch.tensor([[0.8, 0.2]], device="cuda"))
    loss.backward()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n or n.startswith("head.") for n in names)
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    assert any(
        "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
    )
    result.update(
        {
            "lora_and_head_backward": True,
            "optimizer_steps": 0,
            "smoke_loss": float(loss.detach()),
            "max_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds_including_load": time.perf_counter() - started,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("GPU_SMOKE_PASSED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
