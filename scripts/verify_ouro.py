"""Explicit CPU source-equivalence check using tiny random weights, not a model result.

Run after source audit: .venv-train/bin/python scripts/verify_ouro.py
No pretrained weights are loaded. No benchmark data is used.
"""

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from looped_decision.ouro import OuroDecisionModel, weighted_pass_loss


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / ".cache/upstream/ByteDance--Ouro-1.4B"
    config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.vocab_size = 128
    config.layer_types = ["full_attention"] * 2
    config.total_ut_steps = 4
    torch.manual_seed(238)
    lm = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation="eager"
    ).eval()
    runner = OuroDecisionModel(lm, range(64)).eval()
    state, q1, q2 = [1, 2, 3], [4, 5], [6, 7, 8]
    errors = []
    with torch.no_grad():
        _, official, _ = lm.model(input_ids=torch.tensor([state + q1]), use_cache=False)
        full = list(runner.full_passes(state + q1))
        shared = list(runner.shared_passes(state, [q1, q2], max_passes=4))
        full2 = list(runner.full_passes(state + q2))
        for k in range(4):
            torch.testing.assert_close(full[k][0], official[k], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(
                shared[k].question_hidden[0], full[k][0][:, len(state) :], atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                shared[k].question_hidden[1], full2[k][0][:, len(state) :], atol=2e-6, rtol=2e-5
            )
            errors.append(float((full[k][0] - official[k]).abs().max()))
        extended = list(runner.shared_passes(state, [q1], max_passes=8, stage="B"))
        for k in range(4):
            torch.testing.assert_close(extended[k].logits[0], shared[k].logits[0])
        frozen = extended[3].prefix_cache
        for step in extended[4:]:
            assert step.prefix_cache is frozen
            assert step.state_hidden is extended[3].state_hidden
            assert torch.isfinite(step.logits[0]).all()
            assert all(key.shape[2] == len(state) for key, _ in frozen.entries.values())

    # Ensure per-pass soft-target supervision differentiates through shared loops.
    runner.train()
    steps = list(runner.shared_passes(state, [q1], max_passes=8, stage="B"))
    logits = torch.stack([step.logits[0][:, :2] for step in steps])
    loss = weighted_pass_loss(logits, torch.tensor([[0.7, 0.3]]))
    loss.backward()
    assert runner.head.weight.grad is not None
    assert torch.isfinite(runner.head.weight.grad).all()
    assert runner.backbone.layers[0].self_attn.q_proj.weight.grad is not None
    result = {
        "kind": "tiny-random-weight CPU correctness test; NOT model evaluation",
        "source_revision": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        "max_abs_errors_per_pass": errors,
        "shared_prefix_equivalence": True,
        "B_first_four_equal_A": True,
        "B_prefix_four_immutable": True,
        "soft_loss_backprop": True,
    }
    out = root / "results/cpu-equivalence.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
