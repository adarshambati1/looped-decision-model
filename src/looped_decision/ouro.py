"""Correctness-first Ouro pass runner, NOT a CUDA-graph optimized serving engine.

Uses pinned upstream decoder layers, including their native end-of-pass norm.
The external decision head is read-only with respect to recurrent hidden states.
"""

from dataclasses import dataclass

import torch
from torch import nn

OURO_ID = "ByteDance/Ouro-1.4B"
OURO_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"


class PassCache:
    """One pass's layerwise KV; question suffix replaces itself each iteration.

    Prefix KV tensors are never mutated or appended to. Keeping their autograd
    history permits training B to update A's pass-four prefix representations.
    """

    def __init__(self, layers, prefix=None):
        self.layers = layers
        self.prefix = prefix
        self.entries = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        layer = layer_idx % self.layers
        if layer in self.entries:
            raise RuntimeError("A layer was evaluated twice in the same pass cache")
        self.entries[layer] = (key, value)
        if self.prefix is not None:
            pk, pv = self.prefix.entries[layer]
            # expand is a view: a prefix computed once can serve a question batch.
            key = torch.cat([pk.expand(key.shape[0], -1, -1, -1), key], dim=2)
            value = torch.cat([pv.expand(value.shape[0], -1, -1, -1), value], dim=2)
        return key, value


@dataclass
class SharedPass:
    pass_number: int
    state_hidden: torch.Tensor
    question_hidden: list[torch.Tensor]
    logits: list[torch.Tensor]
    prefix_cache: PassCache


class OuroDecisionModel(nn.Module):
    def __init__(self, causal_lm, code_ids):
        super().__init__()
        if len(code_ids) != 64 or len(set(code_ids)) != 64:
            raise ValueError("Need 64 verified distinct code IDs")
        self.backbone = causal_lm.model
        self.config = causal_lm.config
        self.config.early_exit_threshold = 1.0
        causal_lm.early_exit_threshold = 1.0
        weights = causal_lm.get_output_embeddings().weight
        ids = torch.tensor(code_ids, device=weights.device)
        self.head = nn.Linear(
            weights.shape[1], len(code_ids), bias=False, device=weights.device, dtype=weights.dtype
        )
        with torch.no_grad():
            self.head.weight.copy_(weights.index_select(0, ids))
        self.code_ids = tuple(code_ids)

    def embed(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.head.weight.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return self.backbone.embed_tokens(ids)

    def pass_once(self, hidden, pass_number, prefix_cache=None, prefix_length=0):
        """Execute one shared stack; query positions stay fixed across passes."""
        n = hidden.shape[1]
        positions = torch.arange(prefix_length, prefix_length + n, device=hidden.device)
        key_positions = torch.arange(prefix_length + n, device=hidden.device)
        allowed = key_positions[None, :] <= positions[:, None]
        mask = torch.zeros((n, prefix_length + n), device=hidden.device, dtype=hidden.dtype)
        mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        mask = mask[None, None, :, :]
        rotary = self.backbone.rotary_emb(hidden, positions[None, :])
        cache = PassCache(len(self.backbone.layers), prefix_cache)
        for layer in self.backbone.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions[None, :],
                past_key_value=cache,
                use_cache=True,
                cache_position=positions,
                position_embeddings=rotary,
                current_ut=pass_number - 1,
            )
        # Upstream feeds this normalized tensor back, not the pre-norm tensor.
        hidden = self.backbone.norm(hidden)
        return hidden, cache

    def full_passes(self, ids, max_passes=4):
        if not 1 <= max_passes <= 4:
            raise ValueError("Whole-sequence A is limited to four passes")
        hidden = self.embed(ids)
        self.config.total_ut_steps = max_passes
        self.backbone.total_ut_steps = max_passes
        for k in range(1, max_passes + 1):
            hidden, cache = self.pass_once(hidden, k)
            yield hidden, self.head(hidden[:, -1]), cache

    def shared_passes(self, state_ids, question_ids, max_passes=4, stage="A"):
        """Reference prefix sharing. Ragged questions currently execute separately.

        Caller may stop consuming this generator for real early exit. Individual
        question compaction/CUDA graphs are intentionally not claimed here.
        """
        if stage not in {"A", "B"} or not 1 <= max_passes <= (4 if stage == "A" else 8):
            raise ValueError("Invalid stage or maximum pass count")
        if not question_ids or any(len(ids) == 0 for ids in question_ids):
            raise ValueError("Questions must have at least one token")
        state = self.embed(state_ids)
        questions = [self.embed(ids) for ids in question_ids]
        prefix = None
        for k in range(1, max_passes + 1):
            if k <= 4:
                if len(state_ids):
                    state, prefix = self.pass_once(state, k)
                else:
                    prefix = None
            # For k>4, prefix remains the exact state-pass-four layerwise KV.
            questions = [self.pass_once(q, k, prefix, len(state_ids))[0] for q in questions]
            logits = [self.head(q[:, -1]) for q in questions]
            yield SharedPass(k, state, questions, logits, prefix)


def weighted_pass_loss(logits_by_pass, targets):
    """Cross entropy with SOFT targets, pass weights proportional to k.

    logits_by_pass: [K, batch, options]; targets: [batch, options]. Mask/select
    unused codes before calling, and remap targets when shuffling options.
    """
    if logits_by_pass.ndim != 3 or targets.shape != logits_by_pass.shape[1:]:
        raise ValueError("Invalid pass-logit/target shapes")
    if not torch.isfinite(targets).all() or torch.any(targets < 0):
        raise ValueError("Invalid target probabilities")
    if not torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1))):
        raise ValueError("Targets must sum to one")
    losses = -(targets[None] * logits_by_pass.float().log_softmax(-1)).sum(-1).mean(-1)
    weights = torch.arange(1, len(losses) + 1, device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum()


def attach_lora(causal_lm):
    """Freeze base, adapt shared attention and MLP projections at rank 64."""
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        causal_lm,
        LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
