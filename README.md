# looped-decision-model

Can adaptive latent looping make a decision model both fast and smart?

**Status: real pretrained GPU checks passed; a 100-step A research pilot trained and calibrated. Its post-refit stopping policy failed the accuracy-drop tolerance (1.45 points vs 1 allowed). Both GPUs were released. No completed A/B/C experiment, benchmark result, serving-speed measurement, or superiority claim.**

## Experiment

- **A / Frankenstein:** Ouro-1.4B's shared stack advances state and question through at most four passes; an external code head reads each pass after final norm.
- **B / question-only extension:** same first four passes, then up to four question-only passes reading the immutable pass-four state-prefix KV. Initialize from trained A.
- **C:** train the winner. B wins only when the paired 95% bootstrap CI for DecisionBench adaptive max8 minus max4 accuracy is entirely positive, comparing the **same B weights and threshold**. Otherwise use A.

Exact Jev/Laya compatibility, soft-label supervision, calibrated stopping, no benchmark tuning, and same-GPU reference timing are required, not optional follow-ups.

## Review while building

- `docs/contract.md` — verified upstream wire contract and explicit project limits.
- `docs/ouro-audit.md` — verified loop/cache internals and source revisions.
- `STATUS.md` — what works, what remains, and blockers.
- `docs/evaluation-protocol.md` — stage gates and official harness provenance.
- `docs/data-sources.md` — licenses/schema audit, including SST5 exclusion and ARC terms conflict.
- `docs/gpu-smoke.md` — real pretrained BF16 correctness checks.
- [`results/A-pilot-100/`](results/A-pilot-100/) — actual pilot metadata and loss plot (weights remain private on RunPod).
- `AGENTS.md` — experimental and spending invariants.

## Local development (no model weights)

```bash
uv sync --extra dev --frozen
uv run --no-sync pytest -q
```

Training pins live separately from reference serving and benchmark harness dependencies. GPU work runs on RunPod, not the laptop. The initial authorized budget is **$100 total**, with a bounded GPU lifetime and no implicit authorization from account auto-reload.

## Implemented and tested (not model-performance results)

- Exact Jev answer shapes, including score legends/string-keyed probabilities and token-count usage; no fake fallback inference.
- NumPy soft-label temperature fitting, calibrated pmax stopping, one selected-population refit, paired bootstrap and publication gates.
- Pinned real dataset adapters, description-preserving prompts, group-safe splits, quarantined public evaluation 13-gram index, soft-label option permutations and deterministic multihop/balance generators. Pilot: 29,995 training / 1,587 heldout examples, ~10% synthetic; SST5 excluded explicitly.
- Pinned Ouro pass executor and immutable pass-four state cache for B. Head initialized from code embeddings; all 64 contextual codes verified (`A-Z`, `a-z`, `0-9`, `!`, `?`; `+` fails tokenization).
- **27 unit tests pass.** Tiny random-weight CPU execution and a short real pretrained BF16 GPU input match upstream at all four passes (observed max absolute error 0.0). See `results/cpu-equivalence.json` and `results/gpu-smoke.json`.
- Rank-64 LoRA/head pilot trained for 100 optimizer steps on A100 SXM 80 GB. `scripts/train.py` requires a checksum-matched, reviewed data manifest. This noncommercial research pilot is not the full requested training mixture or a release-cleared model.

The current model runner is correctness-first: final batched compaction, CUDA graphs, full-mixture rights clearance/training, qualifying final calibration, and official harness runs remain outstanding. No pretrained weights were downloaded to the laptop.

## Stage results

| Stage | Status | DecisionBench | BBH | p50 / cost | Artifacts |
|---|---|---|---|---|---|
| A | Pilot trained/calibrated; post-refit tolerance failed | Not run | Not run | Not measured | [Pilot log/plot](results/A-pilot-100/) |
| B | Not trained | Not run | Not run | Not measured | None |
| C | Not selected | Not run | Not run | Not measured | None |

JevBench is reserved for final C. No local preview establishes the official leaderboard verdict. Public model publication and official submission remain conditional on the user's gates.

No training data, model weights, or third-party code are relicensed by this repository. Upstream licenses must be verified before training and release.
