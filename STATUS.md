# Build status — initial implementation checkpoint

## Verified locally

- [x] Public GitHub repository created; dedicated VS Code workspace opened.
- [x] Real upstream Jev/Laya schema inspected and pinned. Score outputs have legend, confidence and string-indexed probabilities; usage has token counts.
- [x] Ouro source/config revision pinned; tokenizer code boundary verified for 64 codes.
- [x] Full-pass runner equals upstream on tiny random float32 weights.
- [x] Shared-state prefix computation equals independent concatenated questions through pass 4.
- [x] B freezes pass-four prefix cache and advances question hidden states only; finite tiny-model output through pass 8.
- [x] Per-pass soft-label loss, rank-64 LoRA-only backbone training and external head gradients smoke-tested.
- [x] Calibration/statistics/data hygiene/API tests pass (17 tests).
- [x] Separate local and training environments; model weights never loaded on laptop.

## Needed before first honest trained-stage result

- [ ] Resolve original SST5 training/release license; review attribution/share-alike requirements for ARC/BoolQ.
- [ ] Implement/audit all real dataset adapters, grouped train/calibration manifest, synthetic 10% mixture and quarantined full evaluation 13-gram index.
- [ ] Verify provider-side termination deadline. Installed `runpodctl 2.12.0` live help lacks the skill guide's `--terminate-after` flag. Do not invoke a nonexistent flag or provision with an unverified cost guard.
- [ ] Rent one A100 SXM 80 GB (catalog secure quote $1.79/GPU-hour; recheck stock/rate), with bounded first lease and all charges inside $100 total. No resource provisioned yet; project spend $0.
- [ ] Run pretrained BF16/source equivalence, memory/gradient checks, then an explicitly labelled training pilot.
- [ ] Train A, capture heldout per-pass logits, fit calibration and threshold, validate policy after refit.
- [ ] Train B initialized from A and repeat calibration.
- [ ] Implement active-question compaction, length/active-count CUDA graph buckets and replay correctness tests. Current eager reference must not stand in for final speed.
- [ ] Official DecisionBench 1.0 and BBH runs, same-B adaptive max8/max4 paired gate, measured speed/cost.
- [ ] Select/train C, recalibrate; only then run JevBench and same-GPU author-served references.
- [ ] Conditional public HF checkpoint/card and official submission (request submitter details when eligible).

## Interpretation rules

Nothing in the CPU checks establishes that looping improves decision quality. No model is calibrated yet. A 503 from the local API while no checkpoint is loaded is intentional. No benchmark result, trained weight artifact, or speed superiority is claimed.

The $100 limit is a hard project authorization despite account auto-reload. The initial target rate ceiling is $2/GPU-hour, with storage/setup also charged against total budget; seek approval before increasing the project total.
