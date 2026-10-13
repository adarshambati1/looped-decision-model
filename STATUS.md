# Build status — first real training pilot

Updated 2026-10-13: pretrained GPU checks passed; A pilot completed 100 optimizer steps and heldout calibration. The post-refit stopping policy **failed** the 1-percentage-point tolerance (1.45-point drop). Automatic completion cleanup deleted the GPU; checkpoint stays on a private RunPod volume. This is **not** completion of the requested A/B/C experiment.

## Verified locally

- [x] Public GitHub repository created; dedicated VS Code workspace opened.
- [x] Real upstream Jev/Laya schema inspected and pinned. Score outputs have legend, confidence and string-indexed probabilities; usage has token counts.
- [x] Ouro source/config revision pinned; tokenizer code boundary verified for 64 codes.
- [x] Full-pass runner equals upstream on tiny random float32 weights.
- [x] Shared-state prefix computation equals independent concatenated questions through pass 4.
- [x] B freezes pass-four prefix cache and advances question hidden states only; finite tiny-model output through pass 8.
- [x] Per-pass soft-label loss, rank-64 LoRA-only backbone training and external head gradients smoke-tested.
- [x] Calibration/statistics/data hygiene/API/adapter/watchdog tests pass (27 tests). Exact threshold sweep agrees with brute-force selection.
- [x] Separate local and training environments; model weights never loaded on laptop.

## Real GPU/data progress

- [x] Pretrained BF16 smoke on A100 PCIe 80 GB: full passes matched upstream exactly; shared-prefix checks, B cache checks and LoRA/head gradients passed. `results/gpu-smoke.json`.
- [x] Quarantined public input index: 23,900 canonical DecisionBench rows, 6,511 BBH rows and all pinned public JevBench tiers. No benchmark scores/labels exported.
- [x] Pilot mixture: 29,995 training / 1,587 grouped heldout examples, ~10% synthetic; 114 overlap-related examples removed. `results/pilot-data-summary.json`.
- [x] A pilot trained 100 steps on A100 SXM 80 GB; checkpoint retained privately on RunPod. Public metadata/loss log: `results/A-pilot-100/`.
- [x] Pilot heldout calibration completed. Full depth: 69.69%; stopped before refit: 68.75%; stopped after refit: 68.24%. These are **calibration-set** diagnostics, not benchmark results. Post-refit tolerance failed; no promotion/retuning to disguise it. `results/A-pilot-100/calibration.json`.

## Needed before a completed trained-stage result

- [ ] Resolve SST5 license and ARC's conflicting original/HF terms before full-mixture/model-release claims. BoolQ original CC-BY-SA-3.0 notice verified. Pilot explicitly excludes SST5 and is noncommercial research only.
- [ ] Extend real-weight correctness/memory checks beyond the short smoke to long contexts and full question-count range.
- [x] Run bounded A100 sessions at $1.79/GPU-hour. Live CLI lacks provider scheduling; independently armed in-pod and local watchdogs are used instead, plus completion-triggered cleanup. They are not a provider control-plane guarantee.
- [ ] Reconcile actual provider billing. $6 conservatively reserved (not actual charges); $100 total remains hard. Both GPUs deleted and pod list empty; private 10 GB checkpoint volume incurs ongoing storage (~$0.70/month).
- [ ] Train A, capture heldout per-pass logits, fit calibration and threshold, validate policy after refit.
- [ ] Train B initialized from A and repeat calibration.
- [ ] Implement active-question compaction, length/active-count CUDA graph buckets and replay correctness tests. Current eager reference must not stand in for final speed.
- [ ] Official DecisionBench 1.0 and BBH runs, same-B adaptive max8/max4 paired gate, measured speed/cost.
- [ ] Select/train C, recalibrate; only then run JevBench and same-GPU author-served references.
- [ ] Conditional public HF checkpoint/card and official submission (request submitter details when eligible).

## Interpretation rules

Neither correctness checks nor 100 training steps establish that looping improves decision quality. Pilot calibration is not a benchmark result. A 503 from the local API while no checkpoint is loaded is intentional. A private pilot checkpoint exists. No completed-stage benchmark result, release clearance, or speed superiority is claimed.

The $100 limit is a hard project authorization despite account auto-reload. The initial target rate ceiling is $2/GPU-hour, with storage/setup also charged against total budget; seek approval before increasing the project total.
