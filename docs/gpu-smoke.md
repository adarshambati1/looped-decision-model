# Pretrained GPU smoke — 2026-10-13

Artifact: `results/gpu-smoke.json`; reproduction: `scripts/gpu_smoke.py`.

- Real pinned ByteDance/Ouro-1.4B, BF16, A100 PCIe 80 GB.
- Isolated environment: Torch 2.7.1+cu126, Transformers 4.54.1.
- Full-sequence hidden states matched native upstream exactly across all four passes (maximum absolute errors `[0, 0, 0, 0]`).
- Shared-prefix two-option probabilities matched independent questions exactly on both synthetic test questions, every pass.
- B first four passes matched A. Pass-four prefix/state objects remained fixed for passes 5–8; question hidden states stayed finite.
- Rank-64 LoRA and external decision head backward checks passed. No optimizer steps, trained checkpoint, calibration, or benchmark evaluation.
- Peak PyTorch allocated GPU memory was 3,990,736,896 bytes **for this short smoke input only**. This is not a production context/batch capacity measurement.
- BF16 comparison tolerances were set before execution (`atol=.01, rtol=.02` on probability comparisons); observed differences were zero. This does not prove equivalence for every sequence length or question count.

## Resource lifecycle

A100 SXM was unavailable, so the same-price PCIe 80 GB was used at $1.79/GPU-hour. Results/logs were copied back and the pod was explicitly deleted; subsequent pod listing was empty.

A detached in-pod watchdog was armed after SSH setup and authenticated to the exact pod. It is **not** a provider control-plane scheduler and is vulnerable to container failure. This session was supervised and manually cleaned up before the deadline; the watchdog's timed deletion path was not live-tested. Future unattended jobs need an additional independent cleanup mechanism.

Provider billing had not posted at the first query. `infra/budget.json` reserves $2 conservatively for this session rather than fabricating an exact charge. The total authorization remains $100.
