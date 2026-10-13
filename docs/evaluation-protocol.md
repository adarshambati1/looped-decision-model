# Preregistered evaluation protocol

## Verified sources

- DecisionBench: [Hanno-Labs/decision-bench](https://github.com/Hanno-Labs/decision-bench/tree/89aaa85827912f8975c1835348ead7436710f581), commit `89aaa85827912f8975c1835348ead7436710f581`. README confirms 23,900 rows, nine families, three primitives. `task_specs/decisionbench-dev.toml` specifies version 1.0. Its official CLI provides `run-system-one-http`; use this adapter, not a replacement scorer. The linked HF Space is `Hanno-Labs/decision-bench-leaderboard`, revision recorded in `results/upstream-metadata.json`. This is NOT the unrelated OpenMesh/decisionbench.
- JevBench source commit: `7625ecb1b075cfe4d09d34141d353325825eea1d` at `fstandhartinger/jevbench`. Only repository metadata/tree has been inspected: **no examples loaded and no evaluation run**. Final-stage adapter/aggregate-metric verification is still needed. Do not equate simple accuracy with capability.
- BBH: `lukaemon/bbh` revision in metadata. Only its dataset card has been retrieved. No task examples evaluated.

## Locked roles

Train weights on train split only. Group-related records before a seeded 95/5 split. The 5% is calibration/validation, not gradient data. Preserve soft labels. Keep a quarantined evaluation-content process that produces 13-word-gram hashes for contamination exclusion; no evaluation labels, scores or examples enter training/selection logs. Final JevBench content indexing must be isolated from model-development inspection; it is not an evaluation run. Record precisely what was accessed and when.

## Calibration

1. Fit one soft-label NLL temperature per question type and pass on heldout trajectories.
2. Choose global tau minimizing mean executed passes subject to heldout top-label accuracy within 0.01 of always running maximum. Mean passes is a compute proxy, not measured GPU latency. Record hard-label accuracy plus soft NLL/Brier; official benchmark metrics remain authoritative.
3. Refit each type/pass temperature once on the selected stopping population. Retain the initial temperature where no observations stopped. Freeze tau. Report achieved accuracy and mean passes after refit because the stopping distribution can shift again; do not silently assert the pre-refit guarantee still holds.

The threshold checks calibrated maximum probability, not the API normalized confidence score.

## A vs B

- Record A at K=1,2,3,4 and adaptive max4. B starts from A adapters/head and trains with K uniform in 1..8; A K uniform in 1..4. Loss weight proportional to pass number.
- Primary gate: **same B checkpoint**, same final calibration temperatures, same tau; adaptive max8 vs adaptive max4. Reuse item IDs and input preprocessing. No threshold tuning or model selection on BBH.
- Use official per-item DecisionBench correctness, including failures/unsupported items as incorrect. Paired bootstrap 10,000 resamples, seed 238, percentile 95% CI. Equal item sets are mandatory; never silently intersect successful items.
- If lower CI bound >0, select B for C. Otherwise select A. One pass is descriptive only, never the claimed benefit baseline.

## BBH reporting only

Use exactly tracking_shuffled_objects and logical_deduction with three, five and seven objects (six configs). Split each input at `Options:`: prefix is state; parse labeled options into a choice request with instruction `Which option is correct?`. Assert complete option parsing and gold-label mapping. Pool tasks by object count. Report A K1..4; B K1,2,4,8; adaptive accuracy and average passes. Parsing implementation and examples remain pending.

## C and references

Refit all temperatures and tau for new C weights. A maximum stays four. B maximum may grow only from heldout evidence, with the decision recorded before benchmark evaluation.

References verified by model card, not remeasured:
- `h2oai/h2o-lightning-4b`, revision `820443c99919dae2876e71a20757060360ebae86`. Author specifies **stock vLLM 0.30.0 plus the supplied h2o_lightning_shim.py and serve_config.json**. Optional throughput patches exist; keep disabled for the user's stock-vLLM baseline. The card references the v1.1 model and newer serving-release versions; verify the weight/serving version relationship and disclose exact revisions rather than conflating these labels. This is not `quelmap/Lightning-4b`.
- `chinhnc/Quyet-1.0-Large`, revision `9a0f051112ba4301f6e4c8b87fc088ce00b11aae`: card says 31.3B, 62.5 GB BF16, author `quyet` package, prompt version 2 and shipped temperatures. Native limit **10 options**, not our 64. Unsupported questions must be reported, never quietly dropped. Pin the package version after audit. Author card claims are not independent leaderboard verification.

JevBench capability is the official aggregate of intelligence and calibration. During bootstrap **recompute the official aggregate on each resample**; do not average invented per-item capability values. `evaluation.paired_bootstrap` accepts a metric callback for this reason. Local 231-item comparison is a preview, not a sealed-board verdict.

## Timing and cost

Use the same exact 80 GB GPU SKU for A, B, C, and both references, sequential deployments. Warm up on separate synthetic requests, synchronize CUDA, then time one request at a time with persistent HTTP connection. Record request latency and amortized per-decision latency separately when multiple questions share a request. Report p50, mean, input lengths, per-item passes and question counts. Final loop engine must have active-question compaction and verified CUDA graphs; the current eager reference runner is not eligible for final speed claims.

Cost per 1,000 = measured total serial walltime / total decisions * actual hourly rate * 1000 / 3600. Do not derive cost from p50. Separate warm serving compute from download, setup, training, idle and storage charges.

## Publication gates

- Update README for each completed stage with raw artifact hashes, source/model revisions and plots. No synthetic/example outcomes masquerading as results.
- Publish HF adapter/head/calibration/policy bundle if B extra passes help OR C preview passes.
- C preview: faster p50 than H2O AND capability-gap upper CI bound >=0 (not clearly worse). This is weaker than proving superior capability; label it accurately.
- If preview passes, prepare official Benchmark Heaven submission and ask user for submitter details. Do not submit prematurely.
