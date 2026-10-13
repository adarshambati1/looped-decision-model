# Project invariants

- Scope: the user's A/B/C looped decision experiment. Do not fabricate results or present toy inference as Ouro.
- $100 total project RunPod budget authorized initially. Auto-reload does NOT authorize exceeding it. One 80 GB GPU at a time; quote rate and use bounded lifetime before launch. Persist artifacts before teardown; track storage costs too.
- Public repository is authorized; credentials, raw private requests, private paths, and gated data are not. Check staged changes before each push.
- Read docs/contract.md before API changes. Exact upstream Jev/Laya wire format wins over the sketch. Keep diagnostic timing/pass counts separate unless supported by upstream.
- Pin Ouro remote-code revision and transformers==4.54.1. No model-weight downloads on the laptop. Separate train, reference vLLM, and harness environments.
- Every pass uses shared stack weights and returns an independently normalized readout. The decision head must never feed back into the loop.
- B is identical to A through pass 4; subsequent question-only passes use immutable layerwise state-prefix KV from pass 4. Verify numerical equivalence before performance tuning.
- Soft target distributions stay soft; remap choice/noul labels with option permutations. Score levels never shuffle.
- Split related states/groups before augmentation. Calibration data is never used for weight updates. Index evaluation 13-grams in a quarantined process; no JevBench scoring until final C.
- Calibration: per type/pass temperature -> heldout threshold -> stopped-population temperature refit once. Record policy change after refit; never choose thresholds from benchmark scores.
- DecisionBench gate compares same B weights, same calibration and threshold, adaptive max8 versus max4, paired item resampling. Require CI lower bound >0 for B to win.
- Distinguish C preview gate from confirmed superiority; absence of significant inferiority is not proof of superiority.
- No GPU provisioning until source audits and CPU tests pass. Stop and report unavailable official artifacts instead of substituting unrelated benchmarks.
