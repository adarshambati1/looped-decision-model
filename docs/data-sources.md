# Data provenance and license audit

Research-pilot review: `results/data-license-review.json`. This is not blanket permission to release trained weights.

Exact metadata/card revisions and hashes: `results/upstream-metadata.json`.

| Dataset | Intended mapping | Metadata license | Status |
|---|---|---|---|
| LocalLLaMA/typed-decisions | config `all`, train; expand questions sharing one state/group; keep distributions | Apache-2.0 | Adapter implemented: `all` train only, group shared states, preserve descriptions and soft distributions. Normalize only the source's six-decimal probability rounding. |
| allenai/ai2_arc | ARC-Easy + ARC-Challenge train, choice | CC-BY-SA-4.0 | Adapter implemented. **Original AI2 archive README says non-commercial research only, no redistribution**, conflicting with HF metadata. Research pilot only; resolve rights before model release. Do not redistribute raw data. |
| tau/commonsense_qa | train, choice | MIT | Adapter implemented; retain attribution. |
| google/boolq | train passage as state, question as noul | CC-BY-SA-3.0 | Original Google repository README confirms CC-BY-SA-3.0. Adapter preserves passage grouping; research pilot allowed with attribution/notices. No raw redistribution. |
| SetFit/sst5 | train, score levels very negative through very positive | **Not declared in fetched HF card metadata** | **License unresolved.** Original Stanford page and source repository do not declare it; Stanford HF cards say unknown. Adapter implemented/tested, but pilot ingest explicitly excludes SST5. No claim of a complete A/B mixture. |
| Generated multihop/balance | 10% of training examples for A/B; larger for C | Original generated code/data | Deterministic generators and correctness tests implemented. Record source mixture and step-count strata. |

Dataset metadata is evidence, not a substitute for reviewing original source terms. The repository does not relicense third-party datasets. C's additional public data needs the same review. Raw inputs remain ignored locally. A quarantined process hashes public evaluation inputs only; no evaluation labels/examples are printed and JevBench is not scored.

## Hygiene

- Stable group IDs from normalized state ensure all questions/augmentations from one state stay together. Resolve cross-source duplicate groups before splitting.
- Split train/heldout before stochastic option augmentation. Preserve distribution labels exactly through permutation. Noul labels map back to true/false; score order is immutable.
- Index normalized 13-word-gram hashes from **all applicable evaluation sets** in an isolated pipeline; reject train candidates with any overlap. Shorter text produces no 13-gram, consistent with the brief. Log excluded train IDs/counts only, not protected evaluation text.
- `adapters.py` and `scripts/prepare_data.py` implement pinned ingest, contaminated-group removal, grouped splits, synthetic mixing, and checksum manifests. Training and serving share the option-description renderer; conversation truncation removes whole earliest turns before taking a last-turn tail. The pilot trainer requires a checksum-matched audit manifest.
- `scripts/index_evaluation.py` reads only model-input Parquet columns for DecisionBench/BBH, and public JevBench input fields in its own process. Index canonical DecisionBench `data/eval-*`, not the older 20,000-row `artifact/eval.parquet`. Include original/compact representations and nested text. Unavailable private JevBench inputs cannot be indexed; coverage is explicitly public-only.
- Report split counts per dataset, type, group and synthetic step count; verify a usable calibration sample for each type/pass rather than silently borrowing test data.

## Release caveat

An adapter's permissibility is not established solely by a base model's Apache-2.0 tag. Preserve base/model notices and training-data attribution. Quyet additionally requests retaining its NOTICE if redistributed; reference inference alone is not redistribution.
