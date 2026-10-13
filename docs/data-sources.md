# Data provenance and license audit (initial)

Exact metadata/card revisions and hashes: `results/upstream-metadata.json`.

| Dataset | Intended mapping | Metadata license | Status |
|---|---|---|---|
| LocalLLaMA/typed-decisions | config `all`, train; expand questions sharing one state/group; keep distributions | Apache-2.0 | Card confirms four workflows, 300 train cases each, five decisions per case. Do not combine `all` with individual configs (duplicates). Row adapter still pending. |
| allenai/ai2_arc | ARC-Easy + ARC-Challenge train, choice | CC-BY-SA-4.0 | Attribution/share-alike implications must be reviewed before dataset redistribution; adapter pending. |
| tau/commonsense_qa | train, choice | MIT | Adapter pending; retain attribution. |
| google/boolq | train passage as state, question as noul | CC-BY-SA-3.0 | Attribution/share-alike review; adapter pending. |
| SetFit/sst5 | train, score levels very negative through very positive | **Not declared in fetched HF card metadata** | **License unresolved. Do not silently assume permission or silently omit it from a claimed full training run.** Check original SST distribution/license. |
| Generated multihop/balance | 10% of training examples for A/B; larger for C | Original generated code/data | Deterministic generators and correctness tests implemented. Record source mixture and step-count strata. |

Dataset metadata is evidence, not a substitute for reviewing original source terms. The repository does not relicense third-party datasets. C's additional public data needs the same review. Only dataset cards have been downloaded in the initial audit; no benchmark rows are in the repo.

## Hygiene

- Stable group IDs from normalized state ensure all questions/augmentations from one state stay together. Resolve cross-source duplicate groups before splitting.
- Split train/heldout before stochastic option augmentation. Preserve distribution labels exactly through permutation. Noul labels map back to true/false; score order is immutable.
- Index normalized 13-word-gram hashes from **all applicable evaluation sets** in an isolated pipeline; reject train candidates with any overlap. Shorter text produces no 13-gram, consistent with the brief. Log excluded train IDs/counts only, not protected evaluation text.
- The local `data.py` module implements primitives, not a completed data ingest. Production adapters must preserve option descriptions, state type/conversation semantics, and source licenses. The pilot trainer requires a checksum-matched audit manifest and rejects incomplete audit gates.
- Report split counts per dataset, type, group and synthetic step count; verify a usable calibration sample for each type/pass rather than silently borrowing test data.

## Release caveat

An adapter's permissibility is not established solely by a base model's Apache-2.0 tag. Preserve base/model notices and training-data attribution. Quyet additionally requests retaining its NOTICE if redistributed; reference inference alone is not redistribution.
