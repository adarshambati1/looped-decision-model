# Verified wire contract

Inspected source: [stiermid/laya-serve at b0bcbd53a8531d8c5688e3c9199f914e40cb96b4](https://github.com/stiermid/laya-serve/tree/b0bcbd53a8531d8c5688e3c9199f914e40cb96b4), `src/laya_serve/schemas.py`, `compat.py`, and `tests/test_api.py`. MIT upstream implementation; our small schema implementation is independent. Also located `mohit67890/imajev` at `318e8dad55c12a1c6faddbec5f30489d2ee529ff` (not used as the primary contract).

## Important corrections to the illustrative brief

- `score.probabilities` is an **object keyed by string indices**, not an array: `{"0":0.2,"1":0.8}`.
- Score answers require **both `legend` and `confidence`**. Legend maps string indices to original criteria (which may be structured JSON).
- `usage` is `{input_tokens, output_tokens}`, **not `{passes, ms}`**. Keep measured pass counts and timings in benchmark records or optional response headers, not invented wire fields. A non-generative head emits zero text tokens; report output_tokens=0 rather than inventing usage.
- Noul answers contain only `type` and `noul`. Noul requests may include optional true/false criteria descriptions.
- Instructions/state/criteria can include structured JSON objects or lists, not only strings. Preserve order and serialize deterministically for inference.
- The upstream API permits 1–255 choice options and 2–10 score levels. This project intentionally limits choice to **2–64** and requests to **1–64 questions**, as requested. Score retains upstream 2–10 levels. This is a supported subset, not unrestricted compatibility.

## Decision semantics

Choice output uses original option keys and all calibrated probabilities. Confidence is `(k*pmax-1)/(k-1)` clamped to [0,1]. Score is the probability-weighted zero-based level index; confidence uses the same normalization over level probabilities. Noul is P(true) in canonical false/true order, irrespective of training permutations.

The halting threshold compares **pmax**, not the normalized confidence field. The serving implementation must preserve this distinction.

## Serving state

A backend interface is under construction. Until a real calibrated checkpoint and tested execution backend exist, `/v1/systemone` must return unavailable—not fake decisions. Contract unit tests may use explicit injected test backends. Do not report such tests as model quality or latency.
