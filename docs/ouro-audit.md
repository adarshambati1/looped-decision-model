# Ouro source audit and correctness evidence

Pinned model/source: `ByteDance/Ouro-1.4B` at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. Source URLs and SHA256s are in `results/upstream-metadata.json`. `scripts/audit_sources.py` downloads source/cards only. Real weights must stay on RunPod.

## Verified facts

- 24 shared layers, hidden size 2048, 16 MHA heads, head dimension 128; `total_ut_steps=4` and `early_exit_threshold=1.0` in config. Context config permits 65536, but we enforce the brief's **4096** training-context cap.
- `OuroModel.forward` applies `self.norm(hidden_states)` **inside** the recurrent loop, after all layers. That normalized tensor becomes the next pass's input. Do not retain unnormalized recurrence or normalize twice for the head.
- The stock model returns `(BaseModelOutputWithPast, hidden_states_list, gate_list)` internally. Its CausalLM wrapper does not expose all per-pass hidden states through the ordinary output field.
- Stock `early_exit_threshold` **does not save stack compute** in this implementation: `OuroModel` computes all configured passes before the CausalLM wrapper chooses a per-token readout. Setting 1.0 avoids intentional gate selection, but our own pass executor is needed for actual adaptive runtime.
- `total_ut_steps` is copied onto the decoder instance at construction. Updating only config after construction does not reliably change execution; our reference full-pass runner sets both.
- Attention caches use `current_ut * num_hidden_layers + layer_idx`. KV are rotated at fixed sequence positions before caching. The upstream universal cache appends on update, which is unsuitable for recurrent question-only replacement without adaptation.
- Source imports instantiate successfully under the requested `transformers==4.54.1` / Torch 2.7.1 in an isolated environment.

## Implemented correctness-first runner

`src/looped_decision/ouro.py` directly executes the pinned upstream decoder layers. Each pass has a fresh layerwise cache. State is processed once for the pass. Each question suffix attends to that pass's state-prefix KV plus its own causally masked suffix KV. The causal state cannot see any question. Prefix positions start at 0; suffix positions start at state length and **do not advance across passes**.

For B passes 5–8, the state hidden tensor and pass-four prefix cache remain unchanged; each question gets a fresh suffix cache rather than appended previous-pass question tokens. The pass-four prefix retains its autograd history during training; immutable does not mean detached.

This runner is a **correctness baseline, not the final fast serving implementation**. Ragged questions currently execute individually. CUDA-graph bucket capture, active-question compaction and batched execution are not yet implemented. Do not report this as the final speed test.

## Tokenizer findings

A–Z, a–z, 0–9 continue `Answer: (` as distinct single tokens. `+` does NOT: it merges with the preceding context. The two extra codes are therefore **`!` and `?`**, both verified prefix-stable. This is why tokenization must be audited in context instead of testing isolated letters.

State and question are tokenized separately and those exact concatenated IDs must be shared between training and serving. This deliberately guarantees a shared prefix; raw-string retokenization can change boundary tokens. Question/options are never truncated. A request uses one common state budget sized for its longest question.

## Evidence

`results/cpu-equivalence.json` was produced by `scripts/verify_ouro.py` using **tiny random weights**, not pretrained model evaluation:

- Full-pass executor matches upstream at each of four passes (observed maximum absolute error 0.0).
- Prefix-shared two-question execution matches independent concatenated execution within float32 tolerance at every pass.
- B's first four passes match A; pass-four state/cache remain immutable in passes 5–8.
- Soft-label, later-pass-weighted loss backpropagates through the head and recurrent backbone.

Still required: pretrained BF16 GPU equivalence, LoRA-only gradient checks, memory profiling, finite hidden-state monitoring, batched prefix equivalence, CUDA graph replay equivalence, and calibrated runtime stopping/compaction tests. Tiny-model evidence does not establish model accuracy, speed, or B's scientific benefit.
