# Environment isolation

- `.venv`: API, NumPy calibration, data hygiene, unit tests; no model weights.
- `.venv-train`: `uv venv .venv-train --python 3.11`; install `train.txt` and this repository using `uv pip install --python .venv-train/bin/python -r environments/train.txt -e '.[dev]'`. Keep transformers at 4.54.1. Tiny random-config equivalence tests may run on laptop CPU; real weights/training run on RunPod.
- `.venv-decisionbench`: install ONLY the verified official harness's locked requirements at its recorded revision, plus a thin HTTP adapter. Audit pending.
- `.venv-jevbench`: final-stage only, official pinned harness; never use for A/B selection.
- `.venv-vllm-h2o`: pinned author-recommended vLLM + H2O shim; do not import our training environment here. Version still under audit.
- `.venv-quyet`: author-recommended reference serving stack; do not assume stock vLLM. Version still under audit.

No placeholder requirements are represented as a verified reference deployment. Record `pip freeze`, image digest, GPU SKU, CUDA/driver versions, model revision, and command for each measured run.
