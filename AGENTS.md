# AGENTS.md

Repo-specific guidance for agents working in OpenMythos. Python + PyTorch research library implementing a Recurrent-Depth Transformer. Package dir is `open_mythos` (underscore); distribution name is `open-mythos`.

## Commands

Build system is Poetry (`poetry-core` backend, see `pyproject.toml`). No Makefile, no CI, no pre-commit hooks.

- Install (editable, dev): `pip install -e .` or `poetry install`
- Lint: `ruff check .`
- Format: `black .` (verify with `black --check .`). Line length 88, target py310.
- Test (all): `pytest`
- Test (one case): `pytest tests/test_main.py::TestRMSNorm::test_output_shape`
- Smoke run: `python example.py`
- Train 3B on FineWeb-Edu (single GPU): `python training/3b_fine_web_edu.py`
- Train multi-GPU: `torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py`

`requirements.txt` at root is a loose, hand-maintained pin list. The authoritative deps live in `pyproject.toml` (notably `torch==2.11.0` exact).

## Two parallel model implementations — don't edit the wrong file

- `open_mythos/main.py` — the canonical `OpenMythos` RDT (Prelude → looped Recurrent Block → Coda). All public exports in `__init__.py` come from here. This is what users import.
- `open_mythos/moda.py` — a **separate, self-contained** Mixture-of-Depths-Attention + DeepSeek-MoE model with its own `MoDAConfig` and `MoDAModel`. It is **not** exported by `__init__.py` and is not wired into `OpenMythos`. Treat as an experimental parallel module; do not conflate the two configs.

## Architectural invariants to preserve when editing `main.py`

- **LTI injection stability is by construction**: `LTIInjection.get_A()` returns a diagonal `A` with all entries strictly in `(0, 1)` (computed in log space), so the spectral radius `ρ(A) < 1` always holds regardless of gradients. Verify via `torch.linalg.eigvals(A).abs().max()`. Do not replace with an unconstrained parameterization.
- **`n_loops` may exceed `cfg.max_loop_iters`** at inference (depth extrapolation). `LoRAAdapter` clamps the loop index to its trained range (reusing the last learned per-loop scale rather than indexing out of range); preserve this clamp.
- RoPE is applied to Q and K **before** they enter the KV cache, so cached values are never re-rotated. MLA caches the compressed KV latent (`kv_lora_rank`), not full K/V.
- `flash-attn` is optional and **only affects `GQAttention`** (not `MLAttention`). It falls back silently to manual SDPA when absent — guard any change with the `_HAS_FLASH_ATTN` flag. Install via the `[flash]` extra: `pip install open-mythos[flash]` (requires CUDA + build tools).

## Tests and benchmarks — different run profiles

- `tests/test_main.py` — pure CPU unit tests using tiny `MythosConfig` overrides (dim=64, etc.). Fast, no network. This is the default suite to run after edits.
- `tests/test_tokenizer.py` — **requires network/HF Hub access**: instantiates `MythosTokenizer`, which downloads `openai/gpt-oss-20b`. Will fail/hang offline.
- `tests/bench_vs_transformer.py` and `tests/small_benchmark.py` are **CLI benchmark scripts, not pytest tests**. Run directly, e.g. `python tests/small_benchmark.py --device cuda --steps 5000`. They default to `cuda` if available else `cpu`.
- `tests/test_rope_debug.py` exists alongside the suite.
- No `conftest.py`, no pytest markers, no skip decorators — there is no GPU/slow filtering; run selectively by path/node instead.

## Training script gotchas

`training/3b_fine_web_edu.py` (FSDP + AdamW) has its own deps in `training/requirements.txt` that **differ from the main requirements**: it needs `datasets>=3.6.0` (newer than the library's `>=2.18`) and `loguru`. Install training deps separately before running. The dataset is streamed (`HuggingFaceFW/fineweb-edu`, `sample-10BT` default) and shards by `(rank, worker_id)`; streaming is not seekable so a resumed run replays its shard from the start.

## Known doc/code mismatches (verify, don't trust)

- README's "Model Variants" example calls `mythos_7b()` — **no such variant exists**. Actual presets in `open_mythos/variants.py`: `mythos_1b`, `mythos_3b`, `mythos_10b`, `mythos_50b`, `mythos_100b`, `mythos_500b`, `mythos_1t`. All use `attn_type="mla"` regardless of the README's GQA framing.
- `example.py` computes the "spectral radius" as `A.max().item()` — **incorrect** (it just reads a diagonal entry). Use `torch.linalg.eigvals(A).abs().max().item()` as shown in the README.
- `open_mythos/__init__.py` lists `load_tokenizer` and `get_vocab_size` in `__all__` but never imports or defines them. `from open_mythos import load_tokenizer` will raise `ImportError`. Do not rely on those names.
