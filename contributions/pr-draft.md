# Phase IV PR — pytorch/ao #729

> **Submitted 2026-06-17 as [pytorch/ao#4507](https://github.com/pytorch/ao/pull/4507)** (ready for review).
> This file is the source for the PR body below.

- **PR:** <https://github.com/pytorch/ao/pull/4507>
- **Base:** `pytorch/ao:main`
- **Head:** `mykolas-perevicius:fix-issue-729-moe-quant-example`
- **Closes:** #729
- **Commits (post-rebase):** [`cb65991c3`](https://github.com/pytorch/ao/pull/4507/commits/cb65991c3) (example) · [`ba6c62b09`](https://github.com/pytorch/ao/pull/4507/commits/ba6c62b09) (test)

---

## Title

```
Add weight-only quantization MoE example
```

## Body

### Summary

Adds a runnable example demonstrating weight-only quantization of a Mixture-of-Experts (MoE)
model via `quantize_()`, addressing #729. The issue notes that `quantize_()` should already
support MoE — this PR is the missing showcase plus a short docs note, with **no core changes**.

### What it does

`examples/quantize_moe.py` builds a small token-choice top-2 MoE block (a softmax router plus
`nn.Linear` experts) and quantizes **only the expert weights** via
`quantize_(model, Int8WeightOnlyConfig(), filter_fn=is_expert_linear)`. The router is left in
high precision on purpose, so token-to-expert routing is identical to the baseline (quantizing
the router would change routing decisions, not just numerics).

The script is self-verifying: it prints before/after weight types and serialized sizes, runs a
forward pass, reports SQNR vs the fp32 baseline, and asserts (experts quantized, router not,
≥1.5x smaller, SQNR > 25 dB), exiting non-zero on any failure.

- `--dtype int8|int4` and `--device` flags; int4 is gated behind a hardware/dependency warning.
- The docstring points users with real fused-3D-expert checkpoints
  (e.g. `meta-llama/Llama-4-Scout-17B-16E-Instruct`) at the `FqnToConfig` + `PerRow(1)` pattern,
  mirroring `examples/quantize_llama_4.py`.

### Test plan

**Unit test** — `test/quantization/test_quant_api.py`: adds a module-level `ToyMoEModel`
(mirroring `ToyLinearModel`) and `TestQuantFlow.test_int8_weight_only_moe_experts_only`, which
asserts every expert weight becomes `Int8Tensor`, the router weight stays unquantized, and SQNR
vs float32 exceeds 25 dB. Modeled on the file's existing `compute_error` / isinstance conventions.

**CPU (macOS, torch 2.12):** example runs twice identically — experts → `Int8Tensor`, router
`float32`, serialized 8.40 → 2.15 MB (3.90x), SQNR 45.1 dB, all asserts pass. `pytest … -k moe`
→ 1 passed.

**CUDA (RTX 3090, SM 8.6, torch 2.12.1+cu130, CUDA 13.0):**

- `python examples/quantize_moe.py --device cuda` (int8) → matches the CPU run exactly (3.90x,
  SQNR 45.1 dB, all asserts pass) — the `--device cuda` path is exercised end-to-end.
- `pytest test/quantization/test_quant_api.py -k moe` → 1 passed.
- Full `test/quantization/test_quant_api.py` → **18 passed, 28 skipped, 0 failed**. Skips are all
  hardware-gated (`Need SM 8.9+` / `Checkpoints are produced in SM90+` — float8 paths on this
  SM 8.6 card) or pre-existing unconditional skips; none are introduced by this change.

CI lints `examples/` but does not execute them (only `tutorials/` run via `run_tutorials.yml`),
so the example is self-asserting and the unit test provides the regression guard.

### Open questions for maintainers

1. **Example location** — top-level `examples/quantize_moe.py` (matching `examples/quantize_llama_4.py`,
   added in #3408) vs `examples/inference/` (hinted by the `literalinclude` in `quant_api.py`)?
2. **Keep or drop the unit test?** #3408 was single-file. Since CI never runs examples, I added a
   test so MoE support is regression-checked — happy to drop it if you prefer the single-file shape.
3. **`examples/README.md` entry** — should I add one? #3408 didn't.
4. **Supported public path for int4?** The int4 default path requires `mslk >= 1.0.0`, but the
   `mslk` package on public PyPI is a `0.0.0` placeholder (894-byte empty wheel), and
   `torchao/utils.py:1226` gates real availability on `is_fbcode()`. On a fresh RTX 3090 with the
   public wheel, `--dtype int4` raises `ImportError: Requires mslk >= 1.0.0` (same on CPU and
   CUDA). What's the supported public way to exercise int4 weight-only — build FBGEMM-GenAI, a
   different config, or is it fbcode-only for now? (The example gates int4 behind a flag + warning,
   so the int8 showcase is unaffected.)

### Notes

- Minor, separate from this PR: `torchao/quantization/quant_api.py:1499` has an invalid escape
  `'\.'` in a docstring that triggers `SyntaxWarning: invalid escape sequence` on import under
  Python ≥ 3.12. Happy to send a one-line `r"""` micro-fix as its own PR.
