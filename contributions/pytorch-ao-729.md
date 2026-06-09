# Contribution 1: MoE example — weight-only quantization showcase (pytorch/ao #729)

**Contribution Number:** 1  
**Student:** Mykolas Perevicius — GitHub [@mykolas-perevicius](https://github.com/mykolas-perevicius) — AI301 Section 1A  
**Issue:** https://github.com/pytorch/ao/issues/729  
**Fork:** https://github.com/mykolas-perevicius/ao  
**Branch:** `feat/moe-quant-example`  
**Status:** Phase I Complete

---

## Why I Chose This Issue

I chose pytorch/ao #729 because it's in my strongest language (Python) on a marquee project — PyTorch's architecture-optimization library — and it's scoped as a good first issue: add a runnable example showing weight-only quantization applied to a Mixture-of-Experts (MoE) model. The maintainer believes `quantize_()` already supports this, so the work is a clear showcase plus a short doc note rather than new core logic — a low-risk first contribution where I can learn torchao's quantization API and the project's example/docs conventions. My learning goal is to understand torchao's `quantize_()` workflow well enough to demonstrate it end-to-end, and "fixed" means a self-contained, runnable example (and a docs reference) that quantizes an MoE model.

### Problem summary
torchao supports weight-only quantization, but there is no runnable example demonstrating it on a Mixture-of-Experts model, so users have no reference for the workflow even though the API reportedly already covers it. This matters because MoE models are increasingly common and quantization is a core reason people reach for torchao; a missing example is a real onboarding gap. I chose it because it's bounded, Python-facing, on an active and AI-friendly project, and demonstrating an existing API is a high-confidence path to a merged first PR.

---

## Understanding the Issue

**Problem Description:** No example/tutorial demonstrates weight-only quantization on an MoE model in torchao.

**Expected Behavior:** An example (script and/or doc entry) that builds or loads an MoE model, applies weight-only quantization via `quantize_()`, and demonstrates it working end-to-end.

**Current Behavior:** No such example exists; the capability is undocumented by example.

**Affected Components:**
- **New example file (primary candidate):** `examples/quantize_moe.py` — new single-file script mirroring `examples/quantize_llama_4.py` (BSD-3 Meta license header + module docstring), listed in `examples/README.md`.
  - *Alternative to confirm:* `examples/inference/quantize_moe.py` — `examples/inference/` does not currently exist, but `torchao/quantization/quant_api.py:542` references `examples/inference/int4_weight_only.py` via a docstring `literalinclude`.
- **Core API (no change expected):** `quantize_` at `torchao/quantization/quant_api.py:275` (exported via `torchao.quantization`); weight-only configs `Int8WeightOnlyConfig` (`quant_api.py:727`, CPU-runnable), `Int4WeightOnlyConfig` (`quant_api.py:527`, int4 tinygemm — GPU-preferred).
- **Confirmed gap:** `torchao/prototype/moe_training/` is MoE *training* (different); `examples/quantize_llama_4.py` uses float8 w8a8 dynamic (activation+weight), not weight-only.

---

## Solution Approach

**Proposed Solution:** Add a runnable example that applies weight-only quantization to a small MoE model via `quantize_()`, with a brief doc note; no new core functionality.

**Acceptance Criteria (confirm with maintainer / adjust per baseline):**
- New example builds or loads a small MoE model and applies weight-only quantization via `quantize_()`.
- Runs end-to-end and shows the effect (e.g., dtype/size change and a successful forward pass).
- Includes run instructions and a hardware note (CPU-runnable with `Int8WeightOnlyConfig`; int4/tinygemm prefers a CUDA GPU).
- Follows torchao example conventions (location, style, BSD-3 license header) and passes lint/CI.
- Referenced from docs / `examples/README.md` if the project indexes examples there.

**Open questions for the maintainer (already asked in my claim comment):** preferred MoE model (toy vs real); example location (`examples/` vs `examples/inference/`); CPU vs GPU.

---

## Resources Used

- Issue: https://github.com/pytorch/ao/issues/729
- Maintainer statement (`quantize_` already supports MoE) — issue body by @msaroufim (torchao maintainer): https://github.com/pytorch/ao/issues/729
- Context comment (@felipemello1, jamba usage): https://github.com/pytorch/ao/issues/729#issuecomment-2305152753
- torchao `quantize_` API: `torchao/quantization/quant_api.py:275` (exported via `torchao.quantization`)
- Weight-only configs: `Int8WeightOnlyConfig` (`quant_api.py:727`), `Int4WeightOnlyConfig` (`quant_api.py:527`)
- Example convention reference: `examples/quantize_llama_4.py`; `examples/README.md`
- torchao dev/setup: `CONTRIBUTING.md` (fork + branch from `main`), `dev-requirements.txt`, `pyproject.toml`

---

## Baseline / Investigation Log

### 2026-06-08 — Phase I baseline (no bug repro; this is a feature/example)

**Environment**
- macOS (Darwin 25.5), git 2.54.0, gh 2.93.0.
- Fork: https://github.com/mykolas-perevicius/ao (isFork:true, parent pytorch/ao).
- Local clone: `~/Developer/oss/forks/ao`; remotes `origin` → my fork, `upstream` → `https://github.com/pytorch/ao.git`; `git fetch upstream` done; branch `feat/moe-quant-example` created.

**Re-verify before work (issue still free)**
- `gh issue view 729 -R pytorch/ao` → state OPEN, assignees 0, labels `good first issue`.
- Comments: 2 — @felipemello1 (2024 context) + my own claim (@mykolas-perevicius). No competing claim.
- `gh pr list -R pytorch/ao --state open --search "729"` → returns PR #3416, but the GraphQL timeline cross-ref/connected check returns `[]` → #3416 is a false-positive string match, NOT linked to #729.

**Commands run (baseline investigation, in `~/Developer/oss/forks/ao`)**
```
ls -1                                              # repo has examples/ and tutorials/ dirs
grep -ril -E "mixture.of.experts|\bMoE\b" --include=*.py --include=*.md .
find . -iname '*moe*'                              # only moe_training/ (training, not inference quant)
find examples -maxdepth 2 -type f                  # examples/quantize_llama_4.py, examples/README.md, sam2_*
ls examples/inference/                             # -> does NOT exist
grep -rl "quantize_" examples tutorials            # examples/quantize_llama_4.py uses quantize_ (float8 w8a8)
grep -rn "^def quantize_" torchao/quantization/quant_api.py        # :275
grep -rn "quantize_" torchao/quantization/__init__.py             # exported (:40,:74)
grep -rn -E "class (Int4WeightOnlyConfig|Int8WeightOnlyConfig)" torchao/quantization/quant_api.py  # :527 / :727
sed -n '1,30p' examples/quantize_llama_4.py        # BSD-3 header + module docstring convention
grep -n -E "cuda|device|tinygemm" torchao/quantization/quant_api.py  # int4 tinygemm GPU path; device='cuda' speeds quant
```

**Findings**
1. **No existing weight-only MoE inference example.** `torchao/prototype/moe_training/` is low-precision MoE *training* (distinct feature). `examples/quantize_llama_4.py` quantizes Llama-4 routed experts but to **float8 w8a8 dynamic** (activation + weight), not weight-only. → the weight-only-on-MoE showcase gap is real.
2. **API located.** `quantize_` at `torchao/quantization/quant_api.py:275`, exported from `torchao/quantization/__init__.py` (`from torchao.quantization import quantize_`). Weight-only configs: `Int8WeightOnlyConfig` (`quant_api.py:727`), `Int4WeightOnlyConfig` (`quant_api.py:527`). Legacy `int4_weight_only` / `int8_weight_only` callables also present.
3. **Candidate example location.** Primary: `examples/quantize_moe.py` (top-level single-file convention, mirroring `quantize_llama_4.py`; add to `examples/README.md`). Alternative: `examples/inference/quantize_moe.py` — that dir doesn't exist yet, but `quant_api.py:542` references `examples/inference/int4_weight_only.py` in a docstring `literalinclude`, so confirm the preferred home with the maintainer.
4. **Maintainer statement permalink.** The "`quantize_()` should already support MoE" statement is in the **issue body**, authored by **@msaroufim** (torchao maintainer): https://github.com/pytorch/ao/issues/729 . Reviewers cc'd in body: @jcaip, @cpuhrsch.
5. **GPU need.** `Int8WeightOnlyConfig` weight-only quant runs end-to-end on **CPU** (good for a portable example / CI). `Int4WeightOnlyConfig` uses the **int4 tinygemm kernel** and prefers a **CUDA GPU** (`quant_api.py:288` notes `device="cuda"` speeds quantization; `:298–302` note the int4 tinygemm optimized path + `torch.compile`).

**Candidate file locations (to create in Phase II)**
- `examples/quantize_moe.py` (primary) — new runnable script.
- `examples/README.md` — add a one-line entry pointing to it.
- (alt) `examples/inference/quantize_moe.py` — pending maintainer preference.

**STOP point:** Per workflow, stopping after baseline + README enrichment are committed. No example code written, no PR opened.
