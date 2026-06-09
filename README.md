# Contribution 1: MoE example — weight-only quantization showcase (pytorch/ao #729)

**Contribution Number:** 1  
**Student:** Mykolas Perevicius — GitHub [@mykolas-perevicius](https://github.com/mykolas-perevicius) — AI301 Section 1A  
**Issue:** https://github.com/pytorch/ao/issues/729  
**Fork:** https://github.com/mykolas-perevicius/ao  
**Status:** Phase I Complete

---

## Why I Chose This Issue

I chose pytorch/ao #729 because it's in my strongest language (Python) on a marquee project — PyTorch's architecture-optimization library — and it's scoped as a good first issue: add a runnable example showing weight-only quantization applied to a Mixture-of-Experts (MoE) model. The maintainer believes `quantize_()` already supports this, so the work is a clear showcase plus a short doc note rather than new core logic — a low-risk first contribution where I can learn torchao's quantization API and the project's example/docs conventions. My learning goal is to understand torchao's `quantize_()` workflow well enough to demonstrate it end-to-end, and "fixed" means a self-contained, runnable example (and a docs reference) that quantizes an MoE model.

### Problem summary
torchao supports weight-only quantization, but there is no runnable example demonstrating it on a Mixture-of-Experts model, so users have no reference for the workflow even though the API reportedly already covers it. This matters because MoE models are increasingly common and quantization is a core reason people reach for torchao; a missing example is a real onboarding gap. I chose it because it's bounded, Python-facing, on an active and AI-friendly project, and demonstrating an existing API is a high-confidence path to a merged first PR.

**Skill match:** Python; familiarity with PyTorch `nn.Module` models and the quantize-a-model mental model; comfortable reading library source to find the public API surface.

**Learning goal:** Learn torchao's `quantize_()` config-object workflow (`Int8WeightOnlyConfig` / `Int4WeightOnlyConfig`) end-to-end, and the project's conventions for contributing a runnable example.

**My understanding of the problem:** The issue author (a torchao maintainer) says weight-only quant should already work on an MoE model via `quantize_()`; the deliverable is a clear, runnable *example* that proves it, not a core code change.

**Investigative depth (targets the +3 bonus):**
- Likely files / modules involved: new example at `examples/quantize_moe.py` (mirrors `examples/quantize_llama_4.py`); core API `quantize_` at `torchao/quantization/quant_api.py:275`; weight-only configs `Int8WeightOnlyConfig` (`quant_api.py:727`) and `Int4WeightOnlyConfig` (`quant_api.py:527`).
- Maintainer comments / related issues: issue body statement by **@msaroufim** that `quantize_()` should already support MoE — https://github.com/pytorch/ao/issues/729 ; context comment by @felipemello1 — https://github.com/pytorch/ao/issues/729#issuecomment-2305152753 ; cc'd reviewers @jcaip and @cpuhrsch.
- Acceptance criteria — what "done" means: a self-contained example that builds/loads a small MoE model, applies weight-only quant via `quantize_()`, runs a forward pass, and shows the effect (dtype/size change); runs on CPU with `Int8WeightOnlyConfig`; follows example conventions + BSD-3 header; passes lint/CI.

---

## Understanding the Issue

### Problem Description
No example/tutorial demonstrates weight-only quantization on an MoE model in torchao.

### Expected Behavior
An example (script and/or doc entry) that builds or loads an MoE model, applies weight-only quantization via `quantize_()`, and demonstrates it working end-to-end.

### Current Behavior
No such example exists; the capability is undocumented by example.

### Affected Components
A new example under torchao's examples area — no core API change expected. Specifics confirmed during baseline (2026-06-08):

- **New example file (primary candidate):** `examples/quantize_moe.py` — a new single-file script mirroring the existing `examples/quantize_llama_4.py` convention (BSD-3 Meta license header + module docstring), listed in `examples/README.md`.
  - *Alternative to confirm with maintainer:* `examples/inference/quantize_moe.py`. The top-level `examples/inference/` directory does **not** currently exist, but `torchao/quantization/quant_api.py:542` contains a docstring `literalinclude` pointing at `examples/inference/int4_weight_only.py`, hinting the project may prefer an `examples/inference/` home for weight-only examples.
- **Core API involved (no change expected):**
  - `quantize_` — defined at `torchao/quantization/quant_api.py:275`, exported from `torchao/quantization/__init__.py` → `from torchao.quantization import quantize_`.
  - Weight-only configs (modern `AOBaseConfig` API): `Int8WeightOnlyConfig` (`quant_api.py:727`, **CPU-runnable**) and `Int4WeightOnlyConfig` (`quant_api.py:527`, uses the **int4 tinygemm kernel — GPU-preferred**). Legacy callables `int4_weight_only` / `int8_weight_only` also exist.
- **Confirmed gap:** the `torchao/prototype/moe_training/` tree is low-precision MoE *training* (a different feature), and `examples/quantize_llama_4.py` quantizes Llama-4 routed experts to **float8 w8a8 dynamic** (activation + weight), not weight-only — so a weight-only-on-MoE showcase genuinely does not exist.

---

## Reproduction Process

> _This issue is a feature/example request, not a bug — there is no failing behavior to reproduce. In place of a bug repro, the baseline below (and the per-issue log) record the investigation that confirms the gap and pins the API/example locations. Live reproduction/output capture comes in Phase II._

### Environment Setup
> _Phase II — coming soon._ Fork cloned to `~/Developer/oss/forks/ao`, `upstream` → `pytorch/ao`, branch `feat/moe-quant-example`. Dev setup will follow `CONTRIBUTING.md` (fork + branch from `main`) and `dev-requirements.txt`.

### Steps to Reproduce
> _Phase II — coming soon (will run the example end-to-end and capture output)._

### Reproduction Evidence
- **Commit showing reproduction:** _Phase II._
- **Screenshots / logs:** _Phase II._
- **My findings (baseline):** Confirmed no existing weight-only MoE example; located `quantize_` and the weight-only config classes; identified `examples/quantize_moe.py` as the convention-matching home. See per-issue log `contributions/pytorch-ao-729.md`.

---

## Solution Approach

### Analysis
> _Phase II — coming soon._ (Feature/example: no root cause; the "analysis" is confirming `quantize_()` + a weight-only config applied to MoE expert `nn.Linear` layers produces the expected dtype/size change and a working forward pass.)

### Proposed Solution
Add a runnable example that applies weight-only quantization to a small MoE model via `quantize_()`, with a brief doc note; no new core functionality.

**Acceptance Criteria (confirm with maintainer / adjust per baseline):**
- New example builds or loads a small MoE model and applies weight-only quantization via `quantize_()`.
- Runs end-to-end and shows the effect (e.g., dtype/size change and a successful forward pass).
- Includes run instructions and a hardware note (CPU-runnable with `Int8WeightOnlyConfig`; int4/tinygemm path prefers a CUDA GPU).
- Follows torchao example conventions (location, style, BSD-3 license header) and passes lint/CI.
- Referenced from docs/`examples/README.md` if the project indexes examples there.

**Open questions for the maintainer (already asked in my claim comment):** preferred MoE model (toy vs real); example location (`examples/` vs `examples/inference/`); CPU vs GPU.

### Implementation Plan (UMPIRE)
> _Phase II — coming soon._
- **Understand:** Demonstrate weight-only quant on an MoE model via `quantize_()`.
- **Match:** `examples/quantize_llama_4.py` (script structure/header); `Int8WeightOnlyConfig` / `Int4WeightOnlyConfig` usage.
- **Plan:** _Phase II._
- **Implement:** _Phase II (branch `feat/moe-quant-example`)._
- **Review:** _Phase II._
- **Evaluate:** _Phase II._

---

## Testing Strategy

> _Phase III — coming soon._

### Unit Tests
- [ ] _Phase III._

### Integration Tests
- [ ] _Phase III._

### Manual Testing
> _Phase III — run the example on CPU (int8 weight-only) and capture output._

---

## Implementation Notes

> _Phase III — coming soon._

### Code Changes
- **Files modified:** _Phase III._
- **Key commits:** _Phase III._
- **Approach decisions:** _Phase III._

---

## Pull Request

> _Phase IV — coming soon._

**PR Link:** _Phase IV._

**PR Description:** _Phase IV._

**Maintainer Feedback Log:**
- _Phase IV._

**Status:** Not yet opened.

---

## Learnings & Reflections

> _Phase IV — coming soon._

### Technical Skills Gained
_Phase IV._

### Challenges Overcome
_Phase IV._

### What I'd Do Differently Next Time
_Phase IV._

---

## Resources Used

- Issue: https://github.com/pytorch/ao/issues/729
- Maintainer statement (`quantize_` already supports MoE) — issue body by @msaroufim (torchao maintainer): https://github.com/pytorch/ao/issues/729
- Context comment (@felipemello1, jamba usage): https://github.com/pytorch/ao/issues/729#issuecomment-2305152753
- torchao `quantize_` API: `torchao/quantization/quant_api.py:275` (exported via `torchao.quantization`)
- Weight-only configs: `Int8WeightOnlyConfig` (`quant_api.py:727`), `Int4WeightOnlyConfig` (`quant_api.py:527`)
- Example convention reference: `examples/quantize_llama_4.py`; `examples/README.md`
- torchao dev/setup: `CONTRIBUTING.md` (fork + branch from `main`), `dev-requirements.txt`, `pyproject.toml`
