# Contribution 1: MoE example — weight-only quantization showcase (pytorch/ao #729)

**Contribution Number:** 1  
**Student:** Mykolas Perevicius — GitHub [@mykolas-perevicius](https://github.com/mykolas-perevicius) — AI301 Section 1A  
**Issue:** <https://github.com/pytorch/ao/issues/729>  
**Fork:** <https://github.com/mykolas-perevicius/ao>  
**Status:** Phase II Complete

> **Working log:** the first-look investigation, dated baseline evidence, and command transcripts live in [`contributions/pytorch-ao-729.md`](./contributions/pytorch-ao-729.md). This README keeps the report essentials.

---

## Why I Chose This Issue

I chose pytorch/ao #729 because it's in my strongest language (Python) on a marquee project — PyTorch's architecture-optimization library — and it's scoped as a good first issue: add a runnable example showing weight-only quantization applied to a Mixture-of-Experts (MoE) model. The maintainer believes `quantize_()` already supports this, so the work is a clear showcase plus a short doc note rather than new core logic — a low-risk first contribution where I can learn torchao's quantization API and the project's example/docs conventions. My learning goal is to understand torchao's `quantize_()` config-object workflow (`Int8WeightOnlyConfig` / `Int4WeightOnlyConfig`) well enough to demonstrate it end-to-end, and "fixed" means a self-contained, runnable example (and a docs reference) that quantizes an MoE model.

This matters because MoE models are increasingly common and quantization is a core reason people reach for torchao, so a missing example is a real onboarding gap. The issue is bounded, Python-facing, on an active and AI-friendly project, and demonstrating an existing API is a high-confidence path to a merged first PR.

**Skill match:** Python; familiarity with PyTorch `nn.Module` models and the quantize-a-model mental model; comfortable reading library source to find the public API surface.

### Investigative depth (targets the +3 bonus)

Kept explicit here for grading; full file/line detail, command transcripts, and dated evidence live in the [working log](./contributions/pytorch-ao-729.md).

- **Likely files / modules involved:** new example at `examples/quantize_moe.py` (mirroring `examples/quantize_llama_4.py`); core API `quantize_` plus weight-only configs `Int8WeightOnlyConfig` / `Int4WeightOnlyConfig` in `torchao/quantization/quant_api.py` — exact line numbers in the working log's key-locations map.
- **Maintainer comments / related issues:** issue-body statement by **@msaroufim** (torchao maintainer) that `quantize_()` should already support MoE; context comment by @felipemello1 (<https://github.com/pytorch/ao/issues/729#issuecomment-2305152753>); reviewers cc'd: @jcaip, @cpuhrsch.
- **Acceptance criteria — what "done" means:** a self-contained, runnable example that builds/loads a small MoE model, applies weight-only quant via `quantize_()`, runs a forward pass, and shows the effect (dtype/size change); CPU-runnable with `Int8WeightOnlyConfig`; follows example conventions + BSD-3 header; passes lint/CI. Full checklist under [Solution Approach](#proposed-solution).

---

## Understanding the Issue

### Problem Description

torchao supports weight-only quantization, but no example or tutorial demonstrates it on a Mixture-of-Experts model — users have no reference for the workflow even though the API reportedly already covers it.

### Expected Behavior

An example (script and/or doc entry) that builds or loads an MoE model, applies weight-only quantization via `quantize_()`, and demonstrates it working end-to-end.

### Current Behavior

No such example exists; the capability is undocumented by example. Baseline (2026-06-08) confirmed the gap is real: `torchao/prototype/moe_training/` is MoE *training* (a different feature), and `examples/quantize_llama_4.py` quantizes routed experts to float8 w8a8 dynamic (activation + weight), not weight-only.

### Affected Components

- **New example** under `examples/` — primary candidate `examples/quantize_moe.py`, mirroring `examples/quantize_llama_4.py` conventions (BSD-3 license header + module docstring), plus a one-line entry in `examples/README.md`.
- **Existing core API, no change expected:** `quantize_` and the weight-only configs `Int8WeightOnlyConfig` (CPU-runnable) / `Int4WeightOnlyConfig` (int4 tinygemm, GPU-preferred) in `torchao/quantization/quant_api.py`.

File/line-level detail and the open `examples/` vs `examples/inference/` location question: see the [working log](./contributions/pytorch-ao-729.md).

---

## Reproduction Process

> This issue is a feature/example request, not a bug — so "reproduction" here means two things, both completed in Phase II: (a) confirming the gap is real (no weight-only MoE example exists anywhere in the repo), and (b) verifying the maintainer's claim that `quantize_()` already supports MoE — the claim the planned example depends on. Raw transcripts and command output live in the [working log](./contributions/pytorch-ao-729.md).

### Environment Setup

**Setup approach:** README/`CONTRIBUTING.md` instructions (no dev container). Fork cloned to `~/Developer/oss/forks/ao`; remotes `origin` → my fork, `upstream` → `pytorch/ao`; working branch [`fix-issue-729-moe-quant-example`](https://github.com/mykolas-perevicius/ao/tree/fix-issue-729-moe-quant-example). venv lives outside the repo (keeps the fork clean), then `pip install torch numpy` (torch 2.12.0, CPU/MPS) and torchao's documented CPU-only dev install: `USE_CPP=0 pip install -e . --no-build-isolation` → `torchao 0.18.0+git5165bfb03` editable.

Challenges encountered and how they were resolved:

- **Python 3.14 wheel risk.** Only Python 3.14.4 (Homebrew) is installed locally, and projects often lag new CPython releases. Checked `pip index versions torch` *before* installing anything — torch ≥ 2.9 ships cp314 wheels, so no pyenv detour was needed.
- **C++/CUDA extensions.** torchao's default build compiles C++ extensions; `USE_CPP=0` (per the project's dev docs) skips them, and the editable install then builds in seconds on a laptop.
- **Harmless-but-noisy import warnings on Python 3.14.** `torchao/quantization/quant_api.py:1513` contains a `"\."` in a docstring that now triggers `SyntaxWarning: invalid escape sequence` — cosmetic, but a candidate micro-fix to mention upstream.
- **Int4 on CPU needs more than a GPU note.** Probing `Int4WeightOnlyConfig()` on CPU fails with `ImportError: Requires mslk >= 1.0.0` — the default int4 packing now routes through the `mslk` package. The example's hardware note must mention both the CUDA preference *and* the `mslk` dependency.

### Steps to Reproduce

Prerequisites: macOS/Linux, Python ≥ 3.10 (3.14.4 used here), ~2 GB disk for torch.

1. Clone the fork and check out the working branch:

   ```sh
   git clone https://github.com/mykolas-perevicius/ao.git && cd ao
   git checkout fix-issue-729-moe-quant-example
   ```

2. Confirm the gap — search the repo for any weight-only MoE example:

   ```sh
   grep -ril -E "mixture.of.experts|\bMoE\b" examples/ tutorials/   # no weight-only quant example
   find . -iname '*moe*'   # only torchao/prototype/moe_training/ (training, not inference quant)
   ```

3. Set up the environment:

   ```sh
   python3 -m venv .venv && .venv/bin/pip install torch numpy
   USE_CPP=0 .venv/bin/pip install -e . --no-build-isolation
   ```

4. Run the reproduction script ([`contributions/repro/quantize_moe_repro.py`](./contributions/repro/quantize_moe_repro.py)): it builds an 8-expert top-2 token-choice MoE block, applies `Int8WeightOnlyConfig` to the expert `nn.Linear` layers only (router kept fp32, mirroring `examples/quantize_llama_4.py`), then checks weight types, serialized size, a forward pass, and SQNR — and probes the int4 config on CPU.

5. **Expected** (what a user looking for guidance should find): a runnable weight-only MoE example under `examples/`. **Actual:** none exists (step 2) — the gap is real.

6. **Expected** (maintainer's claim in the issue body): `quantize_()` works on an MoE model with no core changes. **Actual:** confirmed — expert weights become `Int8Tensor` (int8 qdata), the router stays `torch.float32`, serialized state_dict shrinks 8.40 MB → 2.15 MB (3.90x), and the forward pass runs with **SQNR 45.1 dB** vs the fp32 baseline. Identical results across two seeded runs. The int4 variant is *not* CPU-runnable (`ImportError: Requires mslk >= 1.0.0`).

### Reproduction Evidence

- **Branch in my fork:** <https://github.com/mykolas-perevicius/ao/tree/fix-issue-729-moe-quant-example>
- **Commit showing reproduction:** [`89b788b`](https://github.com/mykolas-perevicius/su26-ai301-contribution/commit/89b788b84518faf981071e2355c303fcb1b14f86) — adds [`contributions/repro/quantize_moe_repro.py`](./contributions/repro/quantize_moe_repro.py) (script) and [`contributions/repro/repro-output-2026-06-10.txt`](./contributions/repro/repro-output-2026-06-10.txt) (captured output, both runs).
- **Screenshots/logs:** captured output file above; full session transcript in the [working log](./contributions/pytorch-ao-729.md).
- **My findings:** the gap is real and the feasibility claim holds, so the contribution is purely additive (a new example, no core changes). Two design-relevant discoveries: int8 weight-only is the right CPU-portable default (int4 needs `mslk` + CUDA), and real HF MoE checkpoints store experts as fused 3D weights that need `FqnToConfig` + `PerRow(1)` rather than plain module swap — worth a docstring note in the example.

---

## Solution Approach

### Analysis

For a feature/example issue, the "root cause" is why the gap exists rather than why code misbehaves: `quantize_()` (`torchao/quantization/quant_api.py:275`) walks the module tree and swaps `nn.Linear` weights for quantized tensor subclasses — it is architecture-agnostic, so MoE support never required (or received) any MoE-specific code, and consequently no MoE-specific artifact (example, doc, or test) was ever written. The Phase II reproduction confirms the API works on an MoE block unmodified: expert weights become `Int8Tensor`, the model shrinks 3.90x, and outputs stay within 45 dB SQNR of fp32. The fix is therefore purely additive documentation-by-example.

One real wrinkle the example should surface: toy/simple MoEs use `nn.Linear` experts (plain `quantize_()` + `filter_fn` works), but production HF checkpoints (e.g. Llama-4 Scout) store experts as fused 3D `(B, K, N)` parameters, which need `FqnToConfig` with parameter-fqn regexes and `PerRow(1)` granularity — exactly the pattern `examples/quantize_llama_4.py` uses for float8.

### Proposed Solution

Add a runnable example that applies weight-only quantization to a small MoE model via `quantize_()`, with a brief doc note; no new core functionality.

**Acceptance criteria (confirm with maintainer / adjust per baseline):**

- New example builds or loads a small MoE model and applies weight-only quantization via `quantize_()`.
- Runs end-to-end and shows the effect (e.g., dtype/size change and a successful forward pass).
- Includes run instructions and a hardware note (CPU-runnable with `Int8WeightOnlyConfig`; int4/tinygemm path prefers a CUDA GPU).
- Follows torchao example conventions (location, style, BSD-3 license header) and passes lint/CI.
- Referenced from docs / `examples/README.md` if the project indexes examples there.

**Open questions for the maintainer (already asked in my claim comment):** preferred MoE model (toy vs real); example location (`examples/` vs `examples/inference/`); CPU vs GPU.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** Demonstrate weight-only quantization on an MoE model via `quantize_()`.

**Match:** `examples/quantize_llama_4.py` (script structure / license header); `Int8WeightOnlyConfig` / `Int4WeightOnlyConfig` usage.

**Plan:**

1. Create `examples/quantize_moe.py`: BSD-3 Meta license header + module docstring with run instructions and a hardware note (int8 = CPU-portable default; int4 requires CUDA + `mslk`). Model: small token-choice top-2 MoE block (router + `nn.ModuleList` of `nn.Linear` experts) — same shape as the validated repro script.
2. Quantize the expert linears only, keeping the router in high precision (via `filter_fn` or `FqnToConfig`), mirroring `quantize_llama_4.py`'s experts-only design.
3. Make the script self-verifying: print before/after weight types and serialized sizes, run a forward pass, report SQNR vs the fp32 baseline, and exit non-zero if any check fails.
4. Add a `--dtype int8|int4` flag (int4 gated behind the hardware/dependency check) and a docstring note pointing users with fused-3D-expert HF checkpoints at the `FqnToConfig` + `PerRow(1)` pattern.
5. Optionally add an entry to `examples/README.md` — precedent PR [#3408](https://github.com/pytorch/ao/pull/3408) (which added `quantize_llama_4.py`) did not, so confirm with the maintainer.
6. Comment on issue #729 with the Phase II findings and the open questions (toy vs HF model; `examples/` vs `examples/inference/`).

**Implement:** branch [`fix-issue-729-moe-quant-example`](https://github.com/mykolas-perevicius/ao/tree/fix-issue-729-moe-quant-example) — commits land here in Phase III.

**Review:** self-review against `CONTRIBUTING.md` (fork + branch from `main` — done); run `scripts/run_ruff_fix.sh` / pre-commit (CI lints via `ruff_linter.yml`); match the repo's commit style (short imperative subjects; PRs are squash-merged with the PR number appended).

**Evaluate:** CI does **not** execute `examples/` scripts (only `tutorials/` runs via `run_tutorials.yml`; examples are only linted), so verification is manual + script-internal: run the example end-to-end on CPU at least twice (it self-asserts weight types, ≥3x size reduction, and an SQNR floor), keep ruff/pre-commit clean, and — stretch — validate the int4 path on a CUDA machine to confirm the hardware note.

---

## Testing Strategy

> *Phase III — coming soon.*

### Unit Tests

- [ ] *Phase III.*

### Integration Tests

- [ ] *Phase III.*

### Manual Testing

> *Phase III — run the example on CPU (int8 weight-only) and capture output.*

---

## Implementation Notes

> *Phase III.* Dated progress entries go in the [working log](./contributions/pytorch-ao-729.md); key milestones will be summarized here.

### Code Changes

- **Files modified:** *Phase III.*
- **Key commits:** *Phase III.*
- **Approach decisions:** *Phase III.*

---

## Pull Request

**PR Link:** *Phase IV.*

**PR Description:** *Phase IV.*

**Maintainer Feedback:**
- *Phase IV.*

**Status:** Not yet opened.

---

## Learnings & Reflections

### Technical Skills Gained

*Phase IV.*

### Challenges Overcome

*Phase IV.*

### What I'd Do Differently Next Time

*Phase IV.*

---

## Resources Used

- Issue: <https://github.com/pytorch/ao/issues/729>
- Maintainer statement (`quantize_` already supports MoE) — issue body by @msaroufim (torchao maintainer): <https://github.com/pytorch/ao/issues/729>
- Context comment (@felipemello1, jamba usage): <https://github.com/pytorch/ao/issues/729#issuecomment-2305152753>
- torchao `quantize_` API (`torchao.quantization.quantize_`) and weight-only configs `Int8WeightOnlyConfig` / `Int4WeightOnlyConfig`
- Example convention reference: `examples/quantize_llama_4.py`; `examples/README.md`
- torchao dev/setup: `CONTRIBUTING.md` (fork + branch from `main`), `dev-requirements.txt`, `pyproject.toml`
