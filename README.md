# Contribution 1: MoE example — weight-only quantization showcase (pytorch/ao #729)

**Contribution Number:** 1  
**Student:** Mykolas Perevicius — GitHub [@mykolas-perevicius](https://github.com/mykolas-perevicius) — AI301 Section 1A  
**Issue:** <https://github.com/pytorch/ao/issues/729>  
**Fork:** <https://github.com/mykolas-perevicius/ao>  
**Status:** Phase IV — PR [#4507](https://github.com/pytorch/ao/pull/4507) submitted to pytorch/ao (2026-06-17), awaiting review

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

**Implement:** branch [`fix-issue-729-moe-quant-example`](https://github.com/mykolas-perevicius/ao/tree/fix-issue-729-moe-quant-example) — Phase III commits: [`40f18c99d`](https://github.com/mykolas-perevicius/ao/commit/40f18c99d) (example), [`2b96e6f35`](https://github.com/mykolas-perevicius/ao/commit/2b96e6f35) (test).

**Review:** self-review against `CONTRIBUTING.md` (fork + branch from `main` — done); run `scripts/run_ruff_fix.sh` / pre-commit (CI lints via `ruff_linter.yml`); match the repo's commit style (short imperative subjects; PRs are squash-merged with the PR number appended).

**Evaluate:** CI does **not** execute `examples/` scripts (only `tutorials/` runs via `run_tutorials.yml`; examples are only linted), so verification is manual + script-internal: run the example end-to-end on CPU at least twice (it self-asserts weight types, ≥3x size reduction, and an SQNR floor), keep ruff/pre-commit clean, and — stretch — validate the int4 path on a CUDA machine to confirm the hardware note.

---

## Testing Strategy

### Unit Tests

- [x] `TestQuantFlow.test_int8_weight_only_moe_experts_only` — added to `test/quantization/test_quant_api.py`, following that file's existing conventions (module-level toy model like `ToyLinearModel`, `compute_error` for SQNR, `Int8Tensor` isinstance asserts). It quantizes a new `ToyMoEModel`'s expert linears via `filter_fn`, then asserts every expert weight becomes `Int8Tensor`, the router weight stays unquantized, and SQNR vs float32 exceeds 25 dB. Passes in ~1 s on CPU.

### Integration Tests

- [x] Not applicable by project design: torchao's CI does not execute `examples/` scripts (lint only, via `ruff_linter.yml`). The example is therefore **self-verifying** — it asserts expert weights are quantized, the router is not, the model shrank ≥1.5x, and SQNR > 25 dB, exiting non-zero on any failure.

### Manual Testing

- `python examples/quantize_moe.py` on CPU, twice: identical output both runs — experts → `Int8Tensor`, router stays `torch.float32`, serialized 8.40 → 2.15 MB (3.90x), SQNR 45.1 dB, all assertions pass.
- `--dtype int4` on CPU: prints the documented hardware warning and exits non-zero (`mslk` + CUDA required) — the guard behaves as designed.
- Full `test/quantization/test_quant_api.py` on CPU: 12 passed (including the new test), 32 skipped (GPU-only), 2 failed — both failures reproduce on a clean checkout without my changes (verified via `git stash`; they're Apple-silicon/MPS backend gaps: missing `torch.mps.reset_peak_memory_stats`, and `Float8_e4m3fn` unsupported on MPS), so they're unrelated to this change.
- Lint: `ruff check` + `ruff format` (project-pinned v0.11.6) clean on both changed files; `scripts/check_copyright_header.py` passes on the new example.
- **CUDA validation: complete (2026-06-17, RTX 3090, SM 8.6, torch 2.12.1+cu130, CUDA 13.0).** Full transcript: [`contributions/repro/gpu-validation-MYKO-HQ.txt`](./contributions/repro/gpu-validation-MYKO-HQ.txt).
  - `python examples/quantize_moe.py --device cuda` (int8): experts → `Int8Tensor`, router `torch.float32`, serialized 8.40 → 2.15 MB (3.90x), SQNR 45.1 dB, all assertions pass — **identical to the CPU run**, so the `--device cuda` path is now exercised end-to-end.
  - `pytest test/quantization/test_quant_api.py -k moe`: **1 passed** on CUDA.
  - Full `test/quantization/test_quant_api.py` on CUDA: **18 passed, 28 skipped, 0 failed**. Six more tests pass than on the Mac (GPU-only tests unskip on CUDA), and the **2 MPS failures seen on Apple silicon do not occur on Linux/CUDA** — confirming they were backend-specific, not caused by this change. All 28 skips are hardware-gated (`Need SM 8.9+` / `Checkpoints are produced in SM90+` for float8 on this SM 8.6 card) or pre-existing unconditional skips — none introduced by this change.
  - `--dtype int4 --device cuda`: **could not be exercised end-to-end** — it raises `ImportError: Requires mslk >= 1.0.0` on CUDA just as it does on CPU. The public PyPI `mslk` is a 0.0.0 placeholder (894-byte empty wheel); torchao imports `from mslk.quantize.shuffle import int4_row_quantize_zp`, and `torchao/utils.py:1226` gates real availability on `is_fbcode()` — so the int4 default path depends on Meta's internal FBGEMM-GenAI `mslk`, which isn't installable from public PyPI at the required version. The runbook's `pip install mslk` step is therefore a dead end on public infrastructure. The example already gates int4 behind a `--dtype` flag with a hardware/dependency warning; this empirically confirms the docstring's "requires the `mslk` package" note and surfaces an open question for the maintainer (see PR draft).

---

## Implementation Notes

> Dated raw transcripts live in the [working log](./contributions/pytorch-ao-729.md); milestones are summarized here.

### Week 1 Progress (2026-06-10 → 2026-06-11)

**What I built:**

- `examples/quantize_moe.py` (new, 166 lines): self-contained token-choice top-2 MoE block (softmax router + `nn.Linear` experts), quantized experts-only via `quantize_(…, filter_fn=is_expert_linear)`; prints weight types, serialized sizes, and SQNR; asserts all checks (exits non-zero on failure); `--dtype int8|int4` and `--device` flags with the int4 hardware guard; docstring points fused-3D-expert users at `FqnToConfig` + `PerRow(1)`.
- `test/quantization/test_quant_api.py` (+65 lines): `ToyMoEModel` helper mirroring `ToyLinearModel` conventions, plus `test_int8_weight_only_moe_experts_only`.
- Rebased the branch onto upstream `main` (`abea9e0c4`) before starting, per the daily rhythm.

**Challenges faced:**

- The full test file showed 2 failures after my change. Rather than assume they were mine, I `git stash`-ed the change and re-ran: both failures reproduce on the clean tree (MPS backend gaps on Apple silicon), so they're pre-existing. Restored the change and documented them in Testing Strategy.
- Writing the test idiomatically: instead of inventing scaffolding, I modeled it on the file's existing patterns — module-level toy model, `compute_error` SQNR floor, `Int8Tensor` isinstance asserts — so it reads like the surrounding tests.

**Commits this week:** [`40f18c99d`](https://github.com/mykolas-perevicius/ao/commit/40f18c99d) (example), [`2b96e6f35`](https://github.com/mykolas-perevicius/ao/commit/2b96e6f35) (test).

### Code Changes

- **Files modified:** `examples/quantize_moe.py` (new); `test/quantization/test_quant_api.py` (`ToyMoEModel` + one test).
- **Key commits:** [`40f18c99d`](https://github.com/mykolas-perevicius/ao/commit/40f18c99d) — Add weight-only quantization MoE example; [`2b96e6f35`](https://github.com/mykolas-perevicius/ao/commit/2b96e6f35) — Add test for int8 weight-only quantization of MoE experts.
- **Approach decisions:**
  - **int8 default, int4 behind a flag** — int8 weight-only runs anywhere (CPU-portable, CI-friendly); the default int4 path needs CUDA + `mslk`, so it's opt-in with an explicit warning.
  - **Router deliberately left unquantized** — quantizing the router changes token-to-expert assignments (a behavior change, not just numerics); the experts-only `filter_fn` mirrors `quantize_llama_4.py`'s experts-only design.
  - **Added a unit test beyond the single-file precedent (#3408)** — CI never executes examples, so without a test, MoE support stays demonstrated-but-unguarded. The test makes it regression-checked; it's an easy drop in review if maintainers prefer the single-file shape.
  - **Deferred the `examples/README.md` entry** — precedent PR #3408 didn't add one; the open question to maintainers (location, `examples/` vs `examples/inference/`) is still pending, so this stays a PR-review decision.

### Engineering judgment beyond the minimum

Mapping the work to behaving like a contributing engineer (Phase III +3 bonus criteria):

- **Edge cases maintainers hadn't mentioned, surfaced in code/docs:** the router must stay high-precision (quantizing it changes token-to-expert *routing*, not just numerics) — enforced by the experts-only `filter_fn`; real HF checkpoints store experts as fused 3D `(num_experts, K, N)` params needing `FqnToConfig` + `PerRow(1)` — called out in the example docstring; empty-expert-batch guard (`token_idx.numel() == 0`); int4's hidden dependency is `mslk`, not merely "needs a GPU."
- **Reused project-specific helpers instead of hand-rolling:** `torchao.quantization.utils.compute_error` for SQNR; the test mirrors the file's own `ToyLinearModel` shape, `Int8Tensor` isinstance asserts, and `sqnr` threshold pattern; BSD-3 header verified with the repo's own `scripts/check_copyright_header.py`.
- **Descoped sensibly with documented notes:** deferred the `examples/README.md` entry (matching precedent #3408, location question still open) and flagged the unit test as a clean drop if maintainers prefer the single-file shape — both written down rather than decided silently.
- **Went past the brief and reported the finding:** validated on a real CUDA GPU (RTX 3090) ahead of the PR and, doing so, discovered the int4 `mslk` dependency is **unsatisfiable from public PyPI** (the published `mslk` is a 0.0.0 stub; the real lib is `is_fbcode`-gated). Corrected our own GPU runbook accordingly and turned it into a concrete open question for the maintainer rather than a silent skip. Also caught a pre-existing `SyntaxWarning` (`quant_api.py:1499`) offered as a separate one-line micro-PR.
- **Verified, didn't assume:** when the full test file showed 2 failures on macOS, `git stash`-ed the change and re-ran to prove they were pre-existing MPS backend gaps — later confirmed on CUDA, where those 2 failures don't occur at all.

---

## Pull Request

**PR Link:** <https://github.com/pytorch/ao/pull/4507> — *Add weight-only quantization MoE example* (base `pytorch/ao:main` ← head `mykolas-perevicius:fix-issue-729-moe-quant-example`).

**PR commits (post-rebase):** [`cb65991c3`](https://github.com/pytorch/ao/pull/4507/commits/cb65991c3) (example) · [`ba6c62b09`](https://github.com/pytorch/ao/pull/4507/commits/ba6c62b09) (test). The branch was rebased onto the latest upstream `main` at submission, so these supersede the Phase III SHAs `40f18c99d`/`2b96e6f35` (identical content, new hashes). Diff: 2 files, +231/−0; `Closes #729`.

**PR Description:** Adds a runnable example (`examples/quantize_moe.py`) that applies int8 weight-only quantization to a small MoE block's expert linears via `quantize_(…, filter_fn=…)` — router kept in high precision — and self-verifies the ~3.9x size reduction and 45.1 dB SQNR, plus a unit test (`test_int8_weight_only_moe_experts_only`). Purely additive, no core changes. Full description (why → what → test plan → acceptance criteria → open questions) is mirrored in [`contributions/pr-draft.md`](./contributions/pr-draft.md) and posted on the PR.

**Maintainer Feedback:**
- **2026-06-17** — Opened PR #4507 against `pytorch/ao:main` (ready for review, not draft). Posted a review-request [comment](https://github.com/pytorch/ao/pull/4507#issuecomment-4736932731) @mentioning **@jcaip** (cc'd on #729, reviewed the analogous `quantize_llama_4.py` PR #3408) and **@msaroufim** (issue author). Raised 4 open questions on the PR: example location (`examples/` vs `examples/inference/`), keep/drop the unit test, an `examples/README.md` entry, and the supported public path for int4 given `mslk` is `is_fbcode`-gated. Awaiting first review.
- *Meta CLA:* required for merge (one-time, <https://code.facebook.com/cla>) — will be prompted by the CLA bot on the PR.

**Status:** Awaiting review (PR open since 2026-06-17).

---

## Learnings & Reflections

### Technical Skills Gained

- **torchao's `quantize_()` config-object workflow.** I now understand the model in practice: `quantize_()` walks the module tree and, for modules matching `filter_fn`, swaps the `nn.Linear` weight `Parameter` for a quantized tensor subclass (`Int8Tensor`) — architecture-agnostic, which is exactly why it "just works" on an MoE block. Using `filter_fn` to target only the expert linears (and deliberately *not* the router) was the key design lever.
- **Reading quantization quality, not just running it.** Used the project's own `compute_error` to report SQNR (45.1 dB) rather than eyeballing outputs, and learned where the thresholds live in the existing tests (`sqnr >= 16.5`) so my assertions matched house style.
- **Writing tests that look native to the repo.** Modeling `ToyMoEModel` on the file's `ToyLinearModel` (module-level, `example_inputs()`, `nn.Sequential` experts to avoid an extra import) taught me more about the codebase's conventions than any doc.
- **GPU/CUDA bring-up and reading a test suite's hardware gates.** Setting up torch+CUDA on WSL2, then learning to distinguish *hardware-gated skips* (`Need SM 8.9+`, `Checkpoints are produced in SM90+` — float8 on an SM 8.6 card) from *real failures*. The same file is "12 passed / 2 failed" on a Mac and "18 passed / 0 failed" on CUDA — same code, different backend.
- **Dependency archaeology.** Tracing `ImportError: Requires mslk >= 1.0.0` from `quant_api.py` → `int4_tensor.py:23` → `from mslk.quantize.shuffle import …` → `torchao/utils.py:1226`'s `is_fbcode()` gate, and recognizing that the public PyPI `mslk` (0.0.0, 894 bytes) is a placeholder, not the real kernel library.

### Challenges Overcome

- **Not assuming a failure is mine.** When the full test file showed 2 failures, the instinct is to suspect your own change. Instead I `git stash`-ed and re-ran on a clean tree — the failures reproduced, so they were pre-existing MPS backend gaps. The CUDA run later closed the loop: those failures don't exist on Linux/CUDA at all. Lesson: isolate the variable before drawing a conclusion.
- **Turning a dead end into a contribution.** The int4 `--device cuda` path — the whole point of the GPU validation — couldn't be run end-to-end, because `mslk >= 1.0.0` isn't installable from public PyPI. Rather than silently skip it, I traced *why*, corrected our own GPU runbook (the `pip install mslk` step was a dead end), and converted it into a concrete, well-scoped question for the maintainer. A blocker became signal.
- **Environment friction without a full reinstall.** Python-version/wheel risk and the C++ extension build were handled with `pip index versions` checks up front and `USE_CPP=0` — small habits that avoided losing a day to a build.

### What I'd Do Differently Next Time

- **Front-load the maintainer questions.** I built against the primary plan (toy model, top-level `examples/quantize_moe.py`, int8 default) while the location/model-shape questions were still open. That was a reasonable bet, but asking earlier — before writing code — would de-risk rework if the maintainer prefers `examples/inference/` or a real checkpoint.
- **Probe the hard dependency sooner.** I discovered int4's `mslk` requirement is unsatisfiable on public infra only at GPU-validation time. Probing `Int4WeightOnlyConfig()`'s real import chain in Phase II would have surfaced it a week earlier and shaped the docstring from the start.
- **Tighter commit cadence on the fork branch.** Both code commits landed the same day. Splitting the example and the test across days (each is independently meaningful) would have made the build history easier to follow.

**Teachable insight:** on a mature project, the highest-leverage move for a first-timer isn't writing more code — it's *reusing the project's own vocabulary* (its test helpers, its error metric, its example conventions) and *reporting what you find honestly*, including the parts that don't work. The `mslk` dead end is more useful to a maintainer than a green check would have been, because it documents a real gap in the public int4 story.

---

## Resources Used

- Issue: <https://github.com/pytorch/ao/issues/729>
- Maintainer statement (`quantize_` already supports MoE) — issue body by @msaroufim (torchao maintainer): <https://github.com/pytorch/ao/issues/729>
- Context comment (@felipemello1, jamba usage): <https://github.com/pytorch/ao/issues/729#issuecomment-2305152753>
- torchao `quantize_` API (`torchao.quantization.quantize_`) and weight-only configs `Int8WeightOnlyConfig` / `Int4WeightOnlyConfig`
- Example convention reference: `examples/quantize_llama_4.py`; `examples/README.md`
- torchao dev/setup: `CONTRIBUTING.md` (fork + branch from `main`), `dev-requirements.txt`, `pyproject.toml`
