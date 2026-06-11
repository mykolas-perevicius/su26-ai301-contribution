# pytorch/ao #729 — working log (Contribution 1)

Deep-dive companion to the top-level [README](../README.md), which holds the report essentials (why chosen, problem statement, solution approach, status, resources). This file holds everything else: file/line-level investigation detail, dated baseline evidence with command transcripts, and ongoing work notes as phases progress.

**Issue:** <https://github.com/pytorch/ao/issues/729> · **Fork:** <https://github.com/mykolas-perevicius/ao> · **Branch:** [`fix-issue-729-moe-quant-example`](https://github.com/mykolas-perevicius/ao/tree/fix-issue-729-moe-quant-example) *(renamed from `feat/moe-quant-example` in Phase II)*

---

## Key locations (quick reference, as of 2026-06-08)

- `quantize_` — `torchao/quantization/quant_api.py:275`; exported from `torchao/quantization/__init__.py` (`from torchao.quantization import quantize_`).
- `Int8WeightOnlyConfig` — `quant_api.py:727` — CPU-runnable (good for a portable example / CI).
- `Int4WeightOnlyConfig` — `quant_api.py:527` — int4 tinygemm kernel, GPU-preferred. Legacy `int4_weight_only` / `int8_weight_only` callables also exist.
- **Example home (primary):** `examples/quantize_moe.py` + entry in `examples/README.md`, mirroring `examples/quantize_llama_4.py` (BSD-3 Meta header + module docstring).
- **Example home (alternative — ask maintainer):** `examples/inference/quantize_moe.py`. That directory does not exist today, but `quant_api.py:542` has a docstring `literalinclude` pointing at `examples/inference/int4_weight_only.py`, hinting the project may prefer an `examples/inference/` home.
- **Maintainer context:** issue body by @msaroufim ("`quantize_()` should already support MoE"), reviewers cc'd @jcaip and @cpuhrsch; context comment by @felipemello1 (jamba usage): <https://github.com/pytorch/ao/issues/729#issuecomment-2305152753>

---

## Baseline / Investigation Log

### 2026-06-08 — Phase I baseline (no bug repro; this is a feature/example)

#### Environment

- macOS (Darwin 25.5), git 2.54.0, gh 2.93.0.
- Fork: <https://github.com/mykolas-perevicius/ao> (isFork:true, parent pytorch/ao).
- Local clone: `~/Developer/oss/forks/ao`; remotes `origin` → my fork, `upstream` → `https://github.com/pytorch/ao.git`; `git fetch upstream` done; branch `feat/moe-quant-example` created.

#### Re-verify before work (issue still free)

- `gh issue view 729 -R pytorch/ao` → state OPEN, assignees 0, labels `good first issue`.
- Comments: 2 — @felipemello1 (2024 context) + my own claim (@mykolas-perevicius). No competing claim.
- `gh pr list -R pytorch/ao --state open --search "729"` → returns PR #3416, but the GraphQL timeline cross-ref/connected check returns `[]` → #3416 is a false-positive string match, NOT linked to #729.

#### Commands run (baseline investigation, in `~/Developer/oss/forks/ao`)

```sh
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

#### Findings

1. **No existing weight-only MoE inference example.** `torchao/prototype/moe_training/` is low-precision MoE *training* (distinct feature). `examples/quantize_llama_4.py` quantizes Llama-4 routed experts but to **float8 w8a8 dynamic** (activation + weight), not weight-only. → the weight-only-on-MoE showcase gap is real.
2. **API located.** `quantize_` at `torchao/quantization/quant_api.py:275`, exported from `torchao/quantization/__init__.py` (`from torchao.quantization import quantize_`). Weight-only configs: `Int8WeightOnlyConfig` (`quant_api.py:727`), `Int4WeightOnlyConfig` (`quant_api.py:527`). Legacy `int4_weight_only` / `int8_weight_only` callables also present.
3. **Candidate example location.** Primary: `examples/quantize_moe.py` (top-level single-file convention, mirroring `quantize_llama_4.py`; add to `examples/README.md`). Alternative: `examples/inference/quantize_moe.py` — that dir doesn't exist yet, but `quant_api.py:542` references `examples/inference/int4_weight_only.py` in a docstring `literalinclude`, so confirm the preferred home with the maintainer.
4. **Maintainer statement permalink.** The "`quantize_()` should already support MoE" statement is in the **issue body**, authored by **@msaroufim** (torchao maintainer): <https://github.com/pytorch/ao/issues/729>. Reviewers cc'd in body: @jcaip, @cpuhrsch.
5. **GPU need.** `Int8WeightOnlyConfig` weight-only quant runs end-to-end on **CPU** (good for a portable example / CI). `Int4WeightOnlyConfig` uses the **int4 tinygemm kernel** and prefers a **CUDA GPU** (`quant_api.py:288` notes `device="cuda"` speeds quantization; `:298–302` note the int4 tinygemm optimized path + `torch.compile`).

**STOP point:** Per workflow, stopping after baseline + README enrichment are committed. No example code written, no PR opened.

### 2026-06-10 — Phase II: environment setup, reproduction, plan

#### Re-verify before work

- `gh issue view 729 -R pytorch/ao` → state OPEN, assignees 0, label `good first issue`; 2 comments, last is my own claim (2026-06-09). Still free.

#### Environment setup (commands + outcomes)

```sh
python3 --version                                   # 3.14.4 (Homebrew) — only Python on machine
python3 -m venv ~/Developer/oss/.venv-ao            # venv OUTSIDE the fork (keeps repo clean)
pip index versions torch                            # checked BEFORE installing: cp314 wheels exist (2.9.0–2.12.0)
pip install torch numpy                             # torch 2.12.0, mps available
USE_CPP=0 pip install -e . --no-build-isolation     # editable torchao 0.18.0+git5165bfb03, builds in seconds
python -c "from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig"  # OK
```

Friction log: (1) Python 3.14 wheel availability was the main risk — resolved by checking `pip index versions torch` before installing (cp314 wheels exist since torch 2.9). (2) `USE_CPP=0` needed to skip C++ extension builds. (3) Import emits `SyntaxWarning: "\." is an invalid escape sequence` from `quant_api.py:1513` under Python 3.14 — cosmetic; candidate micro-fix to mention upstream. (4) `Int4WeightOnlyConfig()` on CPU raises `ImportError: Requires mslk >= 1.0.0` — int4's default packing depends on the `mslk` package, not just CUDA.

#### Precedent investigation (for the Match step)

- `git log --follow examples/quantize_llama_4.py` → added in PR **#3408** ("add an example for quantizing LLaMa 4 Scout", merged; commit `59779058e`), later touched by the mslk migration (`9ba99f033`).
- `gh pr view 3408` → **single-file PR** (`examples/quantize_llama_4.py` only — no `examples/README.md` entry), reviewed by **@jcaip** (cc'd on our issue) and **@jerryzh168**. Strong precedent: a single, self-contained example script is an acceptable PR shape.
- `quantize_llama_4.py` design notes: quantizes **routed experts only** (rest of model high-precision) via `FqnToConfig` with `re:.*\.feed_forward\.experts\.(gate_up|down)_proj` parameter regexes; uses `PerRow(1)` granularity because HF Llama-4 expert weights are fused 3D `(B, K, N)` tensors. → Two MoE flavors to acknowledge: `nn.Linear` experts (plain `quantize_` + `filter_fn`) vs fused 3D expert params (`FqnToConfig` + `PerRow(1)`).
- CI check: `grep -r "examples/" .github/workflows/` → only the SAM2 perf dashboard touches `examples/`; **no workflow executes general example scripts**. `ruff_linter.yml` lints; `run_tutorials.yml` only runs `tutorials/`. → example must be self-verifying; CI proof is lint only.

#### Reproduction (feasibility of the maintainer's claim)

- Script: [`repro/quantize_moe_repro.py`](./repro/quantize_moe_repro.py) — 8-expert top-2 token-choice MoE (router + `nn.Linear` experts), `Int8WeightOnlyConfig` applied via `filter_fn` to expert linears only.
- Captured output (two runs): [`repro/repro-output-2026-06-10.txt`](./repro/repro-output-2026-06-10.txt). Results, identical across both seeded runs:
  - expert weight `Parameter/float32` → `Int8Tensor` (qdata `torch.int8`); router stays `Parameter/float32`
  - serialized state_dict 8.40 MB → 2.15 MB (**3.90x smaller**)
  - forward pass OK, **SQNR 45.1 dB** vs fp32 baseline (all 5 self-checks PASS)
  - int4 probe on CPU: `ImportError: Requires mslk >= 1.0.0` → int8 is the right CPU-portable default
- Conclusion: gap real (2026-06-08 baseline) **and** claim verified → contribution is purely additive, no core changes.

#### Branch

- Renamed `feat/moe-quant-example` → `fix-issue-729-moe-quant-example` (program convention: branch named after the issue) and pushed: <https://github.com/mykolas-perevicius/ao/tree/fix-issue-729-moe-quant-example> (tracking set up; currently at upstream `main` HEAD `5165bfb03`, torchao 0.18.0 dev).

#### Edge cases identified for the example (Phase III inputs)

1. Router must stay high-precision (quantizing it changes routing decisions, not just numerics) — use `filter_fn`/`FqnToConfig`.
2. Fused 3D expert weights in real HF checkpoints need `FqnToConfig` + `PerRow(1)` — docstring note, not toy-example scope.
3. int4 path: CUDA-preferred **and** requires `mslk` — gate behind a flag with a clear error message.
4. Tokens routed to zero experts / empty expert batches — the repro's loop already handles `token_idx.numel() == 0`.
5. Python 3.14 `SyntaxWarning` on import — harmless noise users may report; possible separate micro-PR.

**STOP point:** Phase II technical work complete (env + repro + plan documented in README). Remaining user actions: commit/push this log repo, submit check-in form ("Phase II Complete"), optional Slack announcement. No example code written, no PR opened — that is Phase III.
