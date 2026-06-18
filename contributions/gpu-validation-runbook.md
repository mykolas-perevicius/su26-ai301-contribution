# GPU validation runbook — pytorch/ao#729 (run this from the GPU PC)

**Why:** Phase IV (PR) is on hold until this is done. The example's `--dtype int4 --device cuda` path is documented but has never been executed (our Mac is CPU/MPS-only), the `--device cuda` flag needs one real exercise, and on a CUDA machine the 32 GPU-skipped tests in `test/quantization/test_quant_api.py` actually run.

**What you're validating:** branch [`fix-issue-729-moe-quant-example`](https://github.com/mykolas-perevicius/ao/tree/fix-issue-729-moe-quant-example) — commits `40f18c99d` (example) + `2b96e6f35` (test). Context: [working log](./pytorch-ao-729.md), Phase III entry.

## Prerequisites

- NVIDIA GPU + driver installed (`nvidia-smi` works).
- Linux or **WSL2** (used for the 2026-06-17 run). Note: the int4 path's `mslk` dependency is not satisfiable from public PyPI on any OS (see int4 note below); int8-on-CUDA is the path this runbook proves.
- Python ≥ 3.10.

## Steps

```sh
git clone https://github.com/mykolas-perevicius/ao.git && cd ao
git checkout fix-issue-729-moe-quant-example
python3 -m venv .venv && source .venv/bin/activate
pip install torch                                  # Linux default wheels include CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
USE_CPP=0 pip install -e . --no-build-isolation
pip install pytest parameterized expecttest

{
  echo "== env =="
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| sm', torch.cuda.get_device_capability(0))"

  echo "== int8 on CUDA =="
  python examples/quantize_moe.py --device cuda

  echo "== int4 on CUDA (the unproven path) =="
  # NOTE (validated 2026-06-17): `pip install mslk` does NOT enable this path — the public
  # PyPI `mslk` is a 0.0.0 placeholder (894-byte empty wheel). torchao imports
  # `from mslk.quantize.shuffle import int4_row_quantize_zp` and gates real availability on
  # `is_fbcode()` (torchao/utils.py:1226); the required mslk>=1.0.0 is Meta-internal
  # (FBGEMM-GenAI), so int4 raises `ImportError: Requires mslk >= 1.0.0` on CPU and CUDA alike.
  # Skip on public infra — int8 above already validates the `--device cuda` path.
  python examples/quantize_moe.py --dtype int4 --device cuda   # expected: ImportError until mslk>=1.0.0 is available

  echo "== new MoE test =="
  pytest test/quantization/test_quant_api.py -k moe -v

  echo "== full quant_api test file (GPU tests unskip here) =="
  pytest test/quantization/test_quant_api.py

} 2>&1 | tee gpu-validation-$(hostname).txt
```

## Expected results

| Check | Expect |
|-------|--------|
| int8 `--device cuda` | experts → `Int8Tensor`, router `float32`, ~3.9x smaller, SQNR ≳ 40 dB, "succeeded" |
| int4 `--device cuda` | **blocked on public infra (confirmed 2026-06-17):** `ImportError: Requires mslk >= 1.0.0` — public PyPI `mslk` is a 0.0.0 stub; real `mslk>=1.0.0` is Meta-internal (`is_fbcode`-gated). int8 already validates the `--device cuda` path; record the error and move on. |
| `-k moe` test | 1 passed |
| full test file | new failures only if GPU-specific and pre-existing — note count + names; SM < 8.9 keeps float8 tests skipped (fine, note it) |

Fallbacks: int4 path → **expect it to fail** (`ImportError: Requires mslk >= 1.0.0`); record the error and continue. int8-on-CUDA still validates `--device`, and the int4 docstring note stands on its documented requirements. `torch.cuda.is_available()` is False → reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Bring the results back

```sh
git clone https://github.com/mykolas-perevicius/su26-ai301-contribution.git
cp ../ao/gpu-validation-*.txt su26-ai301-contribution/contributions/repro/
cd su26-ai301-contribution
git add contributions/repro/ && git commit -m "Contribution 1: GPU validation results (pytorch/ao#729)" && git push
```

Then, in Claude Code (any machine), say: **"GPU validation results are in `contributions/repro/gpu-validation-*.txt` — fold them into the log and start Phase IV."** That covers updating the README's Manual Testing section, the working log, and drafting the PR with "validated on CPU (macOS) + CUDA (<GPU name>)".
