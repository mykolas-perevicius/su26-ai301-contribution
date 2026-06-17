# PC (CUDA) kickoff — paste this into a fresh Claude Code session

Context: this resumes an in-progress open-source contribution (pytorch/ao#729) on a
CUDA PC under WSL2. The previous work was done on a Mac (CPU/MPS only); the one thing
that machine could not do is validate the GPU paths. That's this session's job.

Open Claude Code at the folder that contains both repos as siblings (`ao/` and
`su26-ai301-contribution/`), then paste the prompt below.

---

You're picking up an in-progress open-source contribution on a CUDA PC (WSL2). This is
a fresh session with no prior context — load it from disk first.

Read before doing anything:
- `su26-ai301-contribution/contributions/gpu-validation-runbook.md` — the exact procedure to run
- `su26-ai301-contribution/README.md` — the contribution report (Phase III complete; Phase IV on hold pending this GPU check)
- `su26-ai301-contribution/contributions/pytorch-ao-729.md` — the detailed working log
- `ao/CLAUDE.md` and `ao/examples/quantize_moe.py` — the code under validation

Current state:
- Both repos are cloned as siblings here. `ao` is my fork of pytorch/ao, checked out on
  branch `fix-issue-729-moe-quant-example` (commits 40f18c99d = example, 2b96e6f35 = test).
- All CPU/MPS validation is done. We are now on a CUDA GPU for the first time.

Your job, work autonomously through steps 1–3:
1. Verify the GPU: `nvidia-smi`, and `torch.cuda.is_available()` after install.
2. Follow the runbook — create a venv, `USE_CPP=0` editable-install torchao, run the int8
   and int4 `--device cuda` paths plus the full `test/quantization/test_quant_api.py`
   (GPU tests unskip here), capturing all output to `gpu-validation-<hostname>.txt`.
3. Fold results into the log repo: update README "Manual Testing" and add a dated CUDA
   entry to the working log; copy the output file into
   `su26-ai301-contribution/contributions/repro/`; commit and push to the log repo.

Then prep Phase IV: draft the PR description (Understand/Match/Plan/test evidence, noting
CPU+CUDA validation and the open maintainer questions). Do NOT open the PR or comment on
the issue yet — those are outward-facing; present the draft for my approval first.

Stop and check with me before any outward-facing action.
