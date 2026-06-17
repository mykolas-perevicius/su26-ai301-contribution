# PC (CUDA) kickoff — full hand-off prompt

This resumes the pytorch/ao#729 contribution on a CUDA PC under WSL2. All prior work was
on a Mac (CPU/MPS only); the GPU paths have never run. This prompt has Claude do
everything: clone, set up, validate on GPU, document, push.

## You do (one-time bootstrap)

```sh
# 1. Install Claude Code (native installer; no Node needed)
curl -fsSL https://claude.ai/install.sh | bash
#    alt: npm install -g @anthropic-ai/claude-code   (needs Node 18+)

# 2. Python venv tooling — the one sudo step (so Claude never hits a password prompt)
sudo apt update && sudo apt install -y python3-venv python3-pip git

# 3. Git auth so the final push works (interactive)
gh auth login          # if gh missing: sudo apt install -y gh   (or use a PAT credential helper)

# 4. Workspace + launch Claude in the WSL filesystem (NOT /mnt/c)
mkdir -p ~/oss && cd ~/oss && claude
```

## Then paste this prompt and let it run

> Set up and resume an in-progress open-source contribution on this CUDA PC (WSL2), then
> validate it on the GPU. Fresh session, no prior context — everything is in two GitHub
> repos you'll clone now. Work autonomously through steps 1–6; stop before any
> outward-facing action.
>
> SETUP (in the current directory):
> 1. Clone both repos as siblings here:
>    `git clone https://github.com/mykolas-perevicius/su26-ai301-contribution.git`
>    `git clone https://github.com/mykolas-perevicius/ao.git`
> 2. In `ao/`, `git checkout fix-issue-729-moe-quant-example` (commits 40f18c99d = example,
>    2b96e6f35 = test; this is my fork of pytorch/ao).
> 3. Read for context: `su26-ai301-contribution/contributions/gpu-validation-runbook.md`
>    (the procedure), its `README.md` (Phase III done; Phase IV on hold pending this GPU
>    check), `contributions/pytorch-ao-729.md` (working log), and `ao/CLAUDE.md` +
>    `ao/examples/quantize_moe.py` (code under validation).
>
> VALIDATE (follow the runbook from the venv-creation step onward — cloning is already done):
> 4. Confirm the GPU: `nvidia-smi`, and `torch.cuda.is_available()` after install.
> 5. Create a venv, `USE_CPP=0` editable-install torchao (+ pytest parameterized expecttest),
>    run the int8 and int4 `--device cuda` paths of `examples/quantize_moe.py` plus the full
>    `test/quantization/test_quant_api.py` (GPU tests unskip on CUDA). Capture everything to
>    `gpu-validation-<hostname>.txt`. If `pip install mslk` (int4 dep) fails, record the
>    error and continue — int8 still validates the `--device` path.
>
> DOCUMENT:
> 6. Copy the output file into `su26-ai301-contribution/contributions/repro/`; update the
>    README "Manual Testing" with the CUDA results; add a dated CUDA-validation entry to the
>    working log. Commit and push to the log repo. If the push fails on auth, stop and tell
>    me — don't retry blindly.
>
> THEN:
> 7. Draft the Phase IV PR description (problem, what the example does, CPU+CUDA test
>    evidence, the open maintainer questions on issue #729). Do NOT open the PR or comment
>    upstream — present the draft for my approval.
>
> If you hit a sudo/password prompt, pause and ask me to run that one command.
