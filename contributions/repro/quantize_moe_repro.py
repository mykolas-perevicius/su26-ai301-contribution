# Reproduction script for pytorch/ao#729 (Phase II evidence, AI301 Contribution 1).
#
# The issue is a feature/example request: torchao has no runnable example of
# weight-only quantization on a Mixture-of-Experts model. "Reproduction" here
# means two things:
#   (a) the gap is real (checked separately: no such example exists in examples/),
#   (b) the maintainer's claim holds — `quantize_()` already supports MoE — so
#       the planned example is feasible without core changes.
#
# This script builds a small token-choice top-2 MoE block with nn.Linear experts,
# applies Int8 weight-only quantization to the expert layers only (router kept in
# high precision, mirroring examples/quantize_llama_4.py's design), and verifies:
#   1. expert weights become torchao quantized tensor subclasses (int8 data),
#   2. the router weight is untouched,
#   3. serialized state_dict shrinks ~4x,
#   4. a forward pass runs and stays numerically close to the fp32 baseline (SQNR).
#
# Run (from the workspace root, venv created per the working log):
#   .venv-ao/bin/python ai301-contribution-log/contributions/repro/quantize_moe_repro.py
#
# Also probes Int4WeightOnlyConfig on CPU to document the hardware constraint
# for the example's docstring (int4 tinygemm path prefers CUDA).

import copy
import io

import torch
import torch.nn.functional as F
from torch import nn

from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

torch.manual_seed(0)


class Expert(nn.Module):
    """Standard FFN expert: up-projection, SiLU, down-projection."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class MoE(nn.Module):
    """Token-choice top-k MoE block (router + nn.Linear experts)."""

    def __init__(self, dim=256, hidden_dim=512, num_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(Expert(dim, hidden_dim) for _ in range(num_experts))
        self.top_k = top_k

    def forward(self, x):
        batch, seq, dim = x.shape
        flat = x.reshape(-1, dim)
        probs = F.softmax(self.router(flat), dim=-1)
        weights, expert_idx = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            routed = (expert_idx == e)
            token_idx = routed.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_idx.numel() == 0:
                continue
            gate = (weights * routed).sum(dim=-1)[token_idx].unsqueeze(-1)
            out[token_idx] += gate * expert(flat[token_idx])
        return out.reshape(batch, seq, dim)


def state_dict_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def sqnr_db(reference, candidate):
    return (
        20 * torch.log10(reference.norm() / (reference - candidate).norm()).item()
    )


def main():
    model = MoE().eval()
    x = torch.randn(4, 64, 256)
    with torch.no_grad():
        baseline = model(x)

    fp32_bytes = state_dict_bytes(model)
    print(f"baseline  | expert weight: {type(model.experts[0].up.weight).__name__} "
          f"{model.experts[0].up.weight.dtype} | state_dict: {fp32_bytes / 1e6:.2f} MB")

    qmodel = copy.deepcopy(model)
    quantize_(
        qmodel,
        Int8WeightOnlyConfig(),
        # quantize expert linears only; keep the router in high precision,
        # mirroring quantize_llama_4.py which targets only routed experts
        filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear) and "experts" in fqn,
    )

    int8_bytes = state_dict_bytes(qmodel)
    expert_w = qmodel.experts[0].up.weight
    router_w = qmodel.router.weight
    with torch.no_grad():
        quantized_out = qmodel(x)

    print(f"int8 wo   | expert weight: {type(expert_w).__name__} "
          f"(qdata dtype: {expert_w.qdata.dtype}) | router untouched: "
          f"{type(router_w).__name__} {router_w.dtype} | "
          f"state_dict: {int8_bytes / 1e6:.2f} MB ({fp32_bytes / int8_bytes:.2f}x smaller)")
    print(f"int8 wo   | forward pass OK, output shape {tuple(quantized_out.shape)}, "
          f"SQNR vs fp32 baseline: {sqnr_db(baseline, quantized_out):.1f} dB")

    checks = {
        "expert weight is quantized subclass": expert_w.__class__ is not torch.nn.Parameter,
        "expert qdata is int8": expert_w.qdata.dtype == torch.int8,
        "router weight not quantized": isinstance(router_w, torch.nn.Parameter),
        "state_dict >= 3x smaller": fp32_bytes / int8_bytes >= 3.0,
        "SQNR > 30 dB": sqnr_db(baseline, quantized_out) > 30.0,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # Hardware-constraint probe for the example's docs: int4 path on CPU.
    int4_model = copy.deepcopy(model)
    try:
        quantize_(
            int4_model,
            Int4WeightOnlyConfig(),
            filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear) and "experts" in fqn,
        )
        with torch.no_grad():
            int4_model(x)
        print("int4 wo   | quantize + forward succeeded on CPU")
    except Exception as exc:
        print(f"int4 wo   | NOT CPU-runnable as expected: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:120]}")

    if not all(checks.values()):
        raise SystemExit(1)
    print("RESULT: quantize_() weight-only works on an MoE model — issue #729 example is feasible.")


if __name__ == "__main__":
    main()
