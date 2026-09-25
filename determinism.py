"""
Is GEMM on a GPU associative?

The paper, sections 1 and 3.1:
  "the foundational reliance on Generalized Matrix Multiplication (GEMM)
   introduces a critical flaw: associative probability"
  "Tensor Cores are optimized for associative mathematical operations, which
   inherently destroys causal sequence enforcement"
  -> the proposal is to impose non-associativity and thereby gain determinism.

Floating-point addition is not associative to begin with, and that is precisely
what makes a transformer's output depend on the reduction schedule. The property
the paper wants to add is already present, and it is the cause of the
non-reproducibility it claims to cure.
"""
import os

import torch

DEV = "cuda"
MODEL = os.environ.get("XT_MODEL", "Qwen/Qwen3-1.7B-Base")

print("=" * 78)
print("1) Associativity of addition on the GPU")
print("=" * 78)
g = torch.Generator(device=DEV).manual_seed(0)
N = 2_000_000
for dt, nm in [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
    a, b, c = (torch.randn(N, generator=g, device=DEV, dtype=dt) for _ in range(3))
    frac = (((a + b) + c) != (a + (b + c))).float().mean().item()
    print(f"  {nm}: (a+b)+c != a+(b+c) for {frac:6.2%} of random triples")
print("  => non-associativity is already in the hardware. It is not a property")
print("     that can be added.")

print()
print("=" * 78)
print("2) Does a GEMM result depend on the reduction schedule (i.e. on batch)?")
print("=" * 78)
torch.backends.cuda.matmul.allow_tf32 = True
W = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
x = torch.randn(1, 4096, device=DEV, dtype=torch.bfloat16)
y_alone = (x @ W.T).float()
for B in (2, 8, 64, 256):
    X = torch.randn(B, 4096, device=DEV, dtype=torch.bfloat16)
    X[0] = x[0]
    y_batch = (X @ W.T).float()[:1]
    md = (y_alone - y_batch).abs().max().item()
    print(f"  batch {B:4d}: bit-identical = {str(torch.equal(y_alone, y_batch)):5s}"
          f"   max abs diff {md:.3e}  ({md/y_alone.abs().max().item():.2e} rel.)")
print("  Same input vector, different batch size => different numbers.")

print()
print("=" * 78)
print("3) The same effect on a real model's logits")
print("=" * 78)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
ids = tok("The capital of France is Paris, and the capital of Germany is",
          return_tensors="pt").input_ids.to(DEV)
with torch.no_grad():
    lo1 = model(ids).logits[0, -1].float()
    for B in (2, 8, 32):
        loB = model(ids.repeat(B, 1)).logits[0, -1].float()
        flips = int((lo1.argsort(descending=True)[:10]
                     != loB.argsort(descending=True)[:10]).sum())
        print(f"  batch {B:3d}: bit-identical = {str(torch.equal(lo1, loB)):5s}  "
              f"max logit diff {(lo1-loB).abs().max().item():.4f}  "
              f"positions reordered in the top 10: {flips}")
print("  Same prompt, same model, same GPU: the logits depend on the batch size.")
print("  The cause is the reduction order inside the GEMM, not 'associative")
print("  probability'. Determinism is obtained by fixing the schedule, not by")
print("  adding more non-associativity.")
