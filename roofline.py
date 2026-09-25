"""
Memory-bandwidth ceiling for the paper's headline claim.

Abstract and conclusion of the paper:
  "reduces memory constraints by approximately 80%, enabling 70B-parameter
   class models to run on single consumer-grade GPUs"
  "(e.g. 24GB VRAM) ... delivers approximately 60 tokens per second"

Single-stream autoregressive decoding is memory-bound: every generated token
requires reading the whole weight set once. The ceiling is therefore set by
memory bandwidth, not by the quantization scheme.
"""
import time

import torch

DEV = "cuda"
# Catalogue bandwidth of the GPU under test; override for a different card.
SPEC_BW = {"NVIDIA GeForce RTX 3090": 936.2}

name = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__}  |  {name}  |  "
      f"VRAM {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
print()

print("=" * 78)
print("1) Measured read bandwidth")
print("=" * 78)
x = torch.empty(1 << 30, dtype=torch.float16, device=DEV).normal_()
torch.cuda.synchronize()
best = 0.0
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(4):
        x.sum()
    torch.cuda.synchronize()
    best = max(best, x.numel() * 2 / ((time.perf_counter() - t0) / 4) / 1e9)
print(f"  read {x.numel()*2/2**30:.1f} GiB per pass, best {best:.0f} GB/s")
del x
torch.cuda.empty_cache()
spec = SPEC_BW.get(name)
if spec:
    print(f"  catalogue bandwidth {spec:.0f} GB/s  =>  efficiency {best/spec:.0%}")
else:
    spec = best
    print("  catalogue bandwidth unknown for this device; using the measured value")

print()
print("=" * 78)
print("2) Token-rate ceiling for a 70B model on one card")
print("=" * 78)
P = 70e9
# bits per weight; for 2:4 the cost per block of 4 is survivors + index metadata
variants = [
    ("FP16", 16.0),
    ("BitNet b1.58 (5 weights / 8 bits)", 1.6),
    ("X-Ternary 2 bit/weight per README", 2.0),
    ("X-Ternary + FP8 scale per 32 (sec. 2.2)", 2.25),
    ("2:4 compressed: 2x2b value + 4b index", (2 * 2 + 4) / 4),
    ("2:4 compressed, best case: 2x1b + 4b", (2 * 1 + 4) / 4),
]
print(f"  {'format':40s} {'GiB':>7s} {'ceil @spec':>11s} {'ceil @measured':>15s}")
for label, bits in variants:
    gb = P * bits / 8 / 1e9
    print(f"  {label:40s} {gb*1e9/2**30:7.1f} {spec/gb:9.1f}/s {best/gb:13.1f}/s")
print()
print("  The ceiling assumes 100% bus utilisation, an empty KV cache, no")
print("  activation traffic and no dequantization overhead. Real engines reach")
print("  60-80% of it.")
print(f"  The paper claims 60 tok/s. The format it specifies (2.25 bit/weight)")
print(f"  has a ceiling of {spec/(P*2.25/8/1e9):.1f}/s.")
print()
print("  Note: 2:4 compression saves no memory. Two surviving values at 2 bits")
print("  plus 4 bits of index is the same 8 bits per 4 weights as dense 2-bit.")
print("  The two ideas do not compose in the memory budget; 2:4 buys compute,")
print("  not bandwidth.")
