"""
Bit accounting: X-Ternary versus BitNet b1.58.

The paper (Zenodo 10.5281/zenodo.22877345, sections 2.1-2.2) and theory.md claim:
  - "Standard BitNet uses log2(3) bits. In an 8-bit byte that leaves unused
     space" -> X-Ternary uses a clean 2 bits, "Zero Packing Overhead"
  - "X-Ternary exploits not only compression but the information redundancy
     of the 2-bit space"
  - block-level FP8 scaling over groups of 32 -> "an average memory footprint
    of slightly over 2 bits per parameter"

This script counts how much information the scheme can actually carry.
"""
import collections
import itertools
import math

import numpy as np

print("=" * 78)
print("1) How many states remain reachable once 2:4 sparsity is enforced")
print("=" * 78)

# An X-Ternary block is 4 weights from {-1,0,1}, but 2:4 allows at most
# 2 non-zeros per block of 4.
all_blocks = list(itertools.product([-1, 0, 1], repeat=4))
valid = [b for b in all_blocks if sum(v != 0 for v in b) <= 2]
by_k = collections.Counter(sum(v != 0 for v in b) for b in valid)
print(f"  all ternary blocks of 4:                3^4 = {len(all_blocks)}")
for k in sorted(by_k):
    print(f"    with {k} non-zeros:                    {by_k[k]:3d}  (C(4,{k})*2^{k})")
print(f"  valid X-Ternary blocks (<=2 non-zero):  {len(valid)}")
print(f"  entropy under a uniform distribution:   log2({len(valid)})/4 = "
      f"{math.log2(len(valid))/4:.4f} bit/weight")

print()
print("=" * 78)
print("2) Information content versus what is actually stored")
print("=" * 78)
H_xt = math.log2(len(valid)) / 4
H_bitnet = math.log2(3)
rows = [
    ("BitNet b1.58 (dense ternary)", H_bitnet, H_bitnet, "5 weights per 8 bits"),
    ("X-Ternary: information content", H_xt, 2.0, "2 bit/weight per README"),
    ("X-Ternary + FP8 scale per 32", H_xt, 2.0 + 8 / 32, "paper section 2.2"),
]
print(f"  {'scheme':34s} {'needs':>9s} {'stores':>8s}  {'waste':>7s}   note")
for name, need, store, note in rows:
    print(f"  {name:34s} {need:8.3f}b {store:7.3f}b  {store/need-1:+6.1%}   {note}")
print()
print(f"  X-Ternary carries {(1 - H_xt/H_bitnet):.1%} LESS information than BitNet")
print(f"  and stores it in {(2.25/H_bitnet - 1):.1%} MORE bits (with the FP8 scale).")
print("  The paper's premise ('BitNet wastes space') is therefore inverted.")

print()
print("=" * 78)
print("3) The README's 'four states': how many are functionally distinct")
print("=" * 78)
print("  01 -> +1 | 10 -> -1 | 00 -> 0 | 11 -> x")
print("  theory.md: 'state x is defined as a Structural Zero ... corresponds to High-Z'")
print("  In y = Wx: '00 -> ignore', '11 -> physically skip the operation'.")
print("  Both contribute zero to y, so a 4-symbol code carries 3 values.")
print(f"  Code capacity: 2 bits. Used: log2(3) = {math.log2(3):.4f} bits.")
print(f"  Wasted by construction: {2 - math.log2(3):.4f} bit/weight = "
      f"{(2-math.log2(3))/2:.1%} of the 2-bit space.")

print()
print("=" * 78)
print("4) Empirical entropy of what the published code actually emits")
print("=" * 78)
rng = np.random.default_rng(0)


def xt_reference(w):
    """Verbatim port of xt_quant.py:18-38, without the torch dependency."""
    gamma = np.abs(w).mean()
    q = np.round(np.clip(w / (gamma + 1e-8), -1, 1))
    flat = q.reshape(-1, 4)
    # topk(abs, k=2, largest=False) returns the lowest indices on ties
    order = np.argsort(np.abs(flat), axis=1, kind="stable")
    mask = np.ones_like(flat)
    np.put_along_axis(mask, order[:, :2], 0, axis=1)
    return (flat * mask), gamma


for label, w in [
    ("gaussian weights", rng.normal(size=(512, 512))),
    ("laplacian weights", rng.laplace(size=(512, 512))),
]:
    blocks, gamma = xt_reference(w)
    cnt = collections.Counter(map(tuple, blocks.astype(int)))
    p = np.array(list(cnt.values()), dtype=float)
    p /= p.sum()
    H = -(p * np.log2(p)).sum() / 4
    nz = (blocks != 0).mean()
    print(f"  {label:20s} blocks used {len(cnt):3d}/{len(valid)}  "
          f"entropy {H:.4f} bit/weight  non-zero {nz:.1%}")
    print(f"  {'':20s} -> stored at 2.0 bit/weight, waste {2.0/H - 1:+.1%}")
