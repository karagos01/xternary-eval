# X-Ternary — independent evaluation

Reproducible evaluation of three claims in M. Mazgal, *X-Ternary: A Deterministic
Causal 2-Bit Inference Engine for Large Language Models* (Zenodo, DOI
[10.5281/zenodo.22877345](https://doi.org/10.5281/zenodo.22877345)) that the
accompanying repository [`michalmazgal/X-Ternary`](https://github.com/michalmazgal/X-Ternary)
does not implement: the bit accounting, the 70B-at-60-tok/s figure, and the
ternary KV cache.

The write-up is in [`PAPER.md`](PAPER.md), in Czech in [`PAPER.cs.md`](PAPER.cs.md).
In short: once 2:4 sparsity is enforced the scheme carries 1.26 bit/weight and
stores it in 2.25, so it holds 20 % less information than BitNet b1.58 in 42 %
more bits; the claimed 60 tok/s is above the memory-bandwidth ceiling of the card
named in the paper; floating-point addition on the GPU is already non-associative
for 31.9 % of random triples, which is the cause of the non-determinism the paper
proposes to cure by adding non-associativity; and a ternary KV cache costs a
factor of 5 001 in perplexity where a correct 4-bit cache costs 1.2 %.

## Requirements

```
python3 -m pip install -r requirements.txt   # numpy, torch, transformers, datasets
```

One CUDA GPU is required for everything except `bits.py`. The measurements in the
paper were taken on an RTX 3090; `roofline.py` knows that card's catalogue
bandwidth and falls back to the measured value on any other.

## Reproducing the results

```
./run_all.sh          # everything, about 5 minutes on one RTX 3090
```

`XT_MODEL` selects the model (default `Qwen/Qwen3-1.7B-Base`, downloaded from
Hugging Face on first use). WikiText-2 is fetched once into `data/`.

| Script | What it computes | Time |
|---|---|---|
| `bits.py` | reachable code space, entropy of the emitted tensors, bit accounting against BitNet | 1 s |
| `roofline.py` | measured read bandwidth, token-rate ceiling for a 70B model | 10 s |
| `determinism.py` | associativity of fp addition, dependence of GEMM and of real logits on batch size | 30 s |
| `kvcache.py` | perplexity with the KV cache in 16 different formats | 70 s |
| `attribution.py` | loss in bits, split between rounding and the 2:4 mask, what the mask removes | 40 s |

`vendor/xt_quant.py` is the author's quantizer copied verbatim at commit
`9f27dfa` (MIT, see `vendor/LICENSE.xt_quant`); `kvcache.py` imports it
unmodified, so the X-Ternary rows measure his code and not a reinterpretation of
it. The full output of a run is in [`results.log`](results.log).

## Key numbers

| KV cache format | bit/value | PPL | × baseline |
|---|---|---|---|
| fp16 baseline | 16.00 | **10.40** | 1.0× |
| INT4, K per-channel + V per-token | 4.13 | 10.52 | 1.0× |
| INT3 in groups of 32 | 3.13 | 273.5 | 26× |
| ternary per-token, best case | 1.58 | 6 287 | 605× |
| **X-Ternary verbatim from the repository** | 2.00 | **52 004** | **5 001×** |
| X-Ternary, gamma never applied | 2.00 | 722 019 | 69 434× |

Qwen3-1.7B-Base, 32 windows of 1024 tokens from the WikiText-2 test split. Of the
12.29 bit/token lost, 9.24 (75 %) is the ternary rounding and 3.05 (25 %) is the
2:4 mask, which selects the positions to prune after rounding — when every
non-zero has the same magnitude — and therefore zeroes position 0 of each block of
four in 81 % of blocks and position 1 in 76 %, against 27 % and 16 % for the other
two.

## Scope

Weight-level accuracy of the scheme was evaluated separately and earlier,
including one-shot PTQ, training from scratch with 2:4 at up to 5 billion tokens,
and a measurement of real 2:4 speedup on an RTX 3090 (1.6–2.0×). The ordering
defect found there was reported upstream as issue #1 on 27 April 2026 and is
still open. This repository covers only the claims added by the September 2026
preprint.

## Companion evaluations

- [`octonion-mppt-eval`](https://github.com/karagos01/octonion-mppt-eval) — the octonion two-layer/associator template and the magnon claims
- [`causal-trilogy-eval`](https://github.com/karagos01/causal-trilogy-eval) — the CQFT / PCTP / SOTP trilogy of September 2026

## Licence

Code (all `*.py` and `run_all.sh`): MIT, see `LICENSE`.
`vendor/xt_quant.py`: MIT, © 2026 michalmazgal, see `vendor/LICENSE.xt_quant`.
`PAPER.md` and `PAPER.cs.md`: CC BY 4.0.
