# Bit accounting, bandwidth and a ternary KV cache: an independent evaluation of X-Ternary

**Author:** karagos01 · **September 2026**
**Licence:** this text CC BY 4.0 · the accompanying code MIT

## Abstract

X-Ternary (Mazgal, Zenodo, DOI [10.5281/zenodo.22877345](https://doi.org/10.5281/zenodo.22877345))
proposes a 2-bit inference engine for large language models: ternary weights
{−1, 0, +1} combined with NVIDIA 2:4 structured sparsity, block-level FP8
scaling, a ternary KV cache, and an octonion "hidden causal memory" bridged into
the tensor layer by the octonion associator. The accompanying repository contains
67 lines of Python implementing the first two of those. This evaluation supplies
the missing measurements for three claims that the repository does not touch.

First, the bit accounting is inverted. Once 2:4 sparsity is enforced, only 33 of
the 81 ternary blocks of four remain reachable, so the scheme carries 1.26
bit/weight of information — 20.4 % less than BitNet b1.58 — while storing it in
2.25 bit/weight, 42 % more. Second, the claim of 70B parameters at 60 tokens per
second on a 24 GB card exceeds the memory-bandwidth ceiling of the card: the
format the paper specifies tops out at 47.6 tok/s at 100 % bus utilisation, and
BitNet b1.58, which the paper rejects, reaches 66.9. Third, floating-point
addition on the GPU is already non-associative for 31.9 % of random triples, and
that is the cause of the run-to-run variation the paper proposes to cure by
adding non-associativity. Finally, a ternary KV cache is measured directly:
perplexity of Qwen3-1.7B-Base on WikiText-2 rises from 10.40 to 52 004 with the
published code, a factor of 5 001, where a properly grouped 4-bit cache costs
1.2 %. All scripts are in this repository.

## What is claimed and what is implemented

| Claim in the paper | In the repository |
| --- | --- |
| Ternary weights {−1, 0, +1}, absmean scaling | yes, `xt_quant.py:22` |
| 2:4 structured sparsity as a fourth state `x` | yes, `xt_quant.py:27-37` |
| Exactly 2 bits per weight, "zero packing overhead" | no packing code; a dense float tensor is returned |
| Block-level FP8 scaling over groups of 32 | absent |
| Ternary KV cache (section 2.3) | absent |
| Octonion hidden causal memory, associator bridge (3.1) | absent |
| TMA and `cuda::barrier` enforcing causal order (3.1) | absent |
| Routing to CUDA ALUs on a trigger token (3.2) | absent |
| STE + EWC two-phase training (4) | absent |
| 70B on 24 GB at ~60 tok/s, 80 % memory reduction | absent |

Weight-level accuracy of the scheme was evaluated earlier by the present author,
including one-shot PTQ on GPT-2, training from scratch with 2:4 at 2.5 and 5
billion tokens, and a measurement of real 2:4 speedup on an RTX 3090
(1.6–2.0×). Those results are prior work and are not repeated here; the ordering
defect they identified was reported upstream as issue #1 on 27 April 2026 and has
not been answered. This evaluation covers only claims that post-date that work.

## Method

`bits.py` enumerates the reachable code space and measures the empirical entropy
of the tensors the published quantizer emits. `roofline.py` measures read
bandwidth on the GPU under test and converts it into a token-rate ceiling.
`determinism.py` measures associativity of floating-point addition and the
dependence of GEMM output, and of a real model's logits, on batch size.
`kvcache.py` evaluates perplexity of Qwen3-1.7B-Base on 32 windows of 1024
tokens from the WikiText-2 test split, with K and V quantized on their way into
attention by a `DynamicCache` subclass; the X-Ternary variants call
`XTQuantizer.quantize` from the upstream repository unmodified (vendored at
commit `9f27dfa` under `vendor/`). `attribution.py` splits the measured loss
between the ternary rounding and the 2:4 mask, and measures which values the mask
removes. Hardware: one RTX 3090, PyTorch 2.5.1, transformers 4.51.3.

## Results

### Bit accounting

A block of four ternary weights has 3⁴ = 81 states. The 2:4 constraint admits at
most two non-zeros, leaving 1 + 8 + 24 = 33 blocks, hence log₂(33)/4 = 1.261
bit/weight.

| scheme | needs | stores | waste |
| --- | --- | --- | --- |
| BitNet b1.58, dense ternary | 1.585 b | 1.585 b | 0 % |
| X-Ternary, 2 bit/weight per README | 1.261 b | 2.000 b | +58.6 % |
| X-Ternary + FP8 scale per 32, paper §2.2 | 1.261 b | 2.250 b | +78.4 % |

On real weight distributions the emitted tensors are further from uniform:
empirical entropy is 1.089 bit/weight for gaussian and 1.170 for laplacian
weights, so the stored 2 bits carry 84 % and 71 % of overhead respectively. The
premise of the paper — that BitNet's log₂3 ≈ 1.58 bits wastes the 2-bit space —
is therefore reversed: X-Ternary carries 20.4 % less information than BitNet and
spends 42 % more bits on it.

The redundancy is explicit in the design. The README lists four states, `01`,
`10`, `00`, `11`, of which `00` ("classic zero") and `11` ("structural zero,
high-Z, physically skip the operation") both contribute zero to `y = Wx`. A
four-symbol alphabet therefore carries three values, and 2 − log₂3 = 0.415
bit/weight is unused by construction. `theory.md` presents this as exploiting
"the information redundancy of the 2-bit space".

### Bandwidth ceiling

Single-stream decoding reads the whole weight set once per token, so the ceiling
is bandwidth, not arithmetic. Measured read bandwidth on the RTX 3090 was
885 GB/s, 94 % of the catalogue 936 GB/s. For 70 · 10⁹ parameters:

| format | GiB | ceiling @spec | ceiling @measured |
| --- | --- | --- | --- |
| FP16 | 130.4 | 6.7/s | 6.3/s |
| BitNet b1.58 | 13.0 | **66.9/s** | 63.2/s |
| X-Ternary, 2 bit/weight | 16.3 | 53.5/s | 50.5/s |
| X-Ternary + FP8 scale per 32 | 18.3 | **47.6/s** | 44.9/s |
| 2:4 compressed, 2 × 2 b value + 4 b index | 16.3 | 53.5/s | 50.5/s |

The ceiling assumes 100 % bus utilisation, an empty KV cache, no activation
traffic and no dequantization overhead; real engines reach 60–80 % of it. The
format the paper specifies therefore cannot reach the claimed 60 tok/s, and the
scheme the paper rejects is faster on the paper's own headline metric.

Two further points. The 2:4 compression saves no memory: two surviving values at
two bits each plus four bits of index metadata is the same eight bits per four
weights as a dense 2-bit layout, so the two ideas do not compose in the memory
budget. And 2:4 buys arithmetic throughput rather than bandwidth, which is the
opposite of what single-stream decoding is limited by. Sparse Tensor Cores
moreover accept the 2:4 pattern in fp16, bf16, int8 or tf32 on Ampere and fp8 on
Hopper; there is no 2-bit sparse path, so the weights must be expanded to a
supported type before the sparse GEMM, and the claim of "immediately doubling the
effective throughput without precision degradation" holds for neither half.

### Determinism

The paper attributes hallucination to "associative probability" in GEMM and
proposes to enforce non-associativity. Floating-point addition is not associative
to begin with:

```
fp16: (a+b)+c != a+(b+c) for 31.92 % of random triples
bf16: 31.88 %      fp32: 31.86 %
```

Consequently the result depends on the reduction schedule. With the same input
vector, `x @ W.T` at batch 64 is not bit-identical to the same row computed
alone, and on Qwen3-1.7B the same prompt at batch 8 reorders four of the ten
highest-scoring tokens. Non-associativity is already present and is the
mechanism behind the very non-reproducibility the paper sets out to remove;
determinism follows from fixing the reduction schedule, not from adding more
non-associativity. Hallucination is a separate matter: a bit-deterministic model
can be confidently wrong.

### Ternary KV cache

Section 2.3 states that the KV cache is "identically compressed using the ternary
scheme". Perplexity of Qwen3-1.7B-Base on WikiText-2, 32 windows of 1024 tokens:

| variant | bit/value | PPL | × baseline |
| --- | --- | --- | --- |
| fp16 baseline | 16.00 | **10.40** | 1.0× |
| INT8 per-token | 8.00 | 10.46 | 1.0× |
| INT4, K per-channel + V per-token | 4.13 | **10.52** | 1.0× |
| INT4 in groups of 32 | 4.13 | 27.35 | 2.6× |
| INT3 in groups of 32 | 3.13 | 273.5 | 26.3× |
| INT2 in groups of 32 | 2.13 | 7 342 | 706× |
| ternary in groups of 32 | 1.71 | 7 326 | 705× |
| ternary per-token, best case | 1.58 | 6 287 | 605× |
| ternary per-tensor, as in `xt_quant.py` | 1.58 | 13 753 | 1 323× |
| **X-Ternary verbatim from the repository** | 2.00 | **52 004** | **5 001×** |
| X-Ternary with per-token scaling | 2.00 | 62 129 | 5 975× |
| X-Ternary, gamma never applied | 2.00 | 722 019 | 69 434× |
| X-Ternary with the issue #1 ordering fix | 2.00 | 11 735 | 1 129× |

A correctly implemented 4-bit cache is free, three bits cost a factor of 26, and
the cliff falls between four and three bits. Ternary is two orders of magnitude
past it, and the published scheme is another factor of eight worse than plain
ternary. The ordering fix helps by 4.4× and is still unusable.

Expressed in bits, with a vocabulary of 151 669 tokens where uniform guessing is
17.21 bit/token, the baseline model predicts at 3.38 bit/token, so it holds 13.83
bits of predictive power. With the published cache it holds 1.54 bits, 11.2 % of
the original. With gamma left unapplied it predicts at 19.46 bit/token, worse
than drawing tokens uniformly at random.

Of the 12.29 bit/token lost, 9.24 (75 %) is the ternary rounding — a cost any
ternary KV cache pays — and 3.05 (25 %) is the 2:4 mask.

### Where the mask does its damage

The mask selects `topk(|q|, k=2, largest=False)` after ternary rounding, when
every non-zero has magnitude γ. Ties are then broken by index, so the choice is
positional rather than by importance. The mask zeroes position 0 of each block in
81 % of blocks and position 1 in 76 %, against 27 % and 16 % for positions 2 and
3; a magnitude-based rule shows no positional preference. The cost, on the K and
V tensors of Qwen3-1.7B:

| tensor | non-zeros killed | energy killed | energy if chosen correctly |
| --- | --- | --- | --- |
| K, layer 0 | 3.9 % | 0.3 % | 0.1 % |
| V, layer 0 | 17.8 % | 31.6 % | 9.3 % |
| K, layer 14 | 14.0 % | 16.0 % | 4.8 % |
| K, layer 27 | 12.8 % | **26.9 %** | **2.2 %** |
| V, layer 27 | 20.0 % | 34.6 % | 12.4 % |

Selecting by position destroys 12 times more energy than necessary in the keys of
the last layer and 3.4 times more in the values. The defect also grows as the
rest of the scheme improves: per-token scaling, normally strictly better, leaves
more survivors of the rounding step (68.5 % against 66.3 % for V in layer 27) and
therefore gives the positional mask more to destroy, which is why it scores worse
overall (62 129 against 52 004).

## Two defects in the published code

`quantize()` returns `(xt_weight, gamma)` with the values in {−1, 0, +1}; the
caller must multiply by γ to recover the original scale. The demonstration in
`xt_quant.py:54` does not, and neither does `estimate_memory_saving`. Used as
shown, the layer output is mis-scaled by roughly 1/γ; that is the 69 434× row.

`estimate_memory_saving` returns `original_size * 2/16` unconditionally. It models
neither the FP8 block scales of section 2.2 nor the 2:4 index metadata, and it is
independent of the tensor it is called on.

## Limitations

The perplexity measurements use one 1.7B model, one dataset and one window
length; the ranking of variants is robust across layers, but absolute values are
not a benchmark of KV-cache methods. The 4-bit reference here is a
straightforward per-channel/per-token grouping rather than a faithful
reimplementation of KIVI or KVQuant, and it is already loss-free at this model
size, so a stronger reference could only widen the gap. The bandwidth ceiling is
a roofline and not an end-to-end throughput measurement; it bounds the claim from
above rather than reproducing it. The octonion layer, the FP8 block scaling, the
CUDA-ALU routing and the two-phase training pipeline are not evaluated at all,
because the paper does not define them to a level that can be implemented — in
particular, no mapping of physical or model quantities onto the octonion
components e₀…e₇ is given anywhere, which is the same gap found in the author's
MPPT proposal.

## Conclusion

Three claims that the repository does not implement were tested and each fails
before reaching any question of implementation quality. The bit accounting is
inverted: the scheme stores less information in more bits than the scheme it
criticises. The headline throughput figure is above the memory-bandwidth ceiling
of the card it names. The determinism argument mistakes the cause for the cure,
since floating-point non-associativity is already present and is what makes the
output depend on the batch size. The one claim that was a genuine open question,
the ternary KV cache, was measured and costs a factor of 5 001 in perplexity
where a correct 4-bit cache costs 1.2 %.

Three quarters of that loss is intrinsic to ternary keys and values and would
befall anyone. The remaining quarter comes from selecting the pruned positions
after rounding, when no ordering information is left — the defect reported
upstream in April 2026, still open, and still the smaller half of the problem.

## Data availability

All scripts, the vendored quantizer and the full output log are at
https://github.com/karagos01/xternary-eval. `./run_all.sh` reproduces every
number in this text in about five minutes on one RTX 3090.

## References

1. M. Mazgal. *X-Ternary: A Deterministic Causal 2-Bit Inference Engine for Large Language Models.* Zenodo, 2026. DOI 10.5281/zenodo.22877345.
2. S. Ma et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764, 2024.
3. A. Mishra et al. *Accelerating Sparse Deep Neural Networks.* arXiv:2104.08378, 2021.
4. Z. Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024, PMLR 235:32332–32344. arXiv:2402.02750.
5. C. Hooper et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* arXiv:2401.18079, 2024.
6. M. Sun, Z. Liu, A. Bair, J. Z. Kolter. *A Simple and Effective Pruning Approach for Large Language Models.* arXiv:2306.11695, 2023.
7. E. Frantar, D. Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML 2023. arXiv:2301.00774.
8. D. Goldberg. *What Every Computer Scientist Should Know About Floating-Point Arithmetic.* ACM Computing Surveys 23(1):5–48, 1991.
9. S. Merity, C. Xiong, J. Bradbury, R. Socher. *Pointer Sentinel Mixture Models.* arXiv:1609.07843, 2016.
10. Qwen Team. *Qwen3-1.7B-Base.* https://huggingface.co/Qwen/Qwen3-1.7B-Base
11. Microsoft. *bitnet-b1.58-2B-4T.* https://huggingface.co/microsoft/bitnet-b1.58-2B-4T — a trained ternary-weight model whose quantization configuration covers linear layers only; its KV cache is kept at 16 bits.
