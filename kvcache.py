"""
Ternary KV cache (paper section 2.3).

  "The Context Window (KV Cache) memory footprint is identically compressed
   using the ternary scheme, resolving the VRAM exhaustion typically observed
   during the analysis of massive contextual inputs."

The repository contains no KV-cache code, so this fills the gap: perplexity of
Qwen3-1.7B-Base on WikiText-2 with the KV cache quantized to ternary states,
against an fp16 baseline and against the INT8/INT4 schemes that are actually
used in practice.

Method: one forward pass per 1024-token window, with the cache quantizing K and
V on their way into attention. That is exactly the state in which the keys and
values are held in memory during decoding.
"""
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from xt_quant import XTQuantizer  # noqa: E402  the author's code, unmodified

MODEL = os.environ.get("XT_MODEL", "Qwen/Qwen3-1.7B-Base")
WIN, NWIN = 1024, 32
DEV = "cuda"
xtq = XTQuantizer()


# --------------------------------------------------------------- quantizers
def q_none(t):
    return t


def q_int(bits):
    """per-token, per-head symmetric integer quantization"""
    lim = 2 ** (bits - 1) - 1

    def f(t):
        s = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / lim
        return torch.round(t / s).clamp(-lim, lim) * s
    return f


def q_group(bits, G=32):
    """symmetric quantization in groups of G along head_dim -- what real
    KV-cache methods do (KIVI, KVQuant)"""
    lim = 2 ** (bits - 1) - 1

    def f(t):
        *lead, d = t.shape
        x = t.float().reshape(-1, d // G, G)
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / lim
        return (torch.round(x / s).clamp(-lim, lim) * s).reshape(*lead, d).to(t.dtype)
    return f


def q_group_perchannel(bits, G=32):
    """groups along the token axis, i.e. per channel -- what post-RoPE keys need"""
    lim = 2 ** (bits - 1) - 1

    def f(t):
        b, h, n, d = t.shape
        if n % G:
            return q_group(bits, G)(t)
        x = t.float().permute(0, 1, 3, 2).reshape(-1, n // G, G)
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / lim
        q = torch.round(x / s).clamp(-lim, lim) * s
        return q.reshape(b, h, d, n).permute(0, 1, 3, 2).to(t.dtype)
    return f


def q_ternary_pertoken(t):
    """BitNet absmean with a scale per token and head (the most favourable case)"""
    g = t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.round(torch.clamp(t / g, -1, 1)) * g


def q_ternary_pertensor(t):
    """BitNet absmean with one scale for the whole tensor -- what xt_quant.py does"""
    g = t.abs().mean().clamp_min(1e-8)
    return torch.round(torch.clamp(t / g, -1, 1)) * g


def q_ternary_group(G=32):
    def f(t):
        *lead, d = t.shape
        x = t.float().reshape(-1, d // G, G)
        g = x.abs().mean(-1, keepdim=True).clamp_min(1e-8)
        q = torch.round(torch.clamp(x / g, -1, 1)) * g
        return q.reshape(*lead, d).to(t.dtype)
    return f


def q_xternary(t):
    """XTQuantizer.quantize() with the returned gamma applied (dequantized)"""
    q, g = xtq.quantize(t.float().contiguous())
    return (q * g).to(t.dtype)


def q_xternary_raw(t):
    """XTQuantizer.quantize() exactly as returned, without multiplying by gamma
    -- the way the demo in xt_quant.py:54 uses it"""
    q, _ = xtq.quantize(t.float().contiguous())
    return q.to(t.dtype)


def q_xternary_pertoken(t):
    """his 2:4 mask with a scale per token and head -- the best case for his scheme"""
    g = t.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.round(torch.clamp(t / g, -1, 1))
    flat = q.reshape(-1, 4)
    _, idx = torch.topk(flat.abs(), k=2, largest=False, dim=1)
    mask = torch.ones_like(flat)
    mask.scatter_(1, idx, 0)
    return (flat * mask).view_as(q) * g


def q_xternary_fixed(t):
    """the fix from issue #1 of the upstream repository: 2:4 mask on the
    full-precision values first, then ternary rounding on the survivors, with
    gamma computed from the survivors only"""
    flat = t.float().reshape(-1, 4)
    _, idx = torch.topk(flat.abs(), k=2, largest=False, dim=1)
    mask = torch.ones_like(flat)
    mask.scatter_(1, idx, 0)
    sp = flat * mask
    g = sp[sp != 0].abs().mean().clamp_min(1e-8)
    return (torch.round(torch.clamp(sp / g, -1, 1)) * g).view_as(t).to(t.dtype)


class QuantCache(DynamicCache):
    """quantizes both K and V with the same function"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return super().update(self.fn(key_states), self.fn(value_states),
                              layer_idx, cache_kwargs)


class SplitCache(DynamicCache):
    """a separate function for K and for V"""

    def __init__(self, fk, fv):
        super().__init__()
        self.fk, self.fv = fk, fv

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return super().update(self.fk(key_states), self.fv(value_states),
                              layer_idx, cache_kwargs)


# name, bits per stored value, factory returning a fresh cache
VARIANTS = [
    ("fp16 baseline (no quantization)", 16.00, lambda: QuantCache(q_none)),
    ("INT8 per-token", 8.00, lambda: QuantCache(q_int(8))),
    ("INT4 per-token", 4.00, lambda: QuantCache(q_int(4))),
    ("INT4 in groups of 32", 4.13, lambda: QuantCache(q_group(4))),
    ("INT4 K per-channel + V per-token (KIVI)", 4.13,
     lambda: SplitCache(q_group_perchannel(4), q_group(4))),
    ("INT3 in groups of 32", 3.13, lambda: QuantCache(q_group(3))),
    ("INT2 in groups of 32 (honest 2 bits)", 2.13, lambda: QuantCache(q_group(2))),
    ("ternary in groups of 32", 1.71, lambda: QuantCache(q_ternary_group())),
    ("ternary per-token (BitNet, best case)", 1.58, lambda: QuantCache(q_ternary_pertoken)),
    ("ternary per-tensor (as in xt_quant.py)", 1.58, lambda: QuantCache(q_ternary_pertensor)),
    ("X-Ternary verbatim from the repository", 2.00, lambda: QuantCache(q_xternary)),
    ("X-Ternary per-token + 2:4", 2.00, lambda: QuantCache(q_xternary_pertoken)),
    ("X-Ternary verbatim, gamma never applied", 2.00, lambda: QuantCache(q_xternary_raw)),
    ("X-Ternary with the issue #1 fix", 2.00, lambda: QuantCache(q_xternary_fixed)),
    ("ternary K only, V left in fp16", 1.58,
     lambda: SplitCache(q_ternary_pertoken, q_none)),
    ("ternary V only, K left in fp16", 1.58,
     lambda: SplitCache(q_none, q_ternary_pertoken)),
]


def wikitext2_test():
    """WikiText-2 test split as one string, cached under data/"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "wikitext2_test.txt")
    if not os.path.exists(path):
        from datasets import load_dataset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        with open(path, "w") as f:
            f.write("\n\n".join(ds["text"]))
    return open(path).read()


def perplexity(model, windows, make_cache):
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for w in windows:
            logits = model(w, past_key_values=make_cache(), use_cache=True).logits
            lg, tg = logits[:, :-1].float(), w[:, 1:]
            nll += torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.size(-1)), tg.reshape(-1), reduction="sum").item()
            ntok += tg.numel()
    return math.exp(nll / ntok)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    ids = tok(wikitext2_test(), return_tensors="pt").input_ids[0]
    windows = [ids[i * WIN:(i + 1) * WIN].unsqueeze(0).to(DEV) for i in range(NWIN)]
    print(f"model {MODEL}  |  {NWIN} windows of {WIN} tokens  |  "
          f"vocabulary {len(tok)}")
    print()
    print(f"  {'variant':44s} {'bit/value':>9s} {'PPL':>10s} {'x baseline':>11s}")
    base = None
    for name, bits, make in VARIANTS:
        t0 = time.perf_counter()
        ppl = perplexity(model, windows, make)
        base = base or ppl
        print(f"  {name:44s} {bits:9.2f} {ppl:10.2f} {ppl/base:10.1f}x"
              f"   [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
