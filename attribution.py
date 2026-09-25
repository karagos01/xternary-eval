"""
Where the damage comes from, and what the perplexities mean in bits.

Two questions:
  1. How much of the loss is the ternary rounding itself (a cost any ternary KV
     cache pays) and how much is the 2:4 mask as published?
  2. The mask selects the two smallest values *after* ternary rounding, when
     every non-zero has magnitude gamma. Does it then still pick the least
     important weights?

The perplexities are taken from kvcache.py; rerun that script if the numbers
here need refreshing.
"""
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvcache import MODEL, wikitext2_test  # noqa: E402

DEV = "cuda"

# from kvcache.py
MEASURED = [
    ("fp16 baseline", 10.40),
    ("INT4 K per-channel + V per-token (KIVI)", 10.52),
    ("INT3 in groups of 32", 273.54),
    ("ternary per-token (best case)", 6287.38),
    ("X-Ternary verbatim from the repository", 52003.53),
    ("X-Ternary per-token + 2:4", 62128.76),
    ("X-Ternary, gamma never applied", 722019.26),
]


def bits_table(vocab):
    rand = math.log2(vocab)
    base = math.log2(MEASURED[0][1])
    print(f"vocabulary {vocab} tokens, random guessing = {rand:.2f} bit/token "
          f"= PPL {vocab}")
    print()
    print(f"  {'variant':44s} {'PPL':>10s} {'bit/token':>10s} {'model retained':>15s}")
    for name, ppl in MEASURED:
        b = math.log2(ppl)
        print(f"  {name:44s} {ppl:10.2f} {b:9.2f}  "
              f"{max((rand - b) / (rand - base), 0.0):14.1%}")
    print()
    tern = math.log2(MEASURED[3][1])
    xt = math.log2(MEASURED[4][1])
    total = xt - base
    print(f"  total loss {total:.2f} bit/token, of which")
    print(f"    ternary rounding alone: {tern-base:.2f} bit ({(tern-base)/total:.0%})")
    print(f"    the 2:4 mask on top:    {xt-tern:.2f} bit ({(xt-tern)/total:.0%})")


def mask_stats(t, per_token):
    """statistics of the published mask on a real K or V tensor"""
    g = (t.abs().mean(-1, keepdim=True) if per_token else t.abs().mean()).clamp_min(1e-8)
    q = torch.round(torch.clamp(t / g, -1, 1))
    flat, orig = q.reshape(-1, 4), t.reshape(-1, 4)

    _, idx = torch.topk(flat.abs(), k=2, largest=False, dim=1)   # as published
    mask = torch.ones_like(flat)
    mask.scatter_(1, idx, 0)

    _, idx_ok = torch.topk(orig.abs(), k=2, largest=False, dim=1)  # correct choice
    mask_ok = torch.ones_like(flat)
    mask_ok.scatter_(1, idx_ok, 0)

    energy = (orig ** 2).sum()
    return dict(
        nz_before=(flat != 0).float().mean().item(),
        nz_after=((flat != 0) & (mask == 1)).float().mean().item(),
        killed=((flat != 0) & (mask == 0)).float().mean().item(),
        e_killed=((orig ** 2 * (mask == 0)).sum() / energy).item(),
        e_killed_ok=((orig ** 2 * (mask_ok == 0)).sum() / energy).item(),
        positions=(mask == 0).float().mean(0).tolist(),
    )


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    bits_table(len(tok))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    ids = tok(wikitext2_test()[:20000], return_tensors="pt").input_ids[:, :1024].to(DEV)
    nlayer = model.config.num_hidden_layers
    want = (0, nlayer // 2, nlayer - 1)
    grabbed = []

    class Grab(DynamicCache):
        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx in want:
                grabbed.append((layer_idx, k.detach().float(), v.detach().float()))
            return super().update(k, v, layer_idx, cache_kwargs)

    with torch.no_grad():
        model(ids, past_key_values=Grab(), use_cache=True)

    print()
    print(f"  {'tensor / scale':30s} {'nz before':>9s} {'nz after':>9s} "
          f"{'nz killed':>10s} {'energy killed':>14s} {'if chosen correctly':>20s}")
    for layer, k, v in grabbed:
        for nm, t in (("K", k), ("V", v)):
            for per_token, lab in ((False, "per-tensor (repo)"), (True, "per-token")):
                s = mask_stats(t, per_token)
                print(f"  L{layer:02d} {nm} {lab:22s} {s['nz_before']:8.1%} "
                      f"{s['nz_after']:8.1%} {s['killed']:9.1%} {s['e_killed']:13.1%} "
                      f"{s['e_killed_ok']:19.1%}")
    print()
    pos = mask_stats(grabbed[0][1], True)["positions"]
    print("  how often the mask zeroes each position of a block of 4: "
          f"{[f'{p:.0%}' for p in pos]}")
    print("  (a magnitude-based rule would show no positional preference)")


if __name__ == "__main__":
    main()
