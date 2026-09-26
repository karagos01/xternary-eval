# X-Ternary — nezávislé vyhodnocení

Reprodukovatelné vyhodnocení tří tvrzení z práce M. Mazgal, *X-Ternary: A
Deterministic Causal 2-Bit Inference Engine for Large Language Models* (Zenodo,
DOI [10.5281/zenodo.22877345](https://doi.org/10.5281/zenodo.22877345)), která
přiložený repozitář [`michalmazgal/X-Ternary`](https://github.com/michalmazgal/X-Ternary)
neimplementuje: bitové účetnictví, číslo 70B při 60 tok/s a ternární KV cache.

Text je v [`PAPER.cs.md`](PAPER.cs.md), anglicky v [`PAPER.md`](PAPER.md). Krátce:
po vynucení 2:4 řídkosti nese schéma 1,26 bitu na váhu a ukládá to ve 2,25, takže
drží o 20 % méně informace než BitNet b1.58 ve 42 % více bitů; tvrzených 60 tok/s
je nad pamětovým stropem karty, kterou článek jmenuje; sčítání v pohyblivé řádové
čárce na GPU už asociativní není u 31,9 % náhodných trojic, a právě to je
příčinou nedeterminismu, který článek chce léčit přidáním nonasociativity; a
ternární KV cache stojí faktor 5 001 na perplexitě tam, kde správná 4bitová cache
stojí 1,2 %.

> Komentáře v kódu a výpisy skriptů jsou anglicky. Anglická verze textu je
> v [`README.md`](README.md) a [`PAPER.md`](PAPER.md).

## Co je potřeba

```
python3 -m pip install -r requirements.txt   # numpy, torch, transformers, datasets
```

Na všechno kromě `bits.py` je potřeba jedna CUDA karta. Měření v článku vznikla
na RTX 3090; `roofline.py` zná katalogovou propustnost té karty a na jiné použije
naměřenou hodnotu.

## Reprodukce výsledků

```
./run_all.sh          # všechno, asi 5 minut na jedné RTX 3090
```

Model se volí přes `XT_MODEL` (výchozí `Qwen/Qwen3-1.7B-Base`, při prvním
spuštění se stáhne z Hugging Face). WikiText-2 se jednou stáhne do `data/`.

| Skript | Co spočítá | Čas |
|---|---|---|
| `bits.py` | dosažitelný kódový prostor, entropie vyrobených tenzorů, bitové účetnictví proti BitNetu | 1 s |
| `roofline.py` | naměřená propustnost čtení, strop tokenů za sekundu pro 70B model | 10 s |
| `determinism.py` | asociativita fp sčítání, závislost GEMM a reálných logitů na velikosti batche | 30 s |
| `kvcache.py` | perplexita s KV cachí v 16 různých formátech | 70 s |
| `attribution.py` | ztráta v bitech, rozdělení mezi zaokrouhlení a 2:4 masku, co maska odstraňuje | 40 s |

`vendor/xt_quant.py` je autorův kvantizátor zkopírovaný beze změny z commitu
`9f27dfa` (MIT, viz `vendor/LICENSE.xt_quant`); `kvcache.py` ho importuje
nezměněný, takže řádky X-Ternary měří jeho kód, ne jeho reinterpretaci. Celý
výstup jednoho běhu je v [`results.log`](results.log).

## Klíčová čísla

| Formát KV cache | bit/hodnotu | PPL | × baseline |
|---|---|---|---|
| fp16 baseline | 16,00 | **10,40** | 1,0× |
| INT4, K per-channel + V per-token | 4,13 | 10,52 | 1,0× |
| INT3 po skupinách 32 | 3,13 | 273,5 | 26× |
| ternary per-token, nejlepší případ | 1,58 | 6 287 | 605× |
| **X-Ternary verbatim z repozitáře** | 2,00 | **52 004** | **5 001×** |
| X-Ternary bez aplikované gammy | 2,00 | 722 019 | 69 434× |

Qwen3-1.7B-Base, 32 okének po 1024 tokenech z testovací části WikiText-2.
Z 12,29 ztraceného bitu na token je 9,24 (75 %) ternární zaokrouhlení a 3,05
(25 %) 2:4 maska, která vybírá prořezávané pozice až po zaokrouhlení — kdy mají
všechny nenulové hodnoty stejnou velikost — a proto nuluje pozici 0 v každém bloku
po čtyřech u 81 % bloků a pozici 1 u 76 %, proti 27 % a 16 % u zbylých dvou.

## Rozsah

Přesnost schématu na úrovni vah byla vyhodnocena samostatně a dřív, včetně
one-shot PTQ, tréninku od nuly s 2:4 až na 5 miliardách tokenů a měření reálného
2:4 zrychlení na RTX 3090 (1,6–2,0×). Vada v pořadí operací, která z toho vyšla,
byla nahlášena jako issue #1 dne 27. dubna 2026 a je pořád otevřená. Tento
repozitář pokrývá jen tvrzení, která přidal preprint ze září 2026.

## Doprovodná vyhodnocení

- [`octonion-mppt-eval`](https://github.com/karagos01/octonion-mppt-eval) — oktonionová dvouvrstvá šablona s asociátorem a magnonová tvrzení
- [`causal-trilogy-eval`](https://github.com/karagos01/causal-trilogy-eval) — trilogie CQFT / PCTP / SOTP ze září 2026

## Licence

Kód (všechny `*.py` a `run_all.sh`): MIT, viz `LICENSE`.
`vendor/xt_quant.py`: MIT, © 2026 michalmazgal, viz `vendor/LICENSE.xt_quant`.
Text `PAPER.md` a `PAPER.cs.md`: CC BY 4.0.
