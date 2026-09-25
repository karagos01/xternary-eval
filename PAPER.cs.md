# Bitové účetnictví, propustnost a ternární KV cache: nezávislé vyhodnocení X-Ternary

**Autor:** karagos01 · **září 2026**
**Licence:** tento text CC BY 4.0 · přiložený kód MIT

## Abstrakt

X-Ternary (Mazgal, Zenodo, DOI [10.5281/zenodo.22877345](https://doi.org/10.5281/zenodo.22877345))
navrhuje 2bitový inferenční engine pro velké jazykové modely: ternární váhy
{−1, 0, +1} v kombinaci s 2:4 strukturní řídkostí od NVIDIA, blokové FP8
škálování, ternární KV cache a oktonionovou „skrytou kauzální paměť" propojenou
s tenzorovou vrstvou přes oktonionový asociátor. Přiložený repozitář obsahuje
67 řádků Pythonu, které implementují první dvě z těch věcí. Tato práce doplňuje
chybějící měření ke třem tvrzením, kterých se repozitář nedotýká.

Za prvé, bitové účetnictví je obrácené. Po vynucení 2:4 řídkosti zbývá z 81
ternárních bloků po čtyřech dosažitelných jen 33, takže schéma nese 1,26 bitu na
váhu — o 20,4 % méně než BitNet b1.58 — a ukládá to ve 2,25 bitu, tedy o 42 %
více. Za druhé, tvrzení o 70 miliardách parametrů při 60 tokenech za sekundu na
24GB kartě je nad pamětovým stropem té karty: formát, který článek specifikuje,
má strop 47,6 tok/s při 100% využití sběrnice, zatímco BitNet b1.58, který
článek odmítá, dosahuje 66,9. Za třetí, sčítání v pohyblivé řádové čárce na GPU
už asociativní není, a to u 31,9 % náhodných trojic; právě to je příčinou
nereprodukovatelnosti, kterou článek chce léčit přidáním nonasociativity.
Nakonec je přímo změřena ternární KV cache: perplexita Qwen3-1.7B-Base na
WikiText-2 stoupne se zveřejněným kódem z 10,40 na 52 004, tedy 5 001×, tam kde
správně provedená 4bitová cache stojí 1,2 %. Všechny skripty jsou v tomto
repozitáři.

## Co článek tvrdí a co je implementováno

| Tvrzení v článku | V repozitáři |
| --- | --- |
| Ternární váhy {−1, 0, +1}, absmean škálování | ano, `xt_quant.py:22` |
| 2:4 strukturní řídkost jako čtvrtý stav `x` | ano, `xt_quant.py:27-37` |
| Přesně 2 bity na váhu, „zero packing overhead" | žádné pakování; vrací denzní float tenzor |
| Blokové FP8 škálování po skupinách 32 | chybí |
| Ternární KV cache (sekce 2.3) | chybí |
| Oktonionová skrytá kauzální paměť, most přes asociátor (3.1) | chybí |
| TMA a `cuda::barrier` vynucující kauzální pořadí (3.1) | chybí |
| Přesměrování na CUDA ALU při spouštěcím tokenu (3.2) | chybí |
| Dvoufázový trénink STE + EWC (4) | chybí |
| 70B na 24 GB při ~60 tok/s, 80% úspora paměti | chybí |

Přesnost schématu na úrovni vah vyhodnotil autor této práce už dříve, včetně
one-shot PTQ na GPT-2, tréninku od nuly s 2:4 na 2,5 a 5 miliardách tokenů a
měření reálného 2:4 zrychlení na RTX 3090 (1,6–2,0×). Tyto výsledky jsou
předchozí práce a nejsou tu opakovány; vada v pořadí operací, kterou odhalily,
byla nahlášena jako issue #1 dne 27. dubna 2026 a zůstala bez odpovědi. Tato
práce pokrývá jen tvrzení, která jsou novější.

## Metoda

`bits.py` vyjmenuje dosažitelný kódový prostor a změří empirickou entropii
tenzorů, které zveřejněný kvantizátor vyrábí. `roofline.py` změří propustnost
čtení na testované kartě a přepočte ji na strop tokenů za sekundu.
`determinism.py` měří asociativitu sčítání v pohyblivé řádové čárce a závislost
výsledku GEMM i logitů reálného modelu na velikosti batche. `kvcache.py`
vyhodnotí perplexitu Qwen3-1.7B-Base na 32 okénkách po 1024 tokenech z testovací
části WikiText-2, s K a V kvantovanými na cestě do attention podtřídou
`DynamicCache`; varianty X-Ternary volají `XTQuantizer.quantize` z původního
repozitáře beze změny (vložený ve `vendor/` z commitu `9f27dfa`).
`attribution.py` rozdělí naměřenou ztrátu mezi ternární zaokrouhlení a 2:4 masku
a změří, které hodnoty maska odstraňuje. Hardware: jedna RTX 3090, PyTorch 2.5.1,
transformers 4.51.3.

## Výsledky

### Bitové účetnictví

Blok čtyř ternárních vah má 3⁴ = 81 stavů. Omezení 2:4 dovolí nejvýš dvě
nenulové hodnoty, takže zbývá 1 + 8 + 24 = 33 bloků, tedy log₂(33)/4 = 1,261
bitu na váhu.

| schéma | potřebuje | ukládá | režie |
| --- | --- | --- | --- |
| BitNet b1.58, denzní ternary | 1,585 b | 1,585 b | 0 % |
| X-Ternary, 2 bity/váhu dle README | 1,261 b | 2,000 b | +58,6 % |
| X-Ternary + FP8 škála /32, článek §2.2 | 1,261 b | 2,250 b | +78,4 % |

Na reálných rozděleních vah jsou vyrobené tenzory ještě dál od uniformního:
empirická entropie je 1,089 bitu na váhu u gaussovských a 1,170 u laplaceovských
vah, takže uložené 2 bity nesou 84 % a 71 % režie. Premisa článku — že BitNetovy
log₂3 ≈ 1,58 bitu plýtvají 2bitovým prostorem — je tedy obrácená: X-Ternary nese
o 20,4 % méně informace než BitNet a vydá na ni o 42 % více bitů.

Ta redundance je v návrhu explicitní. README uvádí čtyři stavy `01`, `10`, `00`,
`11`, z nichž `00` („klasická nula") a `11` („strukturní nula, vysoká impedance,
fyzicky přeskoč operaci") přispívají k `y = Wx` oba nulou. Čtyřsymbolová abeceda
tedy nese tři hodnoty a 2 − log₂3 = 0,415 bitu na váhu je nevyužitých
konstrukcí. `theory.md` to prezentuje jako využití „informační redundance
2bitového prostoru".

### Pamětový strop

Dekódování jednoho toku přečte celou sadu vah jednou na token, takže strop dává
propustnost, ne aritmetika. Naměřená propustnost čtení na RTX 3090 byla 885 GB/s,
tedy 94 % katalogových 936 GB/s. Pro 70 · 10⁹ parametrů:

| formát | GiB | strop @katalog | strop @měřeno |
| --- | --- | --- | --- |
| FP16 | 130,4 | 6,7/s | 6,3/s |
| BitNet b1.58 | 13,0 | **66,9/s** | 63,2/s |
| X-Ternary, 2 bity/váhu | 16,3 | 53,5/s | 50,5/s |
| X-Ternary + FP8 škála /32 | 18,3 | **47,6/s** | 44,9/s |
| 2:4 komprimované, 2 × 2 b hodnota + 4 b index | 16,3 | 53,5/s | 50,5/s |

Strop předpokládá 100% využití sběrnice, prázdnou KV cache, žádné aktivace a
nulovou režii dekvantizace; reálné enginy dosahují 60–80 % z něj. Formát, který
článek specifikuje, tedy tvrzených 60 tok/s dosáhnout nemůže a schéma, které
článek odmítá, je v jeho vlastní hlavní metrice rychlejší.

K tomu dvě věci. 2:4 komprese neušetří žádnou paměť: dvě přeživší hodnoty po
dvou bitech plus čtyři bity indexových metadat je stejných osm bitů na čtyři váhy
jako denzní 2bitové uložení, takže se ty dvě myšlenky v paměťovém rozpočtu
neskládají. A 2:4 kupuje aritmetickou propustnost, ne pamětovou, což je přesně
naopak, než čím je dekódování jednoho toku omezené. Sparse Tensor Cores navíc
přijímají 2:4 vzor v fp16, bf16, int8 nebo tf32 na Ampere a fp8 na Hopper;
2bitová řídká cesta neexistuje, takže se váhy musí před řídkým GEMM rozbalit do
podporovaného typu, a tvrzení o „okamžitém zdvojnásobení propustnosti bez ztráty
přesnosti" tedy neplatí ani v jedné polovině.

### Determinismus

Článek přisuzuje halucinace „asociativní pravděpodobnosti" v GEMM a navrhuje
nonasociativitu vynutit. Sčítání v pohyblivé řádové čárce ale asociativní není
už samo od sebe:

```
fp16: (a+b)+c != a+(b+c) u 31,92 % náhodných trojic
bf16: 31,88 %      fp32: 31,86 %
```

Výsledek tedy závisí na rozvrhu redukce. Se stejným vstupním vektorem není
`x @ W.T` při batchi 64 bitově totožné s týmž řádkem počítaným samostatně a na
Qwen3-1.7B přeskládá stejný prompt při batchi 8 čtyři z deseti nejlépe
hodnocených tokenů. Nonasociativita tam už je a je to mechanismus, který stojí
za právě tou nereprodukovatelností, kterou chce článek odstranit; determinismus
plyne z ustálení rozvrhu redukce, ne z přidání další nonasociativity. Halucinace
jsou samostatná věc: bitově deterministický model umí být s jistotou vedle.

### Ternární KV cache

Sekce 2.3 uvádí, že KV cache je „identically compressed using the ternary
scheme". Perplexita Qwen3-1.7B-Base na WikiText-2, 32 okének po 1024 tokenech:

| varianta | bit/hodnotu | PPL | × baseline |
| --- | --- | --- | --- |
| fp16 baseline | 16,00 | **10,40** | 1,0× |
| INT8 per-token | 8,00 | 10,46 | 1,0× |
| INT4, K per-channel + V per-token | 4,13 | **10,52** | 1,0× |
| INT4 po skupinách 32 | 4,13 | 27,35 | 2,6× |
| INT3 po skupinách 32 | 3,13 | 273,5 | 26,3× |
| INT2 po skupinách 32 | 2,13 | 7 342 | 706× |
| ternary po skupinách 32 | 1,71 | 7 326 | 705× |
| ternary per-token, nejlepší případ | 1,58 | 6 287 | 605× |
| ternary per-tensor, jako `xt_quant.py` | 1,58 | 13 753 | 1 323× |
| **X-Ternary verbatim z repozitáře** | 2,00 | **52 004** | **5 001×** |
| X-Ternary s per-token škálováním | 2,00 | 62 129 | 5 975× |
| X-Ternary bez aplikované gammy | 2,00 | 722 019 | 69 434× |
| X-Ternary s opravou pořadí z issue #1 | 2,00 | 11 735 | 1 129× |

Správně provedená 4bitová cache je zdarma, tři bity stojí faktor 26 a útes je
mezi čtyřmi a třemi bity. Ternary je dva řády za ním a zveřejněné schéma je
dalších osmkrát horší než prosté ternary. Oprava pořadí pomůže 4,4× a pořád je
to nepoužitelné.

V bitech, se slovníkem 151 669 tokenů, kde uniformní hádání je 17,21 bitu na
token: baseline předpovídá na 3,38 bitu, drží tedy 13,83 bitu předpovědní síly.
Se zveřejněnou cachí drží 1,54 bitu, tedy 11,2 % původní. Bez aplikované gammy
předpovídá na 19,46 bitu, tedy hůř než tažení tokenů uniformně náhodně.

Z 12,29 ztraceného bitu na token je 9,24 (75 %) ternární zaokrouhlení — daň,
kterou platí každá ternární KV cache — a 3,05 (25 %) je 2:4 maska.

### Kde maska škodí

Maska vybírá `topk(|q|, k=2, largest=False)` až po ternárním zaokrouhlení, kdy má
každá nenulová hodnota velikost γ. Shody se pak rozhodují podle indexu, takže
výběr je poziční, ne podle důležitosti. Maska nuluje pozici 0 v každém bloku u
81 % bloků a pozici 1 u 76 %, proti 27 % a 16 % u pozic 2 a 3; pravidlo podle
velikosti žádnou pozici nepreferuje. Cena, na tenzorech K a V modelu Qwen3-1.7B:

| tenzor | zabitých nenulových | zabitá energie | energie při správném výběru |
| --- | --- | --- | --- |
| K, vrstva 0 | 3,9 % | 0,3 % | 0,1 % |
| V, vrstva 0 | 17,8 % | 31,6 % | 9,3 % |
| K, vrstva 14 | 14,0 % | 16,0 % | 4,8 % |
| K, vrstva 27 | 12,8 % | **26,9 %** | **2,2 %** |
| V, vrstva 27 | 20,0 % | 34,6 % | 12,4 % |

Výběr podle pozice zničí u klíčů v poslední vrstvě dvanáctkrát víc energie, než
je nutné, a u hodnot 3,4krát. Ta vada navíc roste s tím, jak se zlepšuje zbytek
schématu: per-token škálování, normálně vždy lepší, nechá po zaokrouhlení přežít
víc hodnot (68,5 % proti 66,3 % u V ve vrstvě 27), takže poziční maska má co
zabíjet, a proto celkově skóruje horš (62 129 proti 52 004).

## Dvě vady ve zveřejněném kódu

`quantize()` vrací `(xt_weight, gamma)` s hodnotami v {−1, 0, +1}; volající musí
násobit γ, aby dostal původní měřítko. Ukázka v `xt_quant.py:54` to nedělá a
`estimate_memory_saving` také ne. Použito takto je výstup vrstvy přeškálovaný
přibližně 1/γ; to je ten řádek s 69 434×.

`estimate_memory_saving` vrací `original_size * 2/16` bez ohledu na cokoli.
Nemodeluje ani FP8 blokové škály ze sekce 2.2, ani indexová metadata 2:4, a
nezávisí na tenzoru, na který se volá.

## Omezení

Měření perplexity používá jeden model o 1,7 miliardě parametrů, jeden dataset a
jednu délku okna; pořadí variant je napříč vrstvami stabilní, absolutní hodnoty
ale nejsou benchmarkem metod pro KV cache. Zdejší 4bitová reference je
přímočaré seskupení per-channel/per-token, ne věrná reimplementace KIVI nebo
KVQuant, a na této velikosti modelu je už bezeztrátová, takže silnější reference
by mezeru mohla jen zvětšit. Pamětový strop je roofline, ne end-to-end měření
propustnosti; ohraničuje tvrzení shora, nereprodukuje ho. Oktonionová vrstva,
FP8 blokové škálování, přesměrování na CUDA ALU a dvoufázový trénink se
nevyhodnocují vůbec, protože je článek nedefinuje na úroveň, kterou by šlo
implementovat — konkrétně nikde není přiřazení fyzikálních ani modelových veličin
ke složkám oktonionu e₀…e₇, což je stejná mezera jako v autorově návrhu MPPT.

## Závěr

Byla otestována tři tvrzení, která repozitář neimplementuje, a každé padá
dřív, než se dojde k otázce kvality implementace. Bitové účetnictví je obrácené:
schéma ukládá méně informace ve více bitech než schéma, které kritizuje. Hlavní
číslo o propustnosti je nad pamětovým stropem karty, kterou článek jmenuje.
Argument o determinismu zaměňuje příčinu za lék, protože nonasociativita
pohyblivé řádové čárky tam už je a je to ona, kdo dělá výstup závislý na
velikosti batche. Jediné tvrzení, které bylo skutečně otevřenou otázkou —
ternární KV cache — bylo změřeno a stojí faktor 5 001 na perplexitě tam, kde
správná 4bitová cache stojí 1,2 %.

Tři čtvrtiny té ztráty jsou vlastní ternárním klíčům a hodnotám a potkaly by
kohokoli. Zbylá čtvrtina vzniká tím, že se prořezávané pozice vybírají až po
zaokrouhlení, kdy už žádná informace o uspořádání nezbývá — to je vada nahlášená
původnímu repozitáři v dubnu 2026, pořád otevřená, a pořád ta menší polovina
problému.

## Dostupnost dat

Všechny skripty, vložený kvantizátor a celý výstupní log jsou na
https://github.com/karagos01/xternary-eval. `./run_all.sh` reprodukuje každé
číslo v tomto textu za asi pět minut na jedné RTX 3090.

## Reference

1. M. Mazgal. *X-Ternary: A Deterministic Causal 2-Bit Inference Engine for Large Language Models.* Zenodo, 2026. DOI 10.5281/zenodo.22877345.
2. S. Ma a kol. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764, 2024.
3. A. Mishra a kol. *Accelerating Sparse Deep Neural Networks.* arXiv:2104.08378, 2021.
4. Z. Liu a kol. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024, PMLR 235:32332–32344. arXiv:2402.02750.
5. C. Hooper a kol. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* arXiv:2401.18079, 2024.
6. M. Sun, Z. Liu, A. Bair, J. Z. Kolter. *A Simple and Effective Pruning Approach for Large Language Models.* arXiv:2306.11695, 2023.
7. E. Frantar, D. Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML 2023. arXiv:2301.00774.
8. D. Goldberg. *What Every Computer Scientist Should Know About Floating-Point Arithmetic.* ACM Computing Surveys 23(1):5–48, 1991.
9. S. Merity, C. Xiong, J. Bradbury, R. Socher. *Pointer Sentinel Mixture Models.* arXiv:1609.07843, 2016.
10. Qwen Team. *Qwen3-1.7B-Base.* https://huggingface.co/Qwen/Qwen3-1.7B-Base
11. Microsoft. *bitnet-b1.58-2B-4T.* https://huggingface.co/microsoft/bitnet-b1.58-2B-4T — natrénovaný model s ternárními vahami, jehož kvantizační konfigurace pokrývá jen lineární vrstvy; KV cache drží v 16 bitech.
