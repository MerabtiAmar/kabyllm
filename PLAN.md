# KabyLLM — v0 → v1 Plan

**Goal:** a Kabyle conversational assistant + writing aid.
**Constraints:** Kaggle GPUs only (T4×2 / P100 / TPU v3-8), fully automated data pipeline, personal project.
**Strategy:** cross-lingual adaptation of a strong small base model. Not from scratch.

---

## 0. The reframe

The v0 model failed for a reason no amount of architecture tuning can fix.

**Data-to-parameter ratio.** 43.27M parameters trained on 995,159 tokens. Chinchilla-optimal for
1M tokens is a ~50K-parameter model. You were ~800× over-parameterized. `train=1.33 / val=6.21`
is not an undertrained model, it is a **memorized** one. It had no other option.

**The ceiling.** GPT-3.5 saw ~300B+ tokens. All Kabyle text that exists on Earth is, generously,
40–60M tokens. That is 3–4 orders of magnitude short. **Reasoning and world knowledge cannot be
learned from 40M tokens in any language.**

**The way around it.** Take a base model that already has reasoning and world knowledge from
trillions of tokens of English/French/Chinese, and spend the Kabyle data teaching it *only the
language*. Cross-lingual transfer means the Kabyle corpus carries linguistic competence, not
intelligence. This is the recipe behind SeaLLM, Sailor, Typhoon, LLaMAX, and Aya, and it works
at exactly your data scale.

**What "success" looks like:** a 1.7B model that writes fluent, orthographically correct Kabyle,
follows instructions, translates Kab↔Fr/En competently, and answers general questions with
roughly the world knowledge of its base model. That is a genuinely useful assistant. It will not
be GPT-3.5, and nothing available to you will be.

---

## 1. Diagnosis of v0

### Keep
- `KabyleGPT` architecture is fundamentally sound (pre-LN, weight tying, GELU, warmup+cosine,
  grad clipping, GPT-2 init). Nothing wrong with it. It is simply the wrong tool now.
- Keep [notebooks/v0_gpt_from_scratch_wiki.ipynb](notebooks/v0_gpt_from_scratch_wiki.ipynb) frozen as the **documented v0 baseline**. It is a real
  result to compare against. Do not invest further in it.

### Fix
| Problem | Evidence | Consequence |
|---|---|---|
| Trained on the **partial** corpus | Used `kabyle_wiki_corpus.txt` (741KB — scrape died on rate-limits at batch 21/143), not `kabyle_wiki_corpus_cleaned.txt` (4.6MB) | 6× data left unused |
| **Word-level tokenizer** | 35,207 vocab, 42% of corpus vocab is hapax legomena | Cannot generalize over Kabyle agglutinative morphology. `yettwanɣa`, `zedɣen-tt`, `tessufuɣ-d` are unanalyzable atoms |
| **6.5% UNK rate** | top-35k covers only 93.5% of tokens | Generations full of holes; had to `.replace("<UNK>", " ")` |
| **Clitics split** | `re.findall(r'\w+\|[^\w\s]')` splits on `-` | `zedɣen-tt` → `zedɣen`, `-`, `tt`. Destroys morphology |
| **`.lower()`** | in `build_vocab` and `encode` | Loses proper nouns, sentence boundaries |
| `DataParallel` | not `DistributedDataParallel` | ~40% throughput loss on 2×T4 |
| fp32 training | no AMP | ~3× slower than fp16 on T4 tensor cores |
| Weight decay on everything | `AdamW(model.parameters(), weight_decay=0.01)` | LayerNorm/bias/embeddings should be excluded |
| No eval harness | only eyeballed generations | No way to know if anything improves |

### The encoding problem — solved, and it is not a Unicode bug

`Mouloud_Feraoun_Mmis_n_igellil_extracted.txt` is in a **legacy pre-Unicode Berber font encoding**.
The PDF's embedded fonts map Kabyle letters onto Latin-1 codepoints. PyMuPDF faithfully returned
the codepoints, not the letters. Verified mapping:

| glyph | real | evidence |
|---|---|---|
| `$` `£` `ù` | ɣ Ɣ ɣ | `ùef`→ɣef · `yettwan$a`→yettwanɣa |
| `ε` `∑` `µ` | ɛ Ɛ ɛ | `∑emmi`→Ɛemmi · `lqaµa`→lqaɛa |
| `v` | ḍ | `yewwev-ed`→yewweḍ-ed |
| `ô` `Ö` | ṛ Ṛ | `uôumi`→uṛumi |
| `û` `Û` | ṣ Ṣ | `laûel`→laṣel |
| `î` `§` | ṭ | `aîas`→aṭas · `ime§§i`→imeṭṭi |
| `ê` `Ë` | ḥ Ḥ | `yelêan`→yelḥan |
| `é` `Á` | ẓ Ẓ | `neéra`→neẓra |
| `ç` | č | `maççi`→mačči |

**Critical:** several glyphs map to the same letter (`$`/`ù`→ɣ, `ε`/`µ`→ɛ, `î`/`§`→ṭ). That means
the PDF embeds **multiple font subsets** (regular / bold / italic), each with its own encoding.
A global find-replace will corrupt the text. Extract with `page.get_text("rawdict")` to get the
font name per span, then apply a **per-font** mapping table.

`051_Omar_Dahmoune_extracted.txt` is clean Unicode but uses **Cyrillic `Ԑ`/`ԑ` (U+0510/0511)** and
**Greek `ε` (U+03B5)** where Latin `Ɛ`/`ɛ` (U+0190/U+025B) belong. Your
[data_prep.ipynb](data_prep.ipynb) `valid_chars` whitelist *preserves* these confusables
(`ɛ`, `ɤ`, `Ԑ`, `ԑ` are all in the list) — so it locked the corruption in instead of fixing it.
A whitelist is the wrong tool; you need a normalizer.

---

## 2. Phase 1 — Orthographic normalization layer

**This is the foundation. Everything downstream depends on it. Build it first.**

Every character variant that survives into training splits your vocabulary and wastes capacity.
Deliverable: a single `kab_normalize.py` module used by every later stage.

### 1.1 Unicode normalization
- `unicodedata.normalize("NFC", text)` — collapses `h`+U+0323 into `ḥ` (U+1E25).
- Strip zero-width chars, BOM, NBSP → space, curly quotes → straight, `–—` → `-`.

### 1.2 Homoglyph folding (build the table explicitly)
```
ε U+03B5 GREEK SMALL EPSILON        → ɛ U+025B
ԑ U+0511 CYRILLIC SMALL KOMI DZJE   → ɛ U+025B
Ԑ U+0510 CYRILLIC CAPITAL KOMI DZJE → Ɛ U+0190
ɤ U+0264 LATIN SMALL RAMS HORN      → ɣ U+0263
γ U+03B3 GREEK SMALL GAMMA          → ɣ U+0263
ƹ, ʕ, ς, 3 (in Arabizi context)      → ɛ
```
Target alphabet is the standard Kabyle Latin (INALCO/HCA) set:
`a b c č d ḍ e ɛ f g ǧ ɣ h ḥ i j k l m n q r ṛ s ṣ t ṭ u w x y z ẓ`.

### 1.3 Legacy font decoding
`decode_legacy_pdf(path)` using `page.get_text("rawdict")`:
1. Walk spans, read `span["font"]`.
2. Look up that font in a `FONT_MAPS: dict[str, dict[str, str]]` table.
3. Unknown font → log it, emit the span verbatim, and flag the document for review.

Bootstrap new font maps automatically: decode with a candidate map, then score against
`boffire/hunspell-kab`. The correct map is the one minimizing OOV rate. This turns font
discovery into a search problem you can run unattended.

### 1.4 Ad-hoc / Arabizi orthography
A lot of web Kabyle uses French-influenced spelling (`gh`→ɣ, `kh`→x, `th`→t, `dh`→ḍ, `ch`→c)
or Arabizi digits (`3`→ɛ, `7`→ḥ, `9`→q). Do **not** try to auto-convert — it is ambiguous and
will corrupt correct text. Instead: **detect and quarantine**. Score each document's OOV rate
against the hunspell dictionary and route high-OOV documents to a separate low-priority bucket
(or drop them). Standard-orthography text is plentiful enough that you can afford to be strict.

### 1.5 Tifinagh
If any Tifinagh (ⵜⵉⴼⵉⵏⴰⵖ) text appears, transliterate to Latin — that is where all your other
data lives. `abdelhaqueidali/Kabyle-Latin-to-Tifinagh-Parallel-Corpus` gives you the mapping.

**Gate:** re-run the charset audit on every source. The final alphabet must be exactly the
standard set plus ASCII punctuation and digits. No Greek, no Cyrillic, no Latin-1 accents.

---

## 3. Phase 2 — Corpus aggregation (automated)

Delete [wikiKab.ipynb](wikiKab.ipynb)'s scraper. Two replacements:

**Wikipedia — use the official dump, not the API.**
```
https://dumps.wikimedia.org/kabwiki/latest/kabwiki-latest-pages-articles.xml.bz2
```
Process with `wikiextractor` or `mwparserfromhell`. No rate limits, complete, and vastly cleaner
than the 40-regex `clean_wiki_text` chain (which was leaving `Imagraden iqqenen` section stubs,
empty template residue, and taxonomy tables in the output — visible in
`kabyle_wiki_corpus_cleaned.txt`). Also pull **kabwiktionary** — ~10k+ entries, excellent for
morphology and definitions.

**Everything else — `load_dataset()`.**

### Tier A — monolingual, load directly
| Dataset | Size | Register |
|---|---|---|
| `Imsidag-community/kabyle-corpus-ummto` | 690,917 sent. | Academic/literary (PDF-mined) |
| `boffire/tatoeba-kabyle-mono-cleaned` | 756,774 sent. | Conversational, short |
| `Imsidag-community/kabyle-corpus-ubouira` | 336,982 sent. | Academic |
| `Imsidag-community/kabyle-corpus-hca` | 188,620 sent. | Institutional |
| `Imsidag-community/kabyle-raw-text` | 100K–1M | mixed |
| `Imsidag-community/kabyle-datasets-mono` | ? | mixed |
| Kabyle Wikipedia + Wiktionary | ~1M words | Encyclopedic |
| Common Voice `kab` sentence corpus | ~90k sent. | Colloquial, clean |
| `cis-lmu/GlotCC-V1` (`kab_Latn`) | ? | Web crawl — filter hard |
| `boffire/kabyle-lyrics`, `boffire/timucuha-kabyle-tales` | small | Song, folk narrative |
| Your decoded PDFs (Feraoun, Dahmoune) | ~45k words | Literary |

### Tier B — parallel (for translation skill + SFT construction)
| Dataset | Size | Note |
|---|---|---|
| `boffire/nllb_en_kab` | 2,786,012 pairs | **LASER-mined, noisy — filter aggressively** |
| `Imsidag-community/nllb_en_kab` | 2,484,297 pairs | Already GlotLID-filtered |
| `boffire/tatoeba-en-kab` | 244,954 pairs | Splits provided. High quality |
| `boffire/kabyle-english-TM` | 121,725 pairs | Translation memory |
| `Imsidag-community/english-kabyle-parallel` | 130,883 pairs | Tatoeba-derived |
| Tatoeba `fra-kab` | ~100k pairs | **French matters** — Kabyle speakers code-switch |
| `boffire/kabyle-english-translatewiki` | 8,871 pairs | UI/software register |
| `boffire/kab-en-toponyms-sentences` | 32,024 pairs | Geographic |

### Tier C — structured linguistic resources (high value per byte)
- **`boffire/kabyle-verbs`** — 6,198 verbs × ~344,745 conjugated forms across aorist, preterite,
  imperative, participles. This is a complete morphological paradigm table. Use it for synthetic
  morphology tasks in SFT, and to sanity-check tokenizer segmentation.
- **`boffire/hunspell-kab`** — spell-checker + morphology. Your automated quality metric.
- `boffire/kabyle-named-entities`, `boffire/kabyle-toponyms` (8,849 place names)
- `kmerakeb/kabyle-dialog-corpus` — dialogue, directly on-target
- `Agurzil-Amazigh/tafsut-lexique-mathematiques-kabyle` — technical vocabulary

### Tier D — raw PDFs to mine with your Phase-1 decoder
`boffire/ayamun-pdfs` (230 PDFs), `boffire/bouira` (395 theses), `boffire/adlis-pdfs`,
`Imsidag-community/kabyle-adlis-pdfs`.

### Cleaning pipeline (run in this order)
1. Normalize (Phase 1).
2. **Language ID** — `cis-lmu/GlotLID` v3, keep `kab_Latn` above threshold. Non-negotiable for
   web-crawl and NLLB sources, which are full of Arabic, French, and other Berber varieties.
3. **Quality filters** — length bounds, alphabetic ratio, digit ratio, max repeated-line ratio,
   hunspell OOV rate cap.
4. **Deduplication** — exact hash first, then **MinHash-LSH near-dup at document level**
   (`datasketch`, threshold 0.8). Critical: the community datasets overlap heavily. Tatoeba
   appears in at least four of them. Your own files already had 2,084 duplicated lines between
   `kabyle_wiki_corpus_cleaned.txt` and `kab_Latn.txt`. Dedup **after** merging everything.
5. **Held-out splits, stratified by source** — carve out per-domain val/test sets *before*
   training. You had no eval; this is where you fix that.

**Expected yield: 20–40M words ≈ 35–65M tokens.** 20–40× your v0 corpus.

**Provenance caveat:** `boffire/*` and `Imsidag-community/*` are community uploads of unverified
quality. Sample ~200 sentences from each and read them. You are a native speaker — this is your
one irreplaceable advantage and it costs you an hour. Drop any source that reads wrong.

---

## 4. Phase 3 — Tokenizer

Cheap, decisive, and done before any training.

### 3.1 Measure fertility — **MEASURED, and it corrects this plan**

Run: `python scripts/03_tokenizer.py --bench`

| Tokenizer | tokens/word | diacritic penalty |
|---|---|---|
| Qwen3 (1.7B / 4B) | **2.77** | 1.06× |
| SmolLM3-3B | 2.96 | 1.47× |
| Mistral-7B-v0.3 | 3.03 | 1.13× |
| *NLLB-200 (trained with kab_Latn)* | *2.26* | *0.88×* |

English byte-BPE is ~1.3, so Kabyle costs Qwen3 **2.1× more tokens** for the same content.
That is a direct multiplier on training FLOPs and a halving of effective context.

Two corrections to what this section originally assumed:

1. **The diacritics are not the problem.** I predicted a 2–4× tax from ɣ/ɛ/ḍ being 2-byte
   UTF-8 and byte-falling-back. Measured, Qwen3's penalty on diacritic-bearing words is
   only **1.06×** — its byte-BPE handles them nearly for free. The real cost is that Kabyle
   *morphology* is unfamiliar: agglutinated forms like `yettwanɣa` have no learned subwords.
2. **The realistic target is ~1.9, not 1.6.** NLLB-200 actually trained on Kabyle and still
   needs 2.26 tokens/word. Vocabulary extension is grafting Kabyle morphemes onto a base
   that has none, so beating a model that pretrained on the language is not the expectation.

Extension is still clearly worth doing — but the gain comes from learning morphemes, not
from repairing diacritic handling.

### 3.2 Extend the vocabulary
1. Train a SentencePiece/BPE model (8k–16k merges) on the clean Kabyle corpus.
2. Merge only tokens **not already** in the base vocab.
3. Resize embeddings. **Initialize each new row as the mean of the old-tokenizer embeddings of
   the pieces it decomposes into.** This is the single most important detail — mean-init
   converges dramatically faster than random init and avoids the loss spike.
4. Re-measure fertility. Target ≤1.8 tokens/word.

### 3.3 Decision rule
- Qwen3 (151,936 vocab) — strongest reasoning per parameter, 119 languages claimed.
- Gemma 3 (262,144 vocab, SentencePiece) — broadest low-resource coverage, likely better native
  diacritic handling, and strong French. But a 262k vocab on a 1B model means embeddings dominate
  the parameter budget.

**Default: `Qwen/Qwen3-1.7B-Base`.** Switch to `google/gemma-3-1b-pt` if its measured fertility is
materially better and French quality matters more to you than raw reasoning.

---

## 5. Phase 4 — Continued pretraining (CPT)

Teach the base model Kabyle without destroying what it knows.

### Configuration
| Setting | Value | Why |
|---|---|---|
| Base | `Qwen/Qwen3-1.7B-Base` | fits 2×T4; best size/quality tradeoff |
| Method | LoRA `r=256, alpha=512`, all linear layers | full FT on 35M tokens causes catastrophic forgetting |
| **Also fully trainable** | `embed_tokens` (+ `lm_head` if untied) | new vocab rows must actually learn; LoRA can't do this |
| **Replay mix** | **60% Kabyle / 40% Fr+En** | prevents forgetting. Not optional |
| Precision | fp16 + `GradScaler` | **T4 is Turing (SM 7.5) — bf16 unsupported** |
| Attention | `attn_implementation="sdpa"` | **FlashAttention-2 needs SM 8.0+; not available on T4** |
| Seq length | 2048 | longer wastes memory for marginal gain at this scale |
| Optimizer | `adamw_8bit` (bitsandbytes) | memory |
| LR | 2e-4 (LoRA), 1e-5 (embeddings) | embeddings need a much lower LR |
| Schedule | 3% warmup → cosine → 10% floor | |
| Grad checkpointing | on | |
| Epochs | 2–3 | |

**Use Unsloth.** It supports T4 and Qwen3, gives ~2× throughput and large memory savings, and
handles the fp16/LoRA/checkpointing plumbing that will otherwise eat your weekends.

### Kaggle session management — plan for this explicitly
Kaggle caps notebook runtime at **12h (GPU) / 9h (TPU)** with a **30h/week** quota. Your v0 run
died at step 8207 to a `KeyboardInterrupt` and you lost the run. Build resumability first:
- Checkpoint every ~500 steps to `/kaggle/working`.
- `push_to_hub` each checkpoint (private repo) — Kaggle output does not survive reliably.
- On start, detect the latest hub checkpoint and resume optimizer + scheduler + dataloader state.

### Compute estimate
~200M training tokens (35M Kabyle × ~2.5 epochs, plus replay). FLOPs ≈ 6ND = 6 × 1.7e9 × 2e8
≈ 2.0e18. At a realistic ~35 TFLOPS effective across 2×T4 with LoRA + checkpointing:
**≈16 GPU-hours.** Comfortably inside one week's quota, across two sessions.

Fallbacks: `Qwen3-0.6B-Base` ≈ 6h if you want a fast first loop. `Qwen3-4B-Base` ≈ 38h
(two weeks of quota) if the 1.7B result is promising and you want to push.

**Gate:** held-out Kabyle perplexity must drop substantially, *and* English MMLU must not fall
more than a few points versus the untouched base. If English collapses, raise the replay ratio.

---

## 6. Phase 5 — Instruction tuning (SFT)

This is what turns a language model into an assistant, and it is the hardest part because **no
Kabyle instruction dataset exists.** You have to manufacture one, automatically.

**The failure mode to design around: translationese.** If you bulk-translate an English
instruction set with NLLB, the model learns to produce NLLB-quality Kabyle — stilted, calqued,
wrong. Avoid this by making the *responses* naturally-occurring Kabyle wherever possible, and
letting only the *instructions* be synthetic.

### 5.1 Corpus-grounded tasks (~60% of the mix) — responses are real Kabyle
| Task | Construction | Source |
|---|---|---|
| Q&A | "D acu-t X?" → article lead paragraph | Wikipedia |
| Summarization | article body → its own lead | Wikipedia |
| Continuation / writing aid | first half → second half | Literature, UMMTO corpus |
| Translation (both directions, En **and** Fr) | direct pairs | Tatoeba, TM |
| Morphology | "Sizreg amyag *X* deg urmir" → paradigm | `boffire/kabyle-verbs` |
| Definition | headword → gloss | Wiktionary, Tafsut lexicon |
| Spelling correction | hunspell-informed corruption → original | any clean text |
| NER / geography | toponym + context | `kabyle-toponyms`, `kabyle-named-entities` |
| Dialogue | as-is | `kmerakeb/kabyle-dialog-corpus` |

### 5.2 Translated instructions (~20%) — filtered hard
Take 20–50k high-quality English/French instructions from `HuggingFaceH4/no_robots`,
`databricks/databricks-dolly-15k`, `CohereForAI/aya_dataset`, `OpenAssistant/oasst2` (French
subset). Translate with `facebook/nllb-200-distilled-1.3B` (`kab_Latn` is a supported target).
Then filter — **all automated**:
1. **Round-trip**: kab → en, keep only if LaBSE cosine vs. the original English > 0.75.
2. **GlotLID**: output must classify as `kab_Latn`.
3. **Hunspell OOV rate**: below threshold.
4. Length-ratio sanity check.

Expect to keep 20–40%. That is fine — precision beats volume here.

### 5.3 Capability preservation (~20%)
Keep English and French instruction data in the mix. Also include **cross-lingual** items
(French question → Kabyle answer, and vice versa), because that is how your users will actually
talk to it.

### Training
LoRA `r=64`, LR 1e-5 to 2e-5, 2–3 epochs, mask the loss on prompt tokens, chat template applied.
~2–4 GPU-hours. Define the system prompt in Kabyle.

### 5.4 Optional — preference tuning (DPO)
Only after SFT works. Build pairs automatically: **chosen** = naturally-occurring Kabyle,
**rejected** = the NLLB back-translation of the same content. This teaches the model to prefer
idiomatic Kabyle over translationese — precisely the axis you care about. ~1 GPU-hour with LoRA.

---

## 7. Phase 6 — Evaluation

You currently have none. Build this in Phase 2, not at the end.

**Automated (run every checkpoint):**
| Metric | Tool | Measures |
|---|---|---|
| Tokenizer fertility | tokens/word | efficiency |
| Held-out perplexity, **per domain** | — | language fit; catches domain overfit |
| chrF++ on `boffire/tatoeba-en-kab` test | `sacrebleu` | **chrF++ not BLEU** — morphologically rich language |
| GlotLID pass-rate on generations | `cis-lmu/GlotLID` | % of output that is actually Kabyle |
| **Hunspell OOV rate on generations** | `boffire/hunspell-kab` | orthographic validity — your best cheap proxy for "are these real words" |
| English MMLU vs. base | `lm-eval-harness` | catastrophic forgetting |

**Manual (your advantage):** 50 fixed prompts covering chat, factual Q&A, writing help, and
translation. Score them yourself after each phase. As a native speaker this takes 30 minutes
and tells you more than every automated metric combined.

---

## 8. Phase 7 — Ship

Merge LoRA → `safetensors` → convert to **GGUF** (`llama.cpp`). A 1.7B model at Q4_K_M is ~1.1GB
and runs on a laptop CPU. Publish to HuggingFace with the corpus, tokenizer, and eval numbers.

Given that community datasets are what made this project viable, publishing your normalized
merged corpus back is worth more to Kabyle NLP than the model itself.

---

## 9. Compute budget

| Phase | GPU-hours | Notes |
|---|---|---|
| 1. Normalization | 0 | CPU |
| 2. Corpus aggregation | 0–2 | GlotLID filtering is GPU-accelerable |
| 3. Tokenizer | 0 | CPU |
| 4. CPT (1.7B) | ~16 | 2 sessions |
| 5. Translate + filter SFT data | ~4 | NLLB inference |
| 6. SFT | ~3 | |
| 7. DPO (optional) | ~1 | |
| 8. Eval | ~2 | |
| **Total** | **~28h** | **one week of Kaggle quota** |

The compute is not the bottleneck. **Phases 1–2 are.** Budget your time accordingly: the data
work is ~70% of this project and determines the outcome.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Catastrophic forgetting | 40% replay; MMLU gate after CPT; raise replay if it drops |
| Translationese in outputs | corpus-grounded SFT majority; round-trip+LaBSE filtering; DPO |
| Community data quality unknown | sample 200 sentences per source and read them yourself |
| NLLB en-kab bitext is LASER-mined and noisy | GlotLID + length-ratio + LaBSE filter; expect to discard most |
| Tatoeba domain skew (short, "Tom", translated-from-English) | cap its share of the mix; it is ~750k of your sentences and will dominate if unweighted |
| fp16 instability on T4 | `GradScaler`, grad clipping, LR warmup; watch for NaN |
| Kaggle 12h session cap | hub-checkpoint resume built in Phase 4, before the first long run |

---

## 11. Status

Phases 1–3 are **built and verified**; 4–8 are **built, untested on a GPU** (no accelerator
in the development environment).

| # | Item | Status |
|---|---|---|
| 1 | Orthographic normalizer + confusable folding | done, 40 tests pass |
| 1 | Per-font legacy PDF decoder | done — both books decode, OOV 30%→20% and 27%→17% |
| 1 | Automatic cipher solver for unknown fonts | done — rediscovers the hand-derived map 6/7 |
| 2 | Source registry (Tiers A–D) | done |
| 2 | Quality filters + GlotLID + MinHash dedup | done — vectorized, 8 min per 2M docs |
| 2 | Corpus builder with stratified splits | done — verified on local data |
| 3 | Fertility benchmark | **done and run** — see §3.1 |
| 3 | Vocabulary extension with mean-init | built, needs a GPU run |
| 4 | CPT trainer (LoRA + embeddings, replay, resume) | built, needs a GPU run |
| 5 | SFT data builder | done — task builders verified on real corpus |
| 6 | SFT trainer | built, needs a GPU run |
| 7 | Eval harness | built, needs a GPU run |
| 8 | GGUF export | built |

### Next actions, in order

1. **`python scripts/02_build_corpus.py --audit`** — read the samples. You are the only
   Kabyle speaker in this pipeline and the community datasets are unverified uploads.
   Twenty minutes here beats any automated filter.
2. **`python scripts/02_build_corpus.py --out data/corpus`** — full pull. **Report the
   final token count.** It determines model size, epochs, and replay ratio, and every
   later decision is downstream of it.
3. Re-run the fertility benchmark on the real corpus (the numbers in §3.1 came from the
   local v0 files only).
4. Then, and only then, start Phase 4 on Kaggle.
