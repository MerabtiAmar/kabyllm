#!/usr/bin/env python
"""Phase 5: build the Kabyle instruction dataset.

    python scripts/05_build_sft.py --corpus data/corpus --out data/sft
    python scripts/05_build_sft.py --corpus data/corpus --out data/sft --translate

Without `--translate` the whole set is corpus-grounded: every response is text a
Kabyle speaker actually wrote. That alone produces a usable assistant and costs no GPU
time. `--translate` adds machine-translated instructions through the round-trip
filter, which broadens task coverage at the cost of some translationese.

Read a sample before training on it (`--preview`). Automated filters cannot judge
whether Kabyle sounds right; you can.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import console, lexicon, sft, sources

log = logging.getLogger("sft")


def load_docs(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            out.append(json.loads(line))
    return out


def load_pairs(limit: int) -> dict[str, list[tuple[str, str]]]:
    """Verified parallel pairs, by other-language. Noisy NLLB bitext is excluded."""
    from datasets import load_dataset

    from kabyl.filters import PairConfig, pair_quality
    from kabyl.normalize import normalize

    cfg = PairConfig()
    got: dict[str, list[tuple[str, str]]] = {"taglizit": [], "tafransist": []}
    for src in sources.PARALLEL:
        if src.trust == "noisy":
            continue
        try:
            ds = load_dataset(src.repo, split=src.split)
        except Exception as exc:
            log.warning("%s unavailable: %s", src.repo, str(exc)[:100])
            continue
        cols = ds.column_names
        en = next((c for c in ("en", "english", "eng", "source") if c in cols), None)
        kab = next((c for c in ("kab", "kabyle", "target", "kab_Latn") if c in cols), None)
        if not (en and kab):
            log.warning("%s: no en/kab columns in %s", src.repo, cols)
            continue
        kept = 0
        for row in ds:
            if kept >= limit:
                break
            s, t = str(row[en] or ""), normalize(str(row[kab] or ""))
            if pair_quality(s, t, cfg):
                got["taglizit"].append((s, t))
                kept += 1
        log.info("%-40s %d verified pairs", src.repo, kept)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--translate", action="store_true",
                    help="add NLLB-translated instructions (needs a GPU)")
    ap.add_argument("--translate-n", type=int, default=20_000)
    ap.add_argument("--pairs-per-source", type=int, default=40_000)
    ap.add_argument("--replay-frac", type=float, default=0.2,
                    help="share of English/French instruction data to retain")
    ap.add_argument("--preview", type=int, default=6, help="examples to print per task")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    console.setup()
    rng = random.Random(args.seed)

    corpus = Path(args.corpus)
    train = corpus / "train.jsonl"
    if not train.exists():
        log.error("missing %s -- run scripts/02_build_corpus.py first", train)
        return 1

    docs = load_docs(train)
    log.info("loaded %d corpus documents", len(docs))
    lex = lexicon.load()

    examples: list[sft.Example] = []

    log.info("building corpus-grounded tasks...")
    examples += sft.build_qa(docs, rng)
    examples += sft.build_summarize(docs, rng)
    examples += sft.build_continue(docs, rng)
    examples += sft.build_spelling(docs, rng, lex)

    pairs = load_pairs(args.pairs_per_source)
    for lang, plist in pairs.items():
        if plist:
            rng.shuffle(plist)
            examples += sft.build_translation(plist, rng, other_lang=lang)

    try:
        from datasets import load_dataset

        verbs = load_dataset("boffire/kabyle-verbs", split="train")
        examples += sft.build_morphology(list(verbs), rng)
    except Exception as exc:
        log.warning("verb paradigms unavailable: %s", str(exc)[:100])

    if args.translate:
        log.info("translating instructions with NLLB (this needs a GPU)...")
        examples += translate_instructions(args.translate_n, lex, rng)

    replay = load_replay_instructions(
        int(len(examples) * args.replay_frac / max(1e-9, 1 - args.replay_frac))
    )
    examples += replay

    rng.shuffle(examples)
    print("\n" + sft.mix_report(examples))

    if args.preview:
        seen: set[str] = set()
        print("\n" + "=" * 78 + "\nSAMPLES - read these before training\n" + "=" * 78)
        for ex in examples:
            if ex.task in seen:
                continue
            seen.add(ex.task)
            print(f"\n[{ex.task}] ({ex.lang})")
            print(f"  USER: {ex.instruction[:220]}")
            print(f"  ASST: {ex.response[:220]}")
            if len(seen) >= args.preview:
                break

    n_val = min(2_000, len(examples) // 20)
    sft.write_jsonl(examples[n_val:], Path(args.out) / "train.jsonl")
    sft.write_jsonl(examples[:n_val], Path(args.out) / "val.jsonl")
    return 0


def load_replay_instructions(n: int) -> list[sft.Example]:
    """English/French instruction data, so instruction-following itself survives SFT."""
    from datasets import load_dataset

    out: list[sft.Example] = []
    for repo, lang in [("HuggingFaceH4/no_robots", "en"),
                       ("CohereForAI/aya_dataset", "fr")]:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
        except Exception as exc:
            log.warning("replay %s unavailable: %s", repo, str(exc)[:90])
            continue
        for row in ds:
            if len(out) >= n:
                break
            inst = row.get("prompt") or row.get("inputs") or ""
            resp = row.get("completion") or row.get("targets") or ""
            if isinstance(row.get("messages"), list) and len(row["messages"]) >= 2:
                inst = row["messages"][0].get("content", "")
                resp = row["messages"][1].get("content", "")
            if inst and resp:
                out.append(sft.Example(str(inst), str(resp), "replay", repo, lang))
    log.info("capability replay: %d examples", len(out))
    return out


def translate_instructions(n: int, lexicon_set: set[str], rng: random.Random) -> list[sft.Example]:
    """Translate English instructions to Kabyle, keeping only round-trip survivors."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "facebook/nllb-200-distilled-1.3B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    def translate(texts: list[str], src: str, tgt: str, bs: int = 16) -> list[str]:
        tok.src_lang = src
        out: list[str] = []
        for i in range(0, len(texts), bs):
            batch = tok(texts[i:i + bs], return_tensors="pt",
                        padding=True, truncation=True, max_length=256)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                gen = model.generate(
                    **batch, max_new_tokens=256,
                    forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                )
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [r for r in ds if not r.get("context")][:n]
    prompts = [r["instruction"] for r in rows]
    answers = [r["response"] for r in rows]

    log.info("translating %d instruction/response pairs en->kab", len(rows))
    kab_prompts = translate(prompts, "eng_Latn", "kab_Latn")
    kab_answers = translate(answers, "eng_Latn", "kab_Latn")
    log.info("back-translating for the round-trip check")
    back = translate(kab_answers, "kab_Latn", "eng_Latn")

    keep = sft.TranslationFilter().keep(answers, back, kab_answers, lexicon_set)
    return [
        sft.Example(p, a, "translated", "dolly")
        for p, a, k in zip(kab_prompts, kab_answers, keep) if k
    ]


if __name__ == "__main__":
    raise SystemExit(main())
