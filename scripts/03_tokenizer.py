#!/usr/bin/env python
"""Phase 3: measure tokenizer fertility on Kabyle, then build the extended vocabulary.

    python scripts/03_tokenizer.py --bench
    python scripts/03_tokenizer.py --extend --base Qwen/Qwen3-1.7B-Base --new 8000

`--bench` is the decision point for the whole project: it says how many tokens each
candidate base model spends per Kabyle word, which sets both the compute cost of
continued pretraining and how much text fits in context. Run it before committing to
a base model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import console  # noqa: F401  (UTF-8 stdout)
from kabyl import tokenizer as tk

log = logging.getLogger("tokenizer")


def load_sample(corpus: str, max_chars: int = 400_000) -> str:
    p = Path(corpus)
    if p.suffix == ".jsonl":
        parts, n = [], 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                parts.append(t)
                n += len(t)
                if n >= max_chars:
                    break
        return "\n".join(parts)
    return p.read_text(encoding="utf-8")[:max_chars]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/train.jsonl")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--models", nargs="*", help="override the candidate list")
    ap.add_argument("--new", type=int, default=8_000, help="new tokens to add")
    ap.add_argument("--bpe-vocab", type=int, default=12_000)
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()

    console.setup()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        log.error("corpus not found: %s -- run scripts/02_build_corpus.py first",
                  corpus_path)
        return 1
    text = load_sample(args.corpus)
    log.info("sample: %d chars, %d words\n", len(text), len(text.split()))

    if args.bench:
        print(f"{'model':<34} {'fertility':>9}  (English byte-BPE is ~"
              f"{tk.ENGLISH_BASELINE} tok/word)")
        print("-" * 104)
        results = tk.benchmark(text, args.models)
        tk.save_report(results, Path(args.out) / "fertility.json")
        usable = [r for r in results if r.name not in tk.REFERENCE_ONLY]
        refs = [r for r in results if r.name in tk.REFERENCE_ONLY]
        if usable:
            best = usable[0]
            print("-" * 104)
            print(f"\nlowest fertility: {best.name} at {best.tokens_per_word:.2f} tok/word")
            print(f"cost vs English:  {best.tokens_per_word / tk.ENGLISH_BASELINE:.1f}x "
                  f"more tokens for the same content")
            print(f"diacritic tax:    {best.diacritic_penalty:.2f}x per character on "
                  f"words containing ɣ/ɛ/ḍ/ṭ/ṣ/ẓ/ḥ")
            if refs:
                r = refs[0]
                print(f"\nreference: {r.name} (trained *with* kab_Latn) reaches "
                      f"{r.tokens_per_word:.2f} tok/word --")
                print("that is roughly the floor vocabulary extension can aim for.")
            print("\nThis is the pre-extension number. Target after extension is ~1.9 --")
            print("set by NLLB above, which pretrained on Kabyle and still needs 2.27.")
            print("Beating a model that saw the language is not the expectation.")

    if args.extend:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("\nloading %s", args.base)
        base_tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
        before = tk.fertility(base_tok, text, args.base)
        log.info("before: %s", before)

        docs = [json.loads(l)["text"] for l in
                corpus_path.open(encoding="utf-8")] if corpus_path.suffix == ".jsonl" \
            else [text]
        kab_tok = tk.train_kabyle_bpe(docs, args.bpe_vocab, Path(args.out) / "kab_bpe")
        new_tokens = tk.select_new_tokens(kab_tok, base_tok, args.new)

        model = AutoModelForCausalLM.from_pretrained(
            args.base, trust_remote_code=True, torch_dtype="auto"
        )
        model, base_tok = tk.extend(model, base_tok, new_tokens)

        after = tk.fertility(base_tok, text, args.base + " (extended)")
        log.info("after:  %s", after)
        gain = 1 - after.tokens_per_word / before.tokens_per_word
        log.info("\nfertility improved %.1f%% -> %.1f%% fewer tokens per word",
                 gain * 100, gain * 100)
        log.info("training FLOPs scale linearly with this, so it is also a "
                 "%.2fx compute saving", before.tokens_per_word / after.tokens_per_word)

        out = Path(args.out) / "extended"
        model.save_pretrained(out)
        base_tok.save_pretrained(out)
        log.info("saved extended model + tokenizer -> %s", out)

    if not (args.bench or args.extend):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
