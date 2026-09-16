#!/usr/bin/env python
"""Phase 2: build the Kabyle training corpus from every registered source.

    python scripts/02_build_corpus.py --out data/corpus
    python scripts/02_build_corpus.py --out data/corpus --limit 2000   # smoke test
    python scripts/02_build_corpus.py --audit                          # sample & inspect

The final token count this prints is the number that decides everything downstream:
model size, epoch count, and replay ratio. Do not start a training run without it.

`--audit` prints raw samples per source without building anything. Run it first. The
community datasets are unverified uploads and you are the only Kabyle speaker in this
pipeline -- twenty minutes of reading here is worth more than any automated filter.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import console  # noqa: F401  (UTF-8 stdout)
from kabyl import lexicon, sources
from kabyl.collect import collect, write_splits
from kabyl.filters import FilterConfig
from kabyl.normalize import words

log = logging.getLogger("corpus")


def audit(keys: list[str] | None, n: int = 5) -> None:
    """Print raw samples per source so a native speaker can judge them."""
    from kabyl.collect import load_source

    pool = [s for s in sources.ALL if s.kind is not sources.Kind.PDF]
    if keys:
        pool = [s for s in pool if s.key in keys]

    for src in pool:
        print(f"\n{'=' * 78}\n{src.key}  [{src.kind.value}]  trust={src.trust}  "
              f"weight={src.weight}\n{src.repo}")
        if src.notes:
            print(f"  note: {src.notes}")
        print("=" * 78)
        raw = load_source(src, limit=500)
        if not raw:
            print("  (unavailable)")
            continue
        for t in random.Random(0).sample(raw, min(n, len(raw))):
            snippet = " ".join(t.split())[:280]
            print(f"  - {snippet}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--limit", type=int, default=None,
                    help="rows per source; use for a fast smoke test")
    ap.add_argument("--audit", action="store_true", help="sample sources and exit")
    ap.add_argument("--only", nargs="*", help="restrict to these source keys")
    ap.add_argument("--no-lid", action="store_true",
                    help="skip GlotLID (much faster, much less safe)")
    ap.add_argument("--no-local", action="store_true", help="skip local v0 files")
    ap.add_argument("--dedup-threshold", type=float, default=0.8)
    ap.add_argument("--max-oov", type=float, default=0.55)
    args = ap.parse_args()

    console.setup()

    if args.audit:
        audit(args.only)
        return 0

    lex = lexicon.load(["kabyle_wiki_corpus_cleaned.txt", "kab_Latn.txt"])
    log.info("lexicon: %d entries\n", len(lex))

    pool = [s for s in sources.MONO if s.kind is not sources.Kind.PDF]
    pool += sources.PARALLEL  # Kabyle side only; pairs are handled in Phase 5
    if args.only:
        pool = [s for s in pool if s.key in args.only]

    cfg = FilterConfig(max_oov=args.max_oov)
    docs, rep = collect(
        pool,
        lexicon=lex,
        cfg=cfg,
        use_lid=not args.no_lid,
        limit_per_source=args.limit,
        local_files=None if args.no_local else sources.LOCAL_MONO,
        dedup_threshold=args.dedup_threshold,
    )

    print("\n" + rep.table())

    out = Path(args.out)
    counts = write_splits(docs, out)

    # Apply the sampling weights as a materialized training list. Doing it here
    # rather than in the dataloader keeps the mix reproducible and inspectable.
    weights = {s.key: s.weight for s in sources.ALL}
    train_path = out / "train.jsonl"
    weighted = out / "train_weighted.jsonl"
    rng = random.Random(1337)
    n_written = 0
    with train_path.open(encoding="utf-8") as fin, weighted.open("w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            w = weights.get(d["source"], 1.0)
            reps = int(w) + (1 if rng.random() < (w - int(w)) else 0)
            for _ in range(reps):
                fout.write(line)
                n_written += 1

    total_words = rep.total_words
    # Measured: Qwen3 encodes this corpus at 2.75 tokens/word before vocabulary
    # extension. NLLB-200, which actually pretrained on Kabyle, reaches 2.27 -- so
    # ~1.9 is the realistic post-extension figure, not the 1.6 originally guessed.
    fert_pre, fert_post = 2.75, 1.9
    bar = "=" * 60
    print(
        f"\n{bar}\nCORPUS SUMMARY\n{bar}\n"
        f"documents        {rep.total_docs:12,}\n"
        f"words            {total_words:12,}\n"
        f"dedup removed    {rep.dedup_removed:12,}\n"
        f"splits           {counts}\n"
        f"weighted train   {n_written:12,} docs -> {weighted.name}\n\n"
        f"tokens (measured, pre-extension  @ {fert_pre} tok/word) "
        f"{int(total_words * fert_pre):12,}\n"
        f"tokens (projected, post-extension @ {fert_post} tok/word) "
        f"{int(total_words * fert_post):12,}\n"
        f"v0 corpus was 995,159 tokens -> "
        f"{total_words * fert_post / 995_159:.0f}x more\n"
        f"{bar}"
    )

    from kabyl.training import budget_report
    print("\nCompute budget (Qwen3-1.7B, 2 epochs, 40% replay):")
    kab_tokens = total_words * fert_post
    print(budget_report(1.7e9, kab_tokens / 0.6, epochs=2.0))

    (out / "report.json").write_text(
        json.dumps({
            "per_source": rep.per_source,
            "total_docs": rep.total_docs,
            "total_words": rep.total_words,
            "dedup_removed": rep.dedup_removed,
            "splits": counts,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
