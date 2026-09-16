#!/usr/bin/env python
"""Phase 1: decode PDFs into clean Kabyle text, recovering legacy font encodings.

    python scripts/01_decode_pdfs.py --in . --out data/interim
    python scripts/01_decode_pdfs.py --in pdfs/ --out data/interim --solve

`--solve` runs the cipher search on fonts that are not in the registry, which is what
makes the ~600 community PDFs usable. Solved maps are written to
`data/interim/discovered_fonts.json` for review; add the good ones to
`kabyl/fontmaps.py` so future runs skip the search.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from kabyl import console  # noqa: F401  (UTF-8 stdout)
from kabyl import cipher, fontmaps, lexicon
from kabyl.pdf_decode import _iter_spans, decode_pdf

log = logging.getLogger("decode")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default=".", help="directory of PDFs")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--solve", action="store_true",
                    help="search for maps of unregistered fonts")
    ap.add_argument("--reference", default="kabyle_wiki_corpus_cleaned.txt",
                    help="known-good Kabyle used for frequency analysis")
    ap.add_argument("--min-oov-gain", type=float, default=0.03,
                    help="minimum OOV reduction before a solved map is accepted")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    console.setup(logging.DEBUG if args.verbose else logging.INFO)

    pdfs = sorted(Path(args.src).glob("*.pdf"))
    if not pdfs:
        log.error("no PDFs found in %s", args.src)
        return 1

    lex = lexicon.load([args.reference] if Path(args.reference).exists() else None)
    if not lex:
        log.error("no lexicon available; cannot score decodings")
        return 1
    log.info("lexicon: %d entries", len(lex))

    profile = {}
    if Path(args.reference).exists():
        profile = cipher.reference_profile(
            Path(args.reference).read_text(encoding="utf-8")[:2_000_000]
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    discovered: dict[str, dict[str, str]] = {}

    for pdf in pdfs:
        log.info("\n=== %s", pdf.name)

        if args.solve and profile:
            doc = fitz.open(pdf)
            spans = [(f, t) for _, f, t in _iter_spans(doc)]
            doc.close()
            unknown = [
                (f, t) for f, t in spans
                if f and fontmaps.map_for_font(f) is None
            ]
            if unknown:
                for font, sol in cipher.solve_per_font(unknown, lex, profile).items():
                    gain = sol.oov_before - sol.oov_after
                    if gain < args.min_oov_gain:
                        log.info("  %-32s rejected (gain %.1f%% too small)",
                                 font, gain * 100)
                        continue
                    log.info("  %-32s %s", font, sol.summary().splitlines()[0])
                    log.info("  %-32s %s", "", sol.summary().splitlines()[1].strip())
                    fontmaps.register(font, sol.table)
                    discovered[font] = sol.table

        text, rep = decode_pdf(pdf, lex)
        print(rep.summary())

        dest = out_dir / f"{pdf.stem}.txt"
        dest.write_text(text, encoding="utf-8")
        log.info("  -> %s (%d chars)", dest, len(text))

    if discovered:
        dpath = out_dir / "discovered_fonts.json"
        dpath.write_text(json.dumps(discovered, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        log.info("\n%d font maps discovered -> %s", len(discovered), dpath)
        log.info("Review these, then paste the good ones into kabyl/fontmaps.py.")
        log.info("A solved map is only as good as the lexicon that scored it: check a "
                 "few decoded pages by eye before trusting a book to training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
