"""Extract Kabyle text from PDFs, decoding pre-Unicode Berber fonts span by span.

`page.get_text("text")` throws away the font, which is exactly the information
needed to decide which cipher applies. This module walks `get_text("rawdict")`
instead, so each run of characters is decoded with its own font's map.

For unknown fonts, `discover_map` searches the candidate maps and picks the one that
minimizes out-of-vocabulary rate against the Kabyle hunspell dictionary. That turns
font identification into an unattended search rather than manual cryptanalysis.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from . import fontmaps
from .normalize import normalize, words

log = logging.getLogger(__name__)

# Running headers/footers from the Feraoun scan and similar books: a bare page
# number, or the book title repeated on every page.
_RE_PAGENUM = re.compile(r"^\s*\d{1,4}\s*$")


@dataclass
class DecodeReport:
    path: str = ""
    pages: int = 0
    fonts: Counter = field(default_factory=Counter)
    unknown_fonts: Counter = field(default_factory=Counter)
    chosen: dict[str, str] = field(default_factory=dict)
    oov_before: float = 0.0
    oov_after: float = 0.0

    def summary(self) -> str:
        lines = [f"{self.path}  ({self.pages} pages)"]
        for font, n in self.fonts.most_common():
            pick = self.chosen.get(font, "-")
            lines.append(f"    {n:>8,} chars  {font:<40} map={pick}")
        if self.unknown_fonts:
            lines.append(f"    unknown fonts: {dict(self.unknown_fonts)}")
        lines.append(f"    hunspell OOV  {self.oov_before:.1%} -> {self.oov_after:.1%}")
        return "\n".join(lines)


def _iter_spans(doc: fitz.Document):
    """Yield (page_index, font_name, text) for every span in the document."""
    for pno, page in enumerate(doc):
        raw = page.get_text("rawdict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    chars = span.get("chars")
                    if chars:
                        text = "".join(c["c"] for c in chars)
                    else:
                        text = span.get("text", "")
                    if text:
                        yield pno, span.get("font", "<unknown>"), text
                yield pno, "", "\n"


def oov_rate(text: str, lexicon: set[str]) -> float:
    """Fraction of alphabetic tokens absent from `lexicon`. Lower is better."""
    toks = [w.lower() for w in words(text)]
    if not toks:
        return 1.0
    return sum(1 for w in toks if w not in lexicon) / len(toks)


def score_map(sample: str, table: dict[str, str], lexicon: set[str]) -> float:
    """OOV rate after applying `table` to `sample`. Lower is a better fit."""
    return oov_rate(normalize(fontmaps.apply_map(sample, table)), lexicon)


def discover_map(sample: str, lexicon: set[str]) -> tuple[str, dict[str, str], float]:
    """Pick the candidate map that best decodes `sample`.

    Returns (name, table, oov_rate). The identity map is always in the running, so a
    correctly-encoded Unicode PDF is left alone rather than being mangled by a cipher
    that does not apply to it.
    """
    scored = [
        (score_map(sample, table, lexicon), name, table)
        for name, table in fontmaps.CANDIDATE_MAPS.items()
    ]
    scored.sort(key=lambda t: t[0])
    best_oov, best_name, best_table = scored[0]
    return best_name, best_table, best_oov


def decode_pdf(
    path: str | Path,
    lexicon: set[str] | None = None,
    *,
    drop_headers: bool = True,
    auto_discover: bool = True,
) -> tuple[str, DecodeReport]:
    """Extract and decode a PDF into clean Kabyle text.

    `lexicon` enables automatic font-map discovery and OOV reporting. Without it,
    only fonts registered in `fontmaps.FONT_MAPS` are decoded.
    """
    path = Path(path)
    doc = fitz.open(path)
    rep = DecodeReport(path=path.name, pages=len(doc))

    spans = list(_iter_spans(doc))
    doc.close()

    for _, font, text in spans:
        if font:
            rep.fonts[font] += len(text)

    # Resolve one map per font, using a per-font sample for discovery.
    per_font: dict[str, dict[str, str]] = {}
    for font in rep.fonts:
        table = fontmaps.map_for_font(font)
        if table is not None:
            per_font[font] = table
            rep.chosen[font] = "registered"
            continue
        if not (auto_discover and lexicon):
            rep.unknown_fonts[font] += 1
            per_font[font] = {}
            rep.chosen[font] = "identity (no lexicon)"
            continue
        sample = "".join(t for _, f, t in spans if f == font)[:20_000]
        name, table, oov = discover_map(sample, lexicon)
        per_font[font] = table
        rep.chosen[font] = f"{name} (oov {oov:.1%})"
        if name == "identity":
            rep.unknown_fonts[font] += 1

    raw_parts, dec_parts = [], []
    for _, font, text in spans:
        raw_parts.append(text)
        dec_parts.append(fontmaps.apply_map(text, per_font.get(font, {})) if font else text)

    raw_text = "".join(raw_parts)
    text = normalize("".join(dec_parts))

    if drop_headers:
        text = _strip_running_headers(text)

    if lexicon:
        rep.oov_before = oov_rate(normalize(raw_text), lexicon)
        rep.oov_after = oov_rate(text, lexicon)

    return text, rep


def _strip_running_headers(text: str, min_repeats: int = 5) -> str:
    """Remove page numbers and lines repeated on most pages (running heads)."""
    lines = text.split("\n")
    counts = Counter(l.strip() for l in lines if 0 < len(l.strip()) < 60)
    boiler = {l for l, n in counts.items() if n >= min_repeats and len(l) < 60}
    out = [
        l for l in lines
        if not _RE_PAGENUM.match(l) and l.strip() not in boiler
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def decode_text_file(
    path: str | Path, lexicon: set[str] | None = None
) -> tuple[str, str, float]:
    """Decode an already-extracted .txt whose font information is gone.

    Falls back to the merged map since per-span fonts are unrecoverable. Use
    `decode_pdf` on the original PDF whenever it is available -- this is strictly
    worse and exists only for text you cannot re-extract.
    """
    text = Path(path).read_text(encoding="utf-8")
    if not lexicon:
        return normalize(fontmaps.apply_map(text, fontmaps.BERBER_TIMES_MERGED)), "merged", 0.0
    name, table, oov = discover_map(text[:50_000], lexicon)
    return normalize(fontmaps.apply_map(text, table)), name, oov
