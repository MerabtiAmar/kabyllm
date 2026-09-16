"""Document-level quality filtering for the Kabyle corpus.

Roughly half the raw material is unusable: CommonCrawl boilerplate, French and
Arabic that slipped past whoever assembled the dataset, other Berber varieties,
machine-translation output, and PDF extraction wreckage. None of it is labelled.

Every filter returns a `Verdict` rather than a bool so rejections stay auditable --
silently dropping 60% of a corpus and reporting only the survivors is how you end up
unable to explain your own dataset.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from .normalize import adhoc_score, diacritic_rate, normalize, words

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


ACCEPT = Verdict(True)


@dataclass
class FilterConfig:
    min_words: int = 5
    max_words: int = 20_000
    #: Kabyle-specific letters as a share of all letters. Real Kabyle sits around
    #: 0.05-0.15; near-zero means French/English wearing a Kabyle label.
    min_diacritic_rate: float = 0.015
    #: Share of words showing French/Arabizi spelling. Above this the document is in
    #: a non-standard orthography and gets quarantined rather than converted.
    max_adhoc_score: float = 0.25
    #: Share of tokens absent from the Kabyle lexicon.
    max_oov: float = 0.55
    max_digit_ratio: float = 0.20
    max_punct_ratio: float = 0.25
    min_alpha_ratio: float = 0.55
    #: Share of lines that are exact duplicates of another line in the same document
    #: -- the signature of navigation menus and templated pages.
    max_repeat_line_ratio: float = 0.30
    #: Longest run of one character (PDF rule lines, "-----", "......").
    max_char_run: int = 12
    min_mean_word_len: float = 2.5
    max_mean_word_len: float = 12.0


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    kept_words: int = 0

    def record(self, v: Verdict, n_words: int = 0) -> None:
        self.seen += 1
        if v.ok:
            self.kept += 1
            self.kept_words += n_words
        else:
            self.rejected[v.reason] += 1

    def summary(self) -> str:
        pct = 100 * self.kept / self.seen if self.seen else 0
        lines = [f"kept {self.kept:,}/{self.seen:,} ({pct:.1f}%)  {self.kept_words:,} words"]
        for reason, n in self.rejected.most_common():
            lines.append(f"    -{n:>9,}  {reason}")
        return "\n".join(lines)


_RE_RUN = re.compile(r"(.)\1{11,}")


def quality(text: str, cfg: FilterConfig, lexicon: set[str] | None = None) -> Verdict:
    """Apply every cheap structural check. Language ID is handled separately."""
    if not text or not text.strip():
        return Verdict(False, "empty")

    toks = words(text)
    n = len(toks)
    if n < cfg.min_words:
        return Verdict(False, "too_short")
    if n > cfg.max_words:
        return Verdict(False, "too_long")

    mean_len = sum(len(w) for w in toks) / n
    if mean_len < cfg.min_mean_word_len:
        return Verdict(False, "words_too_short")
    if mean_len > cfg.max_mean_word_len:
        return Verdict(False, "words_too_long")

    total = len(text)
    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    punct = sum((not c.isalnum()) and (not c.isspace()) for c in text)
    if alpha / total < cfg.min_alpha_ratio:
        return Verdict(False, "low_alpha")
    if digits / total > cfg.max_digit_ratio:
        return Verdict(False, "digit_heavy")
    if punct / total > cfg.max_punct_ratio:
        return Verdict(False, "punct_heavy")

    if _RE_RUN.search(text):
        return Verdict(False, "char_run")

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 4:
        uniq = len(set(lines))
        if 1 - uniq / len(lines) > cfg.max_repeat_line_ratio:
            return Verdict(False, "repeated_lines")

    if diacritic_rate(text) < cfg.min_diacritic_rate:
        return Verdict(False, "not_kabyle_charset")

    if adhoc_score(text) > cfg.max_adhoc_score:
        return Verdict(False, "adhoc_orthography")

    if lexicon:
        low = [w.lower() for w in toks]
        oov = sum(1 for w in low if w not in lexicon) / len(low)
        if oov > cfg.max_oov:
            return Verdict(False, "high_oov")

    return ACCEPT


# --- Language identification --------------------------------------------------

class LanguageID:
    """GlotLID wrapper. Kabyle is `kab_Latn`.

    Non-negotiable for GlotCC and the NLLB bitext, which contain large amounts of
    Arabic, French, and neighbouring Berber varieties (Tachelhit, Tarifit, Chaoui)
    that a charset check alone will not catch -- they use the same alphabet.
    """

    def __init__(self, repo: str = "cis-lmu/glotlid", threshold: float = 0.6):
        self.threshold = threshold
        self.model = None
        try:
            import fasttext
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo, filename="model.bin")
            self.model = fasttext.load_model(path)
        except Exception as exc:
            log.warning(
                "GlotLID unavailable (%s).\n"
                "  Falling back to charset heuristics. These catch French, English and "
                "Arabic,\n  but CANNOT separate Kabyle from Tachelhit, Tarifit or "
                "Chaoui -- they share the\n  alphabet. Acceptable for the curated "
                "sources; NOT acceptable for glotcc or\n  nllb_en, which are "
                "web-mined. Run the full build on Linux (Kaggle) where\n"
                "  `pip install fasttext` works, or pass --only to exclude the "
                "web-mined sources.",
                str(exc)[:80],
            )

    def predict(self, text: str) -> tuple[str, float]:
        if self.model is None:
            return ("kab_Latn" if diacritic_rate(text) > 0.02 else "unk", 0.0)
        labels, probs = self.model.predict(text.replace("\n", " ")[:2000], k=1)
        return labels[0].removeprefix("__label__"), float(probs[0])

    def verdict(self, text: str) -> Verdict:
        lang, conf = self.predict(text)
        if not lang.startswith("kab"):
            return Verdict(False, f"lang_{lang}")
        if self.model is not None and conf < self.threshold:
            return Verdict(False, "lang_low_conf")
        return ACCEPT


# --- Parallel-pair filtering --------------------------------------------------

@dataclass
class PairConfig:
    min_words: int = 3
    max_words: int = 200
    #: Ratio of source to target length. LASER-mined bitext is full of pairs where
    #: one side is a fragment of the other.
    max_len_ratio: float = 2.5
    #: Fraction of the target that is identical to the source -- untranslated rows.
    max_copy_ratio: float = 0.6


def pair_quality(src: str, tgt: str, cfg: PairConfig) -> Verdict:
    """Structural checks for a translation pair, before any model-based scoring."""
    s, t = words(src), words(tgt)
    if not s or not t:
        return Verdict(False, "empty_side")
    if not (cfg.min_words <= len(t) <= cfg.max_words):
        return Verdict(False, "target_length")
    ratio = max(len(s), len(t)) / max(1, min(len(s), len(t)))
    if ratio > cfg.max_len_ratio:
        return Verdict(False, "length_mismatch")

    overlap = len(set(w.lower() for w in s) & set(w.lower() for w in t))
    if overlap / len(set(t)) > cfg.max_copy_ratio:
        return Verdict(False, "untranslated")

    if diacritic_rate(tgt) < 0.01:
        return Verdict(False, "target_not_kabyle")
    return ACCEPT


def clean_document(text: str, cfg: FilterConfig, lexicon: set[str] | None = None
                   ) -> tuple[str, Verdict]:
    """Normalize then judge. Returns the normalized text regardless of verdict so
    rejected documents can still be inspected."""
    norm = normalize(text)
    return norm, quality(norm, cfg, lexicon)
