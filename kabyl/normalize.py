"""Kabyle text normalization.

The v0 pipeline used a character *whitelist*, which kept Greek epsilon and Cyrillic
reversed-ze alongside real Latin ɛ and so forked the vocabulary three ways. A
whitelist can only reject; it cannot repair. This module repairs, then audits.

Order matters and is not arbitrary:

    NFC -> ligatures -> confusables -> punctuation -> de-hyphenate -> whitespace

NFC runs first so decomposed sequences (h + U+0323) become single codepoints before
the confusable table is consulted. Confusables run before punctuation so that a
character folded into ɛ is never subsequently mistaken for punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .alphabet import CANONICAL, CONFUSABLES, LIGATURES, PUNCT_MAP, describe

_TRANS_LIG = str.maketrans(LIGATURES)
_TRANS_CONF = str.maketrans(CONFUSABLES)
_TRANS_PUNCT = str.maketrans(PUNCT_MAP)

# A word broken across a PDF line: "tamur-\nt" -> "tamurt". The hyphen is dropped
# only when the continuation is *not* a clitic -- "bab-\nis" is the possessive
# "bab-is" and must keep its hyphen, not become "babis".
_RE_DEHYPHEN = re.compile(r"(\w)-[ \t]*\n[ \t]*([a-zɛɣḥḍṭṣẓṛčǧ]\w*)")

_RE_SPACES = re.compile(r"[ \t]+")
_RE_SPACE_NL = re.compile(r"[ \t]*\n[ \t]*")
_RE_BLANKLINES = re.compile(r"\n{3,}")
_RE_CTRL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Kabyle clitics attach with a hyphen and must survive as one unit. PDF extraction
# often inserts a space ("yenna- yas"), so the hyphen is re-tightened -- but only
# against an explicit clitic inventory. A general `\w - \w` rule would wrongly join
# "13 yennayer - 14 yennayer", which is a separator, not a clitic.
CLITICS = (
    # possessive
    "iw", "inu", "ik", "im", "is", "nneɣ", "nnes", "nwen", "nkent", "nsen", "nsent",
    # direct object
    "yi", "k", "kem", "t", "tt", "aɣ", "wen", "kent", "ten", "tent",
    # indirect object, plain and with the epenthetic y- that appears after a vowel
    # (yenna-yas, fka-yaɣ). Omitting these left "yenna- yas" untightened.
    "as", "asen", "asent", "ay",
    "yas", "yasen", "yasent", "yaɣ", "yawen", "yakent", "yak", "yam",
    # deictic particles
    "d", "id", "n", "ed", "in",
    # demonstratives
    "agi", "a", "nni", "ihin", "inna", "agini",
)
_RE_CLITIC = re.compile(
    r"(\w)\s*-\s+(" + "|".join(sorted(CLITICS, key=len, reverse=True)) + r")\b"
)


_CLITIC_SET = frozenset(CLITICS)


def _rejoin_break(m: re.Match[str]) -> str:
    """Rejoin a line-broken word, preserving the hyphen if it marks a clitic."""
    head, tail = m.group(1), m.group(2)
    sep = "-" if tail.split("-")[0].lower() in _CLITIC_SET else ""
    return f"{head}{sep}{tail}"


@dataclass
class NormalizeReport:
    """What the normalizer changed and what it could not explain."""

    chars_in: int = 0
    chars_out: int = 0
    folded: Counter = field(default_factory=Counter)
    residual: Counter = field(default_factory=Counter)

    @property
    def is_clean(self) -> bool:
        return not self.residual

    def summary(self, top: int = 25) -> str:
        lines = [f"chars {self.chars_in:,} -> {self.chars_out:,}"]
        if self.folded:
            lines.append(f"folded {sum(self.folded.values()):,} confusables:")
            for ch, n in self.folded.most_common(top):
                lines.append(f"    {n:>8,}  {describe(ch)} -> {CONFUSABLES.get(ch, '?')}")
        if self.residual:
            lines.append(f"UNEXPLAINED {sum(self.residual.values()):,} chars:")
            for ch, n in self.residual.most_common(top):
                lines.append(f"    {n:>8,}  {ch!r}  {describe(ch)}")
        else:
            lines.append("no residual characters - charset is canonical")
        return "\n".join(lines)


def normalize(text: str, *, dehyphenate: bool = True, report: NormalizeReport | None = None) -> str:
    """Fold `text` onto the canonical Kabyle charset.

    `dehyphenate` should be True for PDF-extracted text and False for text whose
    line breaks are meaningful (verse, dictionaries, sentence-per-line corpora).
    """
    if report is not None:
        report.chars_in += len(text)
        report.folded.update(c for c in text if c in CONFUSABLES)

    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TRANS_LIG)
    text = text.translate(_TRANS_CONF)
    text = text.translate(_TRANS_PUNCT)
    text = _RE_CTRL.sub("", text)

    if dehyphenate:
        text = _RE_DEHYPHEN.sub(_rejoin_break, text)

    text = _RE_CLITIC.sub(r"\1-\2", text)
    text = _RE_SPACES.sub(" ", text)
    text = _RE_SPACE_NL.sub("\n", text)
    text = _RE_BLANKLINES.sub("\n\n", text)
    text = unicodedata.normalize("NFC", text).strip()

    if report is not None:
        report.chars_out += len(text)
        report.residual.update(c for c in text if c not in CANONICAL)
    return text


def audit(text: str) -> Counter:
    """Characters outside the canonical set, with counts. Empty means clean."""
    return Counter(c for c in text if c not in CANONICAL)


def strip_residual(text: str, keep: str = "") -> str:
    """Drop any character still outside the canonical set.

    Last resort, applied after `normalize` and after inspecting the audit -- silently
    deleting characters you have not looked at hides data problems rather than
    fixing them.
    """
    allowed = CANONICAL | set(keep)
    return "".join(c for c in text if c in allowed)


# --- Orthography scoring ------------------------------------------------------

_RE_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)

#: Digraphs of French-influenced and Arabizi Kabyle spelling. Their presence means
#: the text is in an ad-hoc orthography and should be quarantined, not auto-converted
#: (the mappings are ambiguous and would corrupt correctly-spelled text).
ADHOC_MARKERS = ("gh", "kh", "dh", "th", "ch", "ou", "aa", "3", "7", "9")


def words(text: str) -> list[str]:
    return _RE_WORD.findall(text)


def adhoc_score(text: str) -> float:
    """Fraction of words showing ad-hoc/Arabizi orthography markers.

    High values mean French-style or chat-style spelling (``gher`` for ``ɣer``,
    ``3emmi`` for ``Ɛemmi``). Route those to a quarantine bucket.
    """
    toks = words(text.lower())
    if not toks:
        return 0.0
    hits = sum(1 for w in toks if any(m in w for m in ADHOC_MARKERS))
    return hits / len(toks)


def diacritic_rate(text: str) -> float:
    """Fraction of letters that are Kabyle-specific.

    Real Kabyle runs roughly 0.05-0.15. Near-zero on a long document usually means
    it is French or English that slipped through language ID, or Kabyle written in
    an ad-hoc ASCII orthography.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    special = sum(1 for c in letters if c in "ɛɣḥḍṭṣẓṛčǧƐƔḤḌṬṢẒṚČǦ")
    return special / len(letters)
