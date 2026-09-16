"""Canonical Kabyle Latin alphabet and the confusables that must fold into it.

Every table here is keyed by *what the character should become*. The single source of
truth for "is this text clean" is `CANONICAL`; anything outside it that survives
normalization is a bug in the pipeline or a genuinely foreign character.

Codepoints are spelled out because several of these are visually identical to each
other and to ASCII -- reading them off the screen is not reliable.
"""

import unicodedata

# --- The target alphabet (INALCO / HCA standard Kabyle Latin) ------------------

CONSONANTS = "bcčdḍfgǧɣhḥjklmnpqrṛsṣtṭvwxyzẓ"
VOWELS = "aeiou"
LOWER = "".join(sorted(set(CONSONANTS + VOWELS + "ɛ")))
UPPER = "".join(sorted({c.upper() for c in LOWER}))

#: Letters that legitimately appear in standard Kabyle text.
CANONICAL_LETTERS = frozenset(LOWER) | frozenset(UPPER)

#: Punctuation and digits we keep. Everything else is stripped or folded.
CANONICAL_PUNCT = frozenset(" \n\t0123456789.,;:!?'\"()[]{}-/%&+=*@#_<>«»")

CANONICAL = CANONICAL_LETTERS | CANONICAL_PUNCT


# --- Confusable folding -------------------------------------------------------
# Kabyle's four non-ASCII letters (ɛ ɣ and the emphatics) get spelled a dozen
# different ways in the wild: Greek, Cyrillic, IPA, and Latin-1 lookalikes. Each
# variant silently forks the vocabulary, so they all collapse here.

CONFUSABLES: dict[str, str] = {}


def _fold(target: str, *sources: str) -> None:
    for s in sources:
        assert s != target, f"{target!r} maps to itself"
        CONFUSABLES[s] = target


# ayin / pharyngeal -- ɛ U+025B, Ɛ U+0190
_fold(
    "ɛ",  # ɛ
    "ε",  # ε GREEK SMALL EPSILON
    "ϵ",  # ϵ GREEK LUNATE EPSILON SYMBOL
    "ԑ",  # ԑ CYRILLIC SMALL REVERSED ZE
    "ʕ",  # ʕ LATIN LETTER PHARYNGEAL VOICED FRICATIVE
    "ƹ",  # ƹ LATIN SMALL EZH REVERSED
    "ᵋ",  # ᵋ MODIFIER SMALL OPEN E
    "ⱸ",  # ⱸ LATIN SMALL E WITH NOTCH
)
_fold(
    "Ɛ",  # Ɛ
    "Ε",  # Ε GREEK CAPITAL EPSILON
    "Ԑ",  # Ԑ CYRILLIC CAPITAL REVERSED ZE
    "Ƹ",  # Ƹ LATIN CAPITAL EZH REVERSED
    "∑",  # ∑ N-ARY SUMMATION (legacy Berber fonts)
)

# voiced velar fricative -- ɣ U+0263, Ɣ U+0194
_fold(
    "ɣ",  # ɣ
    "γ",  # γ GREEK SMALL GAMMA
    "ɤ",  # ɤ LATIN SMALL RAMS HORN
    "ˠ",  # ˠ MODIFIER SMALL GAMMA
    "ᵞ",  # ᵞ MODIFIER SMALL GREEK GAMMA
    "ℽ",  # ℽ DOUBLE-STRUCK SMALL GAMMA
)
_fold(
    "Ɣ",  # Ɣ
    "Γ",  # Γ GREEK CAPITAL GAMMA
    "ℾ",  # ℾ DOUBLE-STRUCK CAPITAL GAMMA
)

# pharyngeal fricative -- ḥ U+1E25, Ḥ U+1E24
_fold("ḥ", "ħ", "ѻ")  # ħ, ѻ
_fold("Ḥ", "Ħ")  # Ħ

# emphatics: s-cedilla / s-comma and t-cedilla / t-comma are used for ṣ / ṭ in
# several Berber conventions and are visually identical to the Romanian letters.
_fold("ṣ", "ş", "ș")  # ṣ <- ş ș
_fold("Ṣ", "Ş", "Ș")  # Ṣ <- Ş Ș
_fold("ṭ", "ţ", "ț")  # ṭ <- ţ ț
_fold("Ṭ", "Ţ", "Ț")  # Ṭ <- Ţ Ț

# c-hacek / g-hacek written with other diacritics
_fold("č", "ć")  # č <- ć
_fold("Č", "Ć")  # Č <- Ć
_fold("ǧ", "ǵ", "ğ")  # ǧ <- ǵ ğ
_fold("Ǧ", "Ǵ", "Ğ")  # Ǧ <- Ǵ Ğ


# --- Punctuation and whitespace ----------------------------------------------

PUNCT_MAP: dict[str, str] = {
    # quotes
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"', "´": "'", "`": "'",
    # dashes
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-", "­": "",
    # spaces
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", "　": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    # invisibles
    "​": "", "‌": "", "‍": "", "﻿": "",
    "⁠": "", "᠎": "",
    # misc
    "…": "...", "·": ".", "•": "-",
    "ʼ": "'", "ʻ": "'", "‹": "<", "›": ">",
}

#: PDF text extraction routinely emits these; they wreck tokenization.
LIGATURES: dict[str, str] = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE",
}


def describe(ch: str) -> str:
    """Human-readable identity of a character, for audit reports."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "<unnamed>"
    return f"U+{ord(ch):04X} {name}"
