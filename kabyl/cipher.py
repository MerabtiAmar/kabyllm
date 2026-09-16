"""Automatic recovery of unknown legacy Berber font encodings.

`fontmaps` covers the encodings already identified by hand. That does not scale to
the 600+ community PDFs (`boffire/ayamun-pdfs`, `boffire/bouira`,
`Imsidag-community/kabyle-adlis-pdfs`), each of which may use a different typesetter's
font.

These encodings are simple substitution ciphers over a known plaintext language, so
they can be solved rather than catalogued. The objective is out-of-vocabulary rate
against a Kabyle lexicon; the search is greedy coordinate descent over single-character
assignments, seeded by frequency analysis.

Two properties make this safe to run unattended:

  * the identity map is the baseline, so a correctly-encoded Unicode PDF is returned
    untouched instead of being scrambled;
  * every accepted step must strictly reduce OOV, so the result is never worse than
    doing nothing.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

from .alphabet import CANONICAL_LETTERS
from .normalize import normalize, words

log = logging.getLogger(__name__)

#: Letters a legacy font is likely to be hiding. Plain ASCII is excluded: those
#: render correctly and are never the thing that needs recovering.
TARGETS = "ɣɛḍṭṣẓṛḥčǧ"
TARGETS_UPPER = "ƔƐḌṬṢẒṚḤČǦ"

#: Multi-character targets. Legacy fonts often ligature the geminated emphatics
#: into one glyph, which is why `ime§§i` and `ñagwad` both appear in the wild.
TARGETS_MULTI = ("ṭṭ", "ḍḍ", "ẓẓ", "ṣṣ", "ṛṛ", "ḥḥ", "ɣɣ", "ɛɛ")

#: Never treated as cipher material. Typesetters kept punctuation working, so these
#: are always themselves -- and letting the search touch them is actively harmful:
#: deleting a comma merges two words into something that may hit the lexicon by
#: accident, which the objective would happily reward.
PUNCT_IMMUNE = frozenset(".,;:!?\"'()[]{}«»-/…’‘“”")


@dataclass
class CipherSolution:
    table: dict[str, str]
    oov_before: float
    oov_after: float
    rounds: int

    @property
    def improved(self) -> bool:
        return self.oov_after < self.oov_before - 1e-9

    def summary(self) -> str:
        pairs = " ".join(f"{k}->{v}" for k, v in sorted(self.table.items()))
        return (
            f"oov {self.oov_before:.1%} -> {self.oov_after:.1%} "
            f"in {self.rounds} rounds; {len(self.table)} substitutions\n    {pairs}"
        )


#: Emphatic consonants are written inconsistently in real Kabyle -- Wikipedia largely
#: uses plain r/s/t/z where literary orthography uses ṛ/ṣ/ṭ/ẓ. Scoring against an
#: unfolded lexicon therefore punishes the *correct* substitution (`uôumi -> uṛumi`
#: is absent from a lexicon holding `urumi`), and the search escapes by deleting the
#: glyph instead. Folding both sides removes that trap.
#: Only the true emphatics fold. č/c and ǧ/g are *distinct letters* that Kabyle
#: writes reliably, so folding them would let the search confuse `ç -> č` with
#: `ç -> ṣ` and pick the wrong one.
EMPHATIC_FOLD = str.maketrans({
    "ṛ": "r", "ṣ": "s", "ṭ": "t", "ẓ": "z", "ḍ": "d", "ḥ": "h",
})


def fold_emphatics(word: str) -> str:
    """Collapse emphatic/plain spelling variants. ɣ and ɛ are left alone -- they are
    never ambiguous and folding them would lose real information."""
    return word.translate(EMPHATIC_FOLD)


def _oov(text: str, lexicon: set[str]) -> float:
    toks = [w.lower() for w in words(text)]
    if not toks:
        return 1.0
    return sum(1 for w in toks if w not in lexicon) / len(toks)


def _apply(text: str, table: dict[str, str]) -> str:
    return text.translate({ord(k): v for k, v in table.items()}) if table else text


def _words_with(text: str, extra: set[str]) -> list[str]:
    """Tokenize, treating `extra` characters as ordinary word characters.

    The suspect set has to be part of a single repeated character class. Matching one
    suspect between suspect-*excluding* classes splits `maççi` into `maç` + `çi`, and
    a search scored on fragments cannot find the right substitution.
    """
    if not extra:
        return words(text)
    cls = "".join(re.escape(c) for c in sorted(extra))
    letter = rf"(?:[^\W\d_]|[{cls}])"
    pat = re.compile(rf"{letter}+(?:[-']{letter}+)*", re.UNICODE)
    return [w for w in pat.findall(text) if w]


class _Scorer:
    """Fast OOV evaluation for candidate substitution tables.

    The search runs thousands of trials, so re-normalizing and re-tokenizing the raw
    sample each time is hopeless. Instead the sample is tokenized once into weighted
    word *types*, and each trial only re-checks the types that actually contain a
    substitutable character. On a typical book that is a ~200x reduction in work.
    """

    __slots__ = ("lexicon", "types", "fixed_ok", "total", "fold")

    def __init__(
        self, sample: str, lexicon: set[str], alphabet: set[str], *, fold: bool = True
    ):
        self.fold = fold
        self.lexicon = {fold_emphatics(w) for w in lexicon} if fold else lexicon
        # The default word regex splits on `$`, so `yettwan$a` would be scored as two
        # fragments and the `$ -> ɣ` substitution could never pay off. Suspect
        # characters are therefore promoted to word characters for scoring.
        counts = Counter(w.lower() for w in _words_with(normalize(sample), alphabet))
        self.total = sum(counts.values()) or 1
        # Types containing no substitutable character can never change verdict, so
        # their contribution is folded into a constant up front.
        self.types: list[tuple[str, int]] = []
        self.fixed_ok = 0
        for w, n in counts.items():
            if any(c in alphabet for c in w):
                self.types.append((w, n))
            elif (fold_emphatics(w) if fold else w) in self.lexicon:
                self.fixed_ok += n

    def oov(self, table: dict[str, str]) -> float:
        lex = self.lexicon
        ok = self.fixed_ok
        if table:
            trans = {ord(k): v for k, v in table.items()}
            for w, n in self.types:
                cand = w.translate(trans)
                if (fold_emphatics(cand) if self.fold else cand) in lex:
                    ok += n
        else:
            for w, n in self.types:
                if (fold_emphatics(w) if self.fold else w) in lex:
                    ok += n
        return 1.0 - ok / self.total


def reference_profile(text: str) -> dict[str, float]:
    """Relative letter frequencies of known-good Kabyle, for frequency analysis."""
    counts = Counter(c.lower() for c in text if c.isalpha())
    total = sum(counts.values()) or 1
    return {c: n / total for c, n in counts.items()}


def suspects(
    sample: str, profile: dict[str, float], *, ratio: float = 4.0, min_rate: float = 0.001
) -> list[str]:
    """Characters that are plausibly a disguised Kabyle letter.

    The decisive signal is *word-internal position*: legacy fonts hijack whatever
    codepoint was convenient, so `$`, `§`, `µ` and `^` turn up surrounded by letters.
    That is what separates a hijacked glyph from real punctuation -- `$` between two
    letters is ɣ, `$` before a digit is currency.

    A canonical letter also becomes suspect when it appears far more often than real
    Kabyle justifies (`v` at 3% when Kabyle uses it only in loanwords).
    """
    internal: Counter[str] = Counter()
    letters: Counter[str] = Counter()
    for i, ch in enumerate(sample):
        if ch.isalpha():
            letters[ch] += 1
        if ch.isspace() or ch.isdigit() or ch in PUNCT_IMMUNE:
            continue
        prev = sample[i - 1] if i else " "
        nxt = sample[i + 1] if i + 1 < len(sample) else " "
        if prev.isalpha() or nxt.isalpha():
            internal[ch] += 1

    total = sum(letters.values()) or 1
    out: list[str] = []
    for ch, n in internal.items():
        if n / total < min_rate:
            continue
        if ch not in CANONICAL_LETTERS:
            # Not a Kabyle letter, yet living inside words -> hijacked codepoint.
            out.append(ch)
        else:
            obs = letters[ch] / total
            if obs > ratio * profile.get(ch.lower(), 1e-5) and obs > 0.002:
                out.append(ch)
    out.sort(key=lambda c: -internal[c])
    return out


def solve(
    sample: str,
    lexicon: set[str],
    profile: dict[str, float],
    *,
    max_rounds: int = 14,
    min_gain: float = 0.002,
    seed: dict[str, str] | None = None,
    allow_delete: bool = False,
) -> CipherSolution:
    """Recover the substitution table for an unknown font encoding.

    `seed` pre-seeds known assignments (e.g. a partial map from a related font),
    which both speeds up and stabilizes the search.
    """
    sample = sample[:40_000]
    table: dict[str, str] = dict(seed or {})

    cands = suspects(sample, profile)
    if not cands:
        flat = _oov(normalize(_apply(sample, table)), lexicon)
        return CipherSolution(table, flat, flat, 0)

    alphabet = set(cands)
    scorer = _Scorer(sample, lexicon, alphabet, fold=True)
    base = start = scorer.oov(table)

    # Deletion is deliberately absent: legacy fonts substitute glyphs, they do not
    # insert junk, and an "erase it" move is a cheap way for a greedy search to fake
    # progress on a word it cannot actually solve.
    targets = list(TARGETS) + list(TARGETS_UPPER) + list(TARGETS_MULTI)
    if allow_delete:
        targets.append("")
    log.debug("cipher: %d suspects x %d targets", len(cands), len(targets))

    rounds = 0
    for rounds in range(1, max_rounds + 1):
        # A font encoding is a bijection: each glyph stands for exactly one letter,
        # and no letter is drawn by two glyphs. Without this constraint the greedy
        # search happily assigns `ç -> ṣ` *and* `û -> ṣ`, which lets a wrong early
        # move block the right one (`ç -> č`) from ever being reachable.
        used = {v for v in table.values() if v}
        best_gain, best_move = 0.0, None
        for ch in cands:
            if ch in table:
                continue
            for tgt in targets:
                if tgt in used:
                    continue
                score = scorer.oov({**table, ch: tgt})
                gain = base - score
                if gain > best_gain:
                    best_gain, best_move = gain, (ch, tgt, score)
        if best_move is None or best_gain < min_gain:
            break
        ch, tgt, base = best_move
        table[ch] = tgt
        log.debug("  round %d: %r -> %r  oov %.2f%%", rounds, ch, tgt, base * 100)

    table = _disambiguate_emphatics(table, sample, lexicon, alphabet)
    exact = _Scorer(sample, lexicon, alphabet, fold=False)
    return CipherSolution(table, start, exact.oov(table), rounds)


def _disambiguate_emphatics(
    table: dict[str, str], sample: str, lexicon: set[str], alphabet: set[str]
) -> dict[str, str]:
    """Decide emphatic vs plain for each solved glyph, using the *unfolded* lexicon.

    The main search folds ṛ/r, ṣ/s, ṭ/t, ẓ/z together so orthographic inconsistency
    cannot mislead it. That robustness costs information: it cannot tell `ô -> ṛ` from
    `ô -> r`. This pass recovers it by testing both variants against exact spellings
    and keeping whichever the corpus actually supports.
    """
    # EMPHATIC_FOLD is a maketrans table keyed by ordinal: {ord('ṛ'): 'r', ...}
    plain = {chr(k): v for k, v in EMPHATIC_FOLD.items()}
    exact = _Scorer(sample, lexicon, alphabet, fold=False)
    out = dict(table)
    for ch, tgt in table.items():
        if len(tgt) != 1 or tgt not in plain:
            continue
        alt = plain[tgt]  # the plain counterpart of an emphatic target
        if exact.oov({**out, ch: alt}) < exact.oov(out):
            out[ch] = alt
    return out


def solve_per_font(
    spans: list[tuple[str, str]],
    lexicon: set[str],
    profile: dict[str, float],
    *,
    min_chars: int = 2_000,
) -> dict[str, CipherSolution]:
    """Solve a cipher independently for each font in a document.

    Fonts with too little text to score reliably are skipped -- a map fitted to 200
    characters is noise, and applying it would corrupt the span.
    """
    by_font: dict[str, list[str]] = {}
    for font, text in spans:
        by_font.setdefault(font, []).append(text)

    out: dict[str, CipherSolution] = {}
    for font, parts in by_font.items():
        joined = "".join(parts)
        if len(joined) < min_chars:
            continue
        sol = solve(joined, lexicon, profile)
        if sol.improved:
            out[font] = sol
            log.info("font %-40s %s", font, sol.summary().splitlines()[0])
    return out
