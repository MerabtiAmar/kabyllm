"""Character maps for pre-Unicode Berber PDF fonts.

Algerian Kabyle publishing predates usable Unicode support, so typesetters used
custom fonts that draw ɣ, ɛ, ḍ, ṭ, ṣ, ẓ, ḥ, č on top of unrelated Latin-1
codepoints. A PDF extractor faithfully returns the *codepoints*, which is why
`Mouloud_Feraoun_Mmis_n_igellil_extracted.txt` reads as `aîas` instead of `aṭas`.

These are substitution ciphers, and they are per-font: the same book uses one
encoding for its roman face and a different one for bold/italic. Two glyphs in the
Feraoun extraction map to ɣ (`$` and `ù`) and two to ṭ (`î` and `§`) for exactly
that reason. A global find-replace therefore corrupts the text; decoding must be
done span by span with the span's font in hand.

Maps are verified against the Kabyle hunspell dictionary -- see
`kabyl.pdf_decode.score_map`, which is also how new maps get discovered.
"""

from __future__ import annotations

# --- Verified maps ------------------------------------------------------------

#: The dominant encoding in the Feraoun scan. Confirmed against known words:
#: aîas->aṭas, neéra->neẓra, yelêan->yelḥan, uôumi->uṛumi, laûel->laṣel,
#: maççi->mačči, yewwev-ed->yewweḍ-ed, yettwan$a->yettwanɣa.
BERBER_TIMES_A: dict[str, str] = {
    "$": "ɣ", "£": "Ɣ",
    "ε": "ɛ", "∑": "Ɛ",
    "v": "ḍ", "V": "Ḍ",
    "ô": "ṛ", "Ö": "Ṛ",
    "û": "ṣ", "Û": "Ṣ",
    "î": "ṭ", "Î": "Ṭ",
    "ê": "ḥ", "Ë": "Ḥ",
    "é": "ẓ", "É": "Ẓ",
    "ç": "č", "Ç": "Č",
    "ï": "ǧ", "Ï": "Ǧ",
}

#: The Tayma face, used for the same book's secondary text. A genuinely different
#: cipher from BERBER_TIMES_A -- note `ê` means ṛ here but ḥ there, which is exactly
#: why decoding has to be per-font.
#:
#: Recovered by `kabyl.cipher.solve` and cross-checked by hand against ùef->ɣef,
#: ayùer->ayɣer, lqaµa->lqaɛa, teswiµt->teswiɛt, ime§§i->imeṭṭi, ∑emmi->Ɛemmi.
BERBER_TIMES_B: dict[str, str] = {
    "ù": "ɣ", "Ù": "Ɣ",
    "µ": "ɛ", "∑": "Ɛ",
    "§": "ṭ", "¤": "Ṭ",
    "á": "č", "Á": "Č",
    "â": "ẓ", "Â": "Ẓ",
    "ê": "ṛ", "Ê": "Ṛ",
    "î": "ṣ", "Î": "Ṣ",
    "ô": "ḍ", "Ô": "Ḍ",
    "ú": "ǧ", "Ú": "Ǧ",
    "û": "ḥ", "Û": "Ḥ",
    "ñ": "ṭṭ", "Ñ": "Ṭṭ",
    "^": "ṭṭ",
}

#: Union of both, for documents where font information was lost (e.g. text that was
#: already extracted to .txt before anyone noticed the problem). Conflicting keys
#: resolve to the A map, which is the more common face.
BERBER_TIMES_MERGED: dict[str, str] = {**BERBER_TIMES_B, **BERBER_TIMES_A}


#: Font-name substring -> map. Matching is case-insensitive substring containment
#: because PDF font names carry subset prefixes like "ABCDEF+TimesBerber-Roman".
FONT_MAPS: dict[str, dict[str, str]] = {
    "amazighbangle": BERBER_TIMES_A,
    "amazightimes": BERBER_TIMES_A,
    "tayma": BERBER_TIMES_B,
    "berber": BERBER_TIMES_A,
    "tamazight": BERBER_TIMES_A,
    "amazigh": BERBER_TIMES_A,
    "kabyle": BERBER_TIMES_A,
}

#: Tried in order by the auto-detector when a font name is unknown.
CANDIDATE_MAPS: dict[str, dict[str, str]] = {
    "identity": {},
    "berber_times_a": BERBER_TIMES_A,
    "berber_times_b": BERBER_TIMES_B,
    "berber_times_merged": BERBER_TIMES_MERGED,
}


def map_for_font(font_name: str) -> dict[str, str] | None:
    """Return the map registered for `font_name`, or None if unknown."""
    low = font_name.lower()
    for key, table in FONT_MAPS.items():
        if key in low:
            return table
    return None


def apply_map(text: str, table: dict[str, str]) -> str:
    """Apply a cipher map. Multi-character values (ñ -> ṭṭ) are supported."""
    if not table:
        return text
    return text.translate({ord(k): v for k, v in table.items()})


def register(font_substring: str, table: dict[str, str]) -> None:
    """Teach the decoder a font discovered by `kabyl.pdf_decode.discover_map`."""
    FONT_MAPS[font_substring.lower()] = table
