"""Phase 1 regression tests: orthographic normalization and legacy font decoding.

Every case here is drawn from a bug that actually occurred while building the
pipeline, or from a real string in the source corpora.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import fontmaps
from kabyl.alphabet import CANONICAL
from kabyl.cipher import fold_emphatics, suspects, reference_profile
from kabyl.normalize import adhoc_score, audit, diacritic_rate, normalize, words


# --- Homoglyph folding --------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ε", "ɛ"),        # Greek epsilon      -> ɛ
        ("ԑ", "ɛ"),        # Cyrillic komi dzje -> ɛ
        ("Ԑ", "Ɛ"),        # Cyrillic capital   -> Ɛ
        ("ɤ", "ɣ"),        # rams horn          -> ɣ
        ("γ", "ɣ"),        # Greek gamma        -> ɣ
        ("Γ", "Ɣ"),        # Greek cap gamma    -> Ɣ
        ("ħ", "ḥ"),        # h with stroke      -> ḥ
        ("ș", "ṣ"),        # s with comma       -> ṣ
        ("ț", "ṭ"),        # t with comma       -> ṭ
    ],
)
def test_confusables_fold_to_canonical(raw, expected):
    assert normalize(raw) == expected


def test_decomposed_forms_compose():
    """h + combining dot below must become the single codepoint ḥ."""
    assert normalize("ḥ") == "ḥ"
    assert normalize("ḍ") == "ḍ"


def test_dahmoune_confusables_are_repaired():
    """The real defect in 051_Omar_Dahmoune_extracted.txt: Cyrillic Ԑ/ԑ for Ɛ/ɛ."""
    assert normalize("Ԑmiruc") == "Ɛmiruc"       # Ԑmiruc -> Ɛmiruc
    assert normalize("iԑ-d-ismenԑen") == "iɛ-d-ismenɛen"


def test_normalized_text_is_canonical():
    src = "TԐmaziɤt εemmi — “azul”… ḥed"
    out = normalize(src)
    assert not audit(out), f"residual: {audit(out)}"
    assert set(out) <= CANONICAL


# --- Ligatures and PDF artefacts ---------------------------------------------

def test_pdf_ligatures_expand():
    assert normalize("aﬁx") == "afix"
    assert normalize("ﬂuks") == "fluks"


def test_line_break_hyphen_is_removed_for_split_words():
    assert normalize("tamu-\nrt") == "tamurt"
    assert normalize("imezda-\nɣen") == "imezdaɣen"


def test_single_char_continuation_is_read_as_a_clitic():
    """`bab-\\nd` is ambiguous: hyphenated word, or the deictic clitic `-d`?

    Typographic hyphenation never strands a single character on the next line, so
    the clitic reading is the safer default and `-d`/`-n`/`-t` keep their hyphen.
    """
    assert normalize("yewweḍ-\nd") == "yewweḍ-d"


def test_line_break_hyphen_is_kept_for_clitics():
    """Regression: `bab-\\nis` became `babis`, destroying the possessive."""
    assert normalize("bab-\nis") == "bab-is"
    assert normalize("axxam-\nnneɣ") == "axxam-nneɣ"


def test_spaced_clitic_is_retightened():
    assert normalize("yenna- yas") == "yenna-yas"
    assert normalize("zedɣen- tt") == "zedɣen-tt"


def test_date_ranges_are_not_treated_as_clitics():
    """`13 yennayer - 14 yennayer` in the wiki corpus is a separator, not a clitic."""
    assert normalize("13 yennayer - 14 yennayer") == "13 yennayer - 14 yennayer"


# --- Legacy font decoding -----------------------------------------------------

@pytest.mark.parametrize(
    "encoded, expected",
    [
        ("aîas", "aṭas"),                 # aîas   -> aṭas
        ("neéra", "neẓra"),               # neéra  -> neẓra
        ("yelêan", "yelḥan"),             # yelêan -> yelḥan
        ("uôumi", "uṛumi"),               # uôumi  -> uṛumi
        ("laûel", "laṣel"),               # laûel  -> laṣel
        ("maççi", "mačči"),     # maççi  -> mačči
        ("yettwan$a", "yettwanɣa"),            # $      -> ɣ
        ("yewwev-ed", "yewweḍ-ed"),            # v      -> ḍ
    ],
)
def test_berber_times_a_decodes(encoded, expected):
    assert normalize(fontmaps.apply_map(encoded, fontmaps.BERBER_TIMES_A)) == expected


@pytest.mark.parametrize(
    "encoded, expected",
    [
        ("ùef", "ɣef"),                   # ùef    -> ɣef
        ("ayùer", "ayɣer"),               # ayùer  -> ayɣer
        ("lqaµa", "lqaɛa"),               # lqaµa  -> lqaɛa
        ("ime§§i", "imeṭṭi"),   # ime§§i -> imeṭṭi
    ],
)
def test_berber_times_b_decodes(encoded, expected):
    assert normalize(fontmaps.apply_map(encoded, fontmaps.BERBER_TIMES_B)) == expected


def test_the_two_maps_genuinely_differ():
    """`ê` is ḥ in one face and ṛ in the other -- the reason decoding is per-font."""
    assert fontmaps.BERBER_TIMES_A["ê"] == "ḥ"   # ḥ
    assert fontmaps.BERBER_TIMES_B["ê"] == "ṛ"   # ṛ


def test_font_lookup_prefers_specific_over_generic():
    assert fontmaps.map_for_font("ABCDEF+AmazighBangle") is fontmaps.BERBER_TIMES_A
    assert fontmaps.map_for_font("XYZ+Tayma-Tlika") is fontmaps.BERBER_TIMES_B
    assert fontmaps.map_for_font("TimesNewRoman") is None


def test_maps_are_injective():
    """A font encoding is a bijection; a duplicated target means a bad table."""
    for name, table in (("A", fontmaps.BERBER_TIMES_A), ("B", fontmaps.BERBER_TIMES_B)):
        vals = [v for v in table.values() if len(v) == 1]
        assert len(vals) == len(set(vals)), f"map {name} maps two glyphs to one letter"


# --- Cipher solver primitives -------------------------------------------------

def test_suspects_finds_word_internal_symbols():
    prof = reference_profile("azul fell-awen tamurt n leqbayel ad yili")
    found = suspects("yettwan$a a$rum i$ef ta$ect a$ilif ne$ ma$ef", prof)
    assert "$" in found


def test_suspects_ignores_punctuation():
    """Deleting a comma can merge two words into a lexicon hit; never let it try."""
    prof = reference_profile("azul fell-awen tamurt n leqbayel")
    found = suspects("azul, tamurt. yella; wayed: melmi? nnig!" * 20, prof)
    assert not ({",", ".", ";", ":", "?", "!"} & set(found))


def test_emphatic_fold_collapses_orthographic_variants():
    assert fold_emphatics("uṛumi") == fold_emphatics("urumi")
    assert fold_emphatics("aṭas") == "atas"


def test_emphatic_fold_preserves_real_distinctions():
    """ɣ, ɛ, č and ǧ are unambiguous letters and must survive folding."""
    for ch in "ɣɛčǧ":
        assert fold_emphatics(ch) == ch


# --- Quality signals ----------------------------------------------------------

def test_diacritic_rate_separates_kabyle_from_french():
    kab = "Aṭas n yimezdaɣen i d-yusan ɣer tmurt n leqbayel"
    fra = "Beaucoup de gens sont venus au pays de Kabylie cette annee la"
    assert diacritic_rate(kab) > 0.05
    assert diacritic_rate(fra) < 0.01


def test_adhoc_score_flags_arabizi():
    assert adhoc_score("3emmi ghef thaddarth ikhdem") > 0.5
    assert adhoc_score("Ɛemmi ɣef taddart ixdem") == 0.0


def test_words_keeps_clitics_intact():
    """The v0 tokenizer split on `-` and destroyed Kabyle morphology."""
    assert words("zedɣen-tt deg wexxam") == ["zedɣen-tt", "deg", "wexxam"]
