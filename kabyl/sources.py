"""Registry of every Kabyle corpus the pipeline knows how to pull.

The v0 project hand-rolled a Wikipedia API scraper that got rate-limited and died at
batch 21 of 143, yielding 741 KB. The sources below reach roughly 20-40M words with
no scraping at all -- an active Kabyle NLP community has already done the collection
work and published it.

Each entry declares where the text lives and how much to trust it. `weight` is the
sampling multiplier used when building the training mix: Tatoeba alone contributes
~750k short sentences and would otherwise dominate the corpus with one register
(brief, translated-from-English, heavy on the name "Tom").

Nothing here is verified for quality by the pipeline. `Source.notes` records what to
watch for, and `scripts/02_build_corpus.py --audit` prints samples per source so a
native speaker can spot-check before committing to a training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    MONO = "mono"          # raw Kabyle text
    PARALLEL = "parallel"  # sentence pairs with en/fr
    LEXICAL = "lexical"    # dictionaries, paradigms, name lists
    DIALOG = "dialog"      # multi-turn conversations, for SFT
    PDF = "pdf"            # raw documents needing Phase-1 decoding


@dataclass
class Source:
    key: str
    kind: Kind
    repo: str = ""
    #: Column holding Kabyle text. Always state it explicitly for bilingual datasets:
    #: guessing "the widest string column" silently selects the English side.
    text_col: str | None = None
    #: (other_language_col, kabyle_col) for parallel data.
    pair_cols: tuple[str, str] | None = None
    #: HuggingFace config/subset name.
    config: str | None = None
    #: Load these files individually. Needed when a repo's files have mismatched
    #: schemas, which makes a single load_dataset call fail outright.
    data_files: tuple[str, ...] = ()
    #: Kabyle text lives inside a dict column (HF `Translation` feature).
    translation_key: str | None = None
    #: Column holding a ShareGPT-style list of {from, value} turns.
    conversations_key: str | None = None
    split: str = "train"
    #: Sampling multiplier for the training mix. <1 downweights over-represented
    #: registers; >1 boosts scarce, high-value material.
    weight: float = 1.0
    #: Rough sentence/row count from the dataset card, for budgeting only.
    approx_rows: int = 0
    trust: str = "unverified"
    notes: str = ""
    url: str = ""


# --- Tier A: monolingual ------------------------------------------------------

MONO: list[Source] = [
    Source(
        "ummto", Kind.MONO, "Imsidag-community/kabyle-corpus-ummto",
        approx_rows=690_917, weight=1.3, trust="unverified",
        notes="Paragraphs mined from open-access PDFs at Univ. Mouloud Mammeri "
              "Tizi-Ouzou. Long-form academic/literary register -- the scarcest and "
              "most valuable kind of Kabyle text. Expect PDF extraction artefacts.",
    ),
    Source(
        "ubouira", Kind.MONO, "Imsidag-community/kabyle-corpus-ubouira",
        approx_rows=336_982, weight=1.3, trust="unverified",
        notes="Same pipeline as ummto, Univ. Bouira repository.",
    ),
    Source(
        "hca", Kind.MONO, "Imsidag-community/kabyle-corpus-hca",
        approx_rows=188_620, weight=1.3, trust="unverified",
        notes="Haut Commissariat a l'Amazighite. Institutional, edited, likely the "
              "most orthographically standard source in the set.",
    ),
    Source(
        "tatoeba_mono", Kind.MONO, "boffire/tatoeba-kabyle-mono-cleaned",
        approx_rows=756_774, weight=0.4, trust="good",
        notes="DOWNWEIGHTED. Clean and conversational, but short sentences largely "
              "translated from English by a handful of contributors. At full weight "
              "it swamps every other register; 'tom' is already the 20th most "
              "frequent token in the v0 corpus because of this.",
    ),
    Source(
        "raw_text", Kind.MONO, "Imsidag-community/kabyle-raw-text",
        text_col="text", approx_rows=207_058, weight=1.0, trust="unverified",
    ),
    Source(
        "datasets_mono", Kind.MONO, "Imsidag-community/kabyle-datasets-mono",
        text_col="text", approx_rows=2_475, weight=1.0, trust="unverified",
        # One file carries an extra `kabyle_fix_metadata` column, so a combined load
        # fails on the schema mismatch. Loaded file by file instead.
        data_files=("aghucaf-blog-092025.jsonl", "depeche-kab.jsonl",
                    "sadbendali-blog.jsonl"),
        notes="Blog and news prose (Depeche de Kabylie). Contemporary register.",
    ),
    Source(
        "wikipedia", Kind.MONO, "wikimedia/wikipedia", split="train",
        text_col="text", config="20231101.kab", approx_rows=5_830,
        weight=1.5, trust="good",
        notes="Config kab. Prefer the official dump over the API: "
              "dumps.wikimedia.org/kabwiki/latest/. Encyclopedic register, the only "
              "source with sustained factual prose.",
        url="https://dumps.wikimedia.org/kabwiki/latest/kabwiki-latest-pages-articles.xml.bz2",
    ),
    Source(
        "glotcc", Kind.MONO, "cis-lmu/GlotCC-V1", config="kab-Latn",
        text_col="content", weight=0.7, trust="noisy",
        notes="CommonCrawl-derived. Config is `kab-Latn` with a hyphen, not an "
              "underscore. Needs the strictest filtering of anything here: "
              "boilerplate, other Berber varieties, MT output.",
    ),
    Source(
        "tales", Kind.MONO, "boffire/timucuha-kabyle-tales", text_col="kabyle",
        approx_rows=874, weight=1.5, trust="unverified",
        notes="Folk tales with fr/en translations. Narrative register. The Kabyle "
              "column is `kabyle`; column guessing picks `french` here.",
    ),
]

#: Registered but not loadable, kept so the reason is recorded rather than rediscovered.
UNAVAILABLE: dict[str, str] = {
    "boffire/kabyle-lyrics": "repo contains no data files (checked 2026-08)",
}


# --- Tier B: parallel ---------------------------------------------------------

PARALLEL: list[Source] = [
    Source(
        "tatoeba_en", Kind.PARALLEL, "boffire/tatoeba-en-kab",
        translation_key="translation", pair_cols=("en", "kab"),
        approx_rows=240_056, weight=1.0, trust="good",
        notes="Both sides live in a single `translation` dict column (HF Translation "
              "feature). Train/dev/test splits are provided -- use its test split as "
              "the chrF++ benchmark rather than carving a new one.",
    ),
    Source(
        "tm_en", Kind.PARALLEL, "boffire/kabyle-english-TM",
        pair_cols=("source", "target"), text_col="target",
        approx_rows=121_725, weight=1.0, trust="unverified",
        notes="Translation memory; `source`=en, `target`=kab (with *_lang columns "
              "confirming). Professionally produced, usually high quality.",
    ),
    Source(
        "nllb_en", Kind.PARALLEL, "Imsidag-community/nllb_en_kab",
        pair_cols=("english", "kabyle"), text_col="kabyle",
        approx_rows=2_484_297, weight=0.3, trust="noisy",
        notes="LASER-mined bitext from CommonCrawl, pre-filtered with GlotLID. Huge "
              "but misaligned often enough that most should be discarded. Column "
              "guessing selects `english` here -- hence the explicit text_col.",
    ),
    Source(
        "translatewiki", Kind.PARALLEL, "boffire/kabyle-english-translatewiki",
        translation_key="translation", pair_cols=("en", "kab"),
        approx_rows=7_085, weight=0.8, trust="good",
        notes="Software UI strings. Useful register for an assistant, but fragmentary.",
    ),
    Source(
        "toponym_sents", Kind.PARALLEL, "boffire/kab-en-toponyms-sentences",
        translation_key="translation", pair_cols=("en", "kab"),
        approx_rows=32_024, weight=0.6, trust="unverified",
        notes="Template-generated. Card claims 'grammatically flawless' -- verify; "
              "templated text at high weight teaches the model to be repetitive.",
    ),
]


# --- Tier C: lexical / structured --------------------------------------------

LEXICAL: list[Source] = [
    Source(
        "verbs", Kind.LEXICAL, "boffire/kabyle-verbs", config="conjugation-tables",
        approx_rows=344_745, trust="good",
        notes="6,198 verbs x full paradigms. The repo has three configs "
              "(conjugation-tables, lemmatizer, seq2seq) and requires one to be "
              "named. Drives the SFT morphology tasks.",
    ),
    Source(
        "toponyms", Kind.LEXICAL, "boffire/kabyle-toponyms", text_col="kabyle",
        pair_cols=("french", "kabyle"), approx_rows=8_849, trust="good",
        notes="Georeferenced Algerian place names from OpenStreetMap. Column "
              "guessing picks the `name_tags` JSON blob; the clean column is `kabyle`.",
    ),
    Source(
        "named_entities", Kind.LEXICAL, "boffire/kabyle-named-entities",
        text_col="kabyle_standardized",
        pair_cols=("english_translation", "kabyle_standardized"),
        approx_rows=292, trust="good", notes="Manually curated.",
    ),
    Source(
        "dialog", Kind.DIALOG, "kmerakeb/kabyle-dialog-corpus", trust="noisy",
        data_files=("kabyle_aya_fr.jsonl", "kabyle_oasst1.jsonl"),
        conversations_key="conversations", weight=0.5,
        notes="ShareGPT-format conversations, on-target for a chat model. BUT the "
              "`source` field reads *-fr-kab-nllb200: these are NLLB machine "
              "translations from French, and some assistant turns were never "
              "translated at all and are still in French. Per-turn language "
              "filtering is mandatory; treat as a supplement, not a foundation.",
    ),
    Source(
        "math_lexicon", Kind.LEXICAL, "Agurzil-Amazigh/tafsut-lexique-mathematiques-kabyle",
        trust="good", data_files=("tafsut_math_lexicon.jsonl",
                                  "tafsut_math_translation_pairs.jsonl"),
        notes="Technical/neologism vocabulary. Two files, mismatched schemas.",
    ),
]

#: Not loadable through `datasets` -- raw dictionary files fetched with
#: `snapshot_download` by `kabyl.lexicon.from_hunspell`.
HUNSPELL_REPO = "boffire/hunspell-kab"


# --- Tier D: raw PDFs ---------------------------------------------------------

PDFS: list[Source] = [
    Source("ayamun", Kind.PDF, "boffire/ayamun-pdfs", approx_rows=230,
           notes="Kabyle cultural revue. Expect legacy Berber font encodings."),
    Source("bouira_theses", Kind.PDF, "boffire/bouira", approx_rows=395,
           notes="Magistere theses. Mixed Kabyle/French -- filter per page."),
    Source("adlis", Kind.PDF, "boffire/adlis-pdfs", notes="Books."),
    Source("adlis_imsidag", Kind.PDF, "Imsidag-community/kabyle-adlis-pdfs"),
]


ALL: list[Source] = MONO + PARALLEL + LEXICAL + PDFS
BY_KEY: dict[str, Source] = {s.key: s for s in ALL}


def of_kind(kind: Kind) -> list[Source]:
    return [s for s in ALL if s.kind is kind]


#: Local files already present in the project, folded in alongside the Hub sources.
LOCAL_MONO: dict[str, str] = {
    "wiki_v0": "kabyle_wiki_corpus_cleaned.txt",
    "kab_latn_v0": "kab_Latn.txt",
    "feraoun": "data/interim/Mouloud Feraoun Moussa Ould Taleb.txt",
    "dahmoune": "data/interim/051_Omar Dahmoune.txt",
}


#: Replay data. Continued pretraining on Kabyle alone would erode the reasoning the
#: base model already has, so 30-40% of every batch stays in languages it knows.
#: French is weighted heavily because Kabyle speakers code-switch with it constantly
#: and the assistant will be addressed in both.
REPLAY: list[dict] = [
    {"repo": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "lang": "en", "share": 0.4},
    {"repo": "HuggingFaceFW/fineweb-2", "config": "fra_Latn", "lang": "fr", "share": 0.6},
]
