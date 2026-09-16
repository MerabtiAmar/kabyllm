"""Kabyle word lists, used as the objective function for automated quality checks.

Out-of-vocabulary rate against a real Kabyle lexicon is the cheapest honest signal
available for "is this actually well-formed Kabyle". It drives three things:

  * discovering legacy PDF font maps (`kabyl.pdf_decode.discover_map`)
  * filtering scraped documents (`kabyl.filters`)
  * scoring model generations (`kabyl.evaluate`)

Preference order is hunspell (a real curated dictionary with morphology) > the verb
paradigm table > a corpus-derived fallback. The fallback is weaker -- it inherits
whatever misspellings the corpus contains -- but it needs no network and makes the
pipeline runnable offline.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from .normalize import normalize, words

log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parent.parent / "data" / "interim" / "lexicon.json"


def from_hunspell(repo: str = "boffire/hunspell-kab") -> set[str]:
    """Load the Kabyle hunspell wordlist from the Hub.

    The .dic format is `word/FLAGS`, one per line, with a leading count line.
    """
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id=repo, repo_type="dataset"))
    lex: set[str] = set()
    for dic in list(root.rglob("*.dic")) + list(root.rglob("*.txt")):
        for line in dic.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.isdigit() or line.startswith("#"):
                continue
            lex.add(normalize(line.split("/")[0]).lower())
    log.info("hunspell lexicon: %d entries from %s", len(lex), repo)
    return {w for w in lex if w}


def from_verbs(repo: str = "boffire/kabyle-verbs") -> set[str]:
    """Load ~344k inflected verb forms. Complements hunspell's stem coverage."""
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    lex: set[str] = set()
    for col in ds.column_names:
        if ds.features[col].dtype == "string":
            lex.update(normalize(str(v)).lower() for v in ds[col] if v)
    forms = {w for entry in lex for w in words(entry)}
    log.info("verb lexicon: %d forms from %s", len(forms), repo)
    return forms


def from_corpus(paths: list[str | Path], min_freq: int = 2) -> set[str]:
    """Build a fallback lexicon from known-clean Kabyle text.

    `min_freq` >= 2 drops most OCR noise and typos, which appear once.
    """
    counts: Counter[str] = Counter()
    for p in paths:
        text = normalize(Path(p).read_text(encoding="utf-8", errors="replace"))
        counts.update(w.lower() for w in words(text))
    lex = {w for w, n in counts.items() if n >= min_freq}
    log.info("corpus lexicon: %d types (>= %d occurrences)", len(lex), min_freq)
    return lex


def load(paths: list[str | Path] | None = None, *, use_cache: bool = True) -> set[str]:
    """Best available lexicon, cached to disk.

    Tries hunspell + verbs from the Hub, then falls back to `paths`.
    """
    if use_cache and CACHE.exists():
        lex = set(json.loads(CACHE.read_text(encoding="utf-8")))
        log.info("lexicon: %d entries (cached)", len(lex))
        return lex

    lex: set[str] = set()
    for name, fn in (("hunspell", from_hunspell), ("verbs", from_verbs)):
        try:
            lex |= fn()
        except Exception as exc:  # offline, gated repo, schema drift
            log.warning("lexicon source %r unavailable: %s", name, exc)

    if len(lex) < 5_000 and paths:
        log.warning("falling back to corpus-derived lexicon")
        lex |= from_corpus(paths)

    if lex:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(sorted(lex), ensure_ascii=False), encoding="utf-8")
    return lex
