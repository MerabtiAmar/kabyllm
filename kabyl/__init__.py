"""KabyLLM - Kabyle language model toolkit.

Pipeline stages, in dependency order:

    normalize / pdf_decode   canonical orthography, legacy font recovery
    sources / filters        corpus aggregation, language ID, dedup
    tokenizer                fertility measurement, vocabulary extension
    sft                      instruction data construction
    evaluate                 fertility, perplexity, chrF++, OOV, forgetting
"""

__version__ = "0.1.0"

from .alphabet import CANONICAL, CANONICAL_LETTERS
from .normalize import NormalizeReport, adhoc_score, audit, diacritic_rate, normalize, words

__all__ = [
    "CANONICAL",
    "CANONICAL_LETTERS",
    "NormalizeReport",
    "adhoc_score",
    "audit",
    "diacritic_rate",
    "normalize",
    "words",
]
