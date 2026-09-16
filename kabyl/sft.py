"""Construction of Kabyle instruction data.

No Kabyle instruction dataset exists, so one has to be manufactured. The obvious
approach -- machine-translate a large English instruction set -- produces a model that
speaks NLLB-flavoured Kabyle: calqued word order, wrong particles, French loanwords
where native ones exist. Since the goal is a model a Kabyle speaker wants to use, that
failure mode is fatal rather than cosmetic.

The mix is therefore built the other way round. Most examples take their **response**
from text a Kabyle person actually wrote, and synthesize only the *instruction* that
would have elicited it. Translation is used where it is unavoidable, and everything it
produces passes a round-trip filter before being kept.

    corpus-grounded   ~60%   responses are real Kabyle prose
    translated        ~20%   filtered by round-trip similarity + language ID
    capability replay ~20%   English/French instructions, so the model keeps following
                             instructions at all
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import normalize, words

log = logging.getLogger(__name__)


@dataclass
class Example:
    instruction: str
    response: str
    task: str
    source: str = ""
    lang: str = "kab"

    def to_messages(self, system: str | None = None) -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": self.instruction})
        msgs.append({"role": "assistant", "content": self.response})
        return {"messages": msgs, "task": self.task, "lang": self.lang}


#: Default system prompt, in Kabyle: "You are a helpful assistant who speaks Kabyle."
SYSTEM_KAB = "Kečč d amɛiwen aseklan i yessawalen taqbaylit."


# --- Instruction templates ----------------------------------------------------
# Multiple phrasings per task so the model learns the task rather than the wording.

TEMPLATES: dict[str, list[str]] = {
    "qa": [
        "D acu-t {topic}?",                      # What is X?
        "Ini-yi-d ɣef {topic}.",                 # Tell me about X.
        "Sefhem-iyi-d ɣef {topic}.",             # Explain X to me.
        "Acu i tessneḍ ɣef {topic}?",            # What do you know about X?
    ],
    "summarize": [
        "Ggez-d aḍris-agi:\n\n{text}",           # Summarize this text
        "Fk-iyi-d agzul n uḍris-agi:\n\n{text}",
        "Acu i d-yenna uḍris-agi s wudem uzzil?\n\n{text}",
    ],
    "continue": [
        "Kemmel aḍris-agi:\n\n{text}",           # Continue this text
        "Kemmel tira n wayen i d-yeddan:\n\n{text}",
    ],
    "translate_to_kab": [
        "Suqel ɣer teqbaylit: {text}",
        "Amek i d-nniɣ ayagi s teqbaylit? {text}",
    ],
    "translate_from_kab": [
        "Suqel ɣer {lang}: {text}",
        "Acu i d lmeɛna n wayagi s {lang}? {text}",
    ],
    "conjugate": [
        "Sizreg amyag «{verb}» deg {tense}.",     # Conjugate verb X in tense Y
        "Amek i nesseqdac amyag «{verb}» deg {tense}?",
    ],
    "define": [
        "D acu i d lmeɛna n wawal «{word}»?",     # What does word X mean?
        "Sefhem-iyi-d awal «{word}».",
    ],
    "correct": [
        "Ṣeggem tira n uḍris-agi:\n\n{text}",     # Correct the spelling
        "Sefru tuccḍiwin n tira deg uḍris-agi:\n\n{text}",
    ],
}

TENSE_NAMES = {
    "aorist": "urmir",
    "preterite": "izri",
    "imperative": "anab",
    "participle": "amerna",
}


def _pick(rng: random.Random, task: str, **kw) -> str:
    return rng.choice(TEMPLATES[task]).format(**kw)


# --- Corpus-grounded builders -------------------------------------------------

_RE_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _RE_SENT.split(text) if s.strip()]


def build_qa(docs: list[dict], rng: random.Random, limit: int = 20_000) -> list[Example]:
    """Wikipedia-style: the article's own opening is the answer to "what is X?".

    The response is untouched human-written Kabyle. Only the question is synthetic,
    which is exactly the direction that avoids translationese.
    """
    out: list[Example] = []
    for d in docs:
        text = d["text"]
        sents = sentences(text)
        if len(sents) < 2:
            continue
        lead = " ".join(sents[:3])
        if len(words(lead)) < 15:
            continue
        # The subject is usually the first content word or two of the first sentence.
        head = words(sents[0])[:2]
        if not head:
            continue
        topic = " ".join(head)
        if len(topic) < 4:
            continue
        out.append(Example(_pick(rng, "qa", topic=topic), lead, "qa", d.get("source", "")))
        if len(out) >= limit:
            break
    return out


def build_summarize(docs: list[dict], rng: random.Random, limit: int = 10_000) -> list[Example]:
    """Body -> lead. Both halves are real text; the compression is genuine."""
    out: list[Example] = []
    for d in docs:
        sents = sentences(d["text"])
        if len(sents) < 8:
            continue
        lead, body = " ".join(sents[:2]), " ".join(sents[2:14])
        if len(words(body)) < 60:
            continue
        out.append(Example(_pick(rng, "summarize", text=body), lead,
                           "summarize", d.get("source", "")))
        if len(out) >= limit:
            break
    return out


def build_continue(docs: list[dict], rng: random.Random, limit: int = 10_000) -> list[Example]:
    """Writing-aid task: the model continues real prose in the author's register."""
    out: list[Example] = []
    for d in docs:
        w = words(d["text"])
        if len(w) < 80:
            continue
        cut = len(w) // 3
        head = " ".join(d["text"].split()[:cut])
        tail = " ".join(d["text"].split()[cut:cut + 150])
        if len(tail.split()) < 30:
            continue
        out.append(Example(_pick(rng, "continue", text=head), tail,
                           "continue", d.get("source", "")))
        if len(out) >= limit:
            break
    return out


def build_translation(pairs: list[tuple[str, str]], rng: random.Random,
                      other_lang: str = "taglizit", limit: int = 30_000) -> list[Example]:
    """Both directions from verified parallel data.

    `other_lang` is the Kabyle name of the other language: taglizit = English,
    tafransist = French.
    """
    out: list[Example] = []
    for src, tgt in pairs:
        if len(out) >= limit:
            break
        out.append(Example(_pick(rng, "translate_to_kab", text=src), tgt,
                           "translate_to_kab", "parallel"))
        if len(out) < limit:
            out.append(Example(
                _pick(rng, "translate_from_kab", text=tgt, lang=other_lang),
                src, "translate_from_kab", "parallel", lang="kab",
            ))
    return out


def build_morphology(verb_rows: list[dict], rng: random.Random,
                     limit: int = 8_000) -> list[Example]:
    """Conjugation drills from the 344k-form paradigm table.

    Explicit morphology supervision is unusually valuable for Kabyle: the language
    packs subject, object, aspect and deixis into one word, and this is the only
    source in the corpus that states those patterns rather than merely exhibiting them.
    """
    out: list[Example] = []
    for row in verb_rows:
        if len(out) >= limit:
            break
        lemma = str(row.get("lemma") or row.get("verb") or "").strip()
        if not lemma:
            continue
        for key, kab_name in TENSE_NAMES.items():
            forms = row.get(key)
            if not forms:
                continue
            if isinstance(forms, (list, tuple)):
                body = "\n".join(f"- {f}" for f in forms if f)
            else:
                body = str(forms)
            if not body.strip():
                continue
            out.append(Example(
                _pick(rng, "conjugate", verb=lemma, tense=kab_name),
                body, "conjugate", "verbs",
            ))
    return out


def build_spelling(docs: list[dict], rng: random.Random, lexicon: set[str],
                   limit: int = 6_000) -> list[Example]:
    """Corrupt clean text, ask for it back.

    Corruptions mimic the errors real Kabyle writing actually contains -- emphatics
    dropped, ɣ written as gh, ɛ written as 3 -- rather than random character noise.
    """
    corrupt = {"ɣ": "gh", "ɛ": "3", "ḥ": "h", "ḍ": "d", "ṭ": "t", "ṣ": "s", "ẓ": "z",
               "č": "c", "ǧ": "j", "ṛ": "r"}
    out: list[Example] = []
    for d in docs:
        if len(out) >= limit:
            break
        sents = sentences(d["text"])
        for s in sents:
            if not (8 <= len(words(s)) <= 40):
                continue
            bad = "".join(
                corrupt[c] if c in corrupt and rng.random() < 0.6 else c for c in s
            )
            if bad == s:
                continue
            out.append(Example(_pick(rng, "correct", text=bad), s, "correct", "spelling"))
            break
    return out


def build_definitions(entries: list[tuple[str, str]], rng: random.Random,
                      limit: int = 5_000) -> list[Example]:
    out: list[Example] = []
    for word, gloss in entries[:limit]:
        out.append(Example(_pick(rng, "define", word=word), gloss, "define", "lexicon"))
    return out


# --- Translation filtering ----------------------------------------------------

@dataclass
class TranslationFilter:
    """Keep only machine translations that survive a round trip.

    Translate en -> kab with NLLB, translate the result back, and keep the pair only
    if the returned English still means what the original did. Combined with a
    language-ID check and a lexicon check this discards most of what NLLB produces for
    a low-resource pair -- which is the intent. Precision matters far more than volume:
    a bad instruction pair teaches the model to produce bad Kabyle.
    """

    sim_threshold: float = 0.75
    max_oov: float = 0.5
    embedder: str = "sentence-transformers/LaBSE"
    _model: object = field(default=None, init=False, repr=False)

    def _embed(self, texts: list[str]):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedder)
        return self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)

    def keep(self, original: list[str], round_trip: list[str],
             kabyle: list[str], lexicon: set[str] | None = None) -> list[bool]:
        import numpy as np

        a = self._embed(original)
        b = self._embed(round_trip)
        sims = (a * b).sum(axis=1)

        out: list[bool] = []
        for sim, kab in zip(sims, kabyle):
            ok = bool(sim >= self.sim_threshold)
            if ok and lexicon:
                toks = [w.lower() for w in words(kab)]
                if toks:
                    oov = sum(1 for w in toks if w not in lexicon) / len(toks)
                    ok = oov <= self.max_oov
            out.append(ok)
        log.info("round-trip filter kept %d/%d (%.0f%%)",
                 sum(out), len(out), 100 * sum(out) / max(1, len(out)))
        return out


# --- Assembly -----------------------------------------------------------------

def write_jsonl(examples: list[Example], path: str | Path,
                system: str | None = SYSTEM_KAB) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_messages(system), ensure_ascii=False) + "\n")
    log.info("wrote %d examples -> %s", len(examples), p)


def mix_report(examples: list[Example]) -> str:
    from collections import Counter

    by_task = Counter(e.task for e in examples)
    by_lang = Counter(e.lang for e in examples)
    total = len(examples) or 1
    lines = [f"{total:,} examples"]
    for task, n in by_task.most_common():
        lines.append(f"    {n:>8,}  {n / total:5.1%}  {task}")
    lines.append(f"  languages: {dict(by_lang)}")
    return "\n".join(lines)
