"""Evaluation for a Kabyle language model.

The v0 project had no evaluation at all -- generations were eyeballed, so there was no
way to tell improvement from noise, and the val/train gap that revealed pure
memorization (6.21 vs 1.33) went unnoticed for 8,000 steps.

Six metrics, each answering a different question:

  fertility        how efficiently is Kabyle encoded?
  perplexity       per domain -- a pooled number hides fitting Tatoeba and ignoring prose
  chrF++           translation quality; chrF not BLEU, because Kabyle morphology makes
                   word-level n-gram overlap far too sparse to be informative
  language purity  what fraction of generated text is actually Kabyle?
  lexicon OOV      are the generated words real words?
  English MMLU     how much of the base model's reasoning survived adaptation?

The last one is the gate on continued pretraining. A model that speaks beautiful
Kabyle and has forgotten how to think is not what adapting a pretrained model was for.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .normalize import diacritic_rate, normalize, words

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    model: str = ""
    fertility: float = 0.0
    perplexity: dict[str, float] = field(default_factory=dict)
    chrf: dict[str, float] = field(default_factory=dict)
    language_purity: float = 0.0
    oov_rate: float = 0.0
    mmlu_en: float | None = None
    mmlu_en_base: float | None = None
    samples: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"model: {self.model}", ""]
        lines.append(f"  fertility          {self.fertility:.2f} tokens/word")
        if self.perplexity:
            lines.append("  perplexity by domain:")
            for dom, ppl in sorted(self.perplexity.items(), key=lambda kv: kv[1]):
                lines.append(f"      {ppl:9.2f}  {dom}")
            vals = list(self.perplexity.values())
            spread = max(vals) / max(1e-9, min(vals))
            lines.append(f"      spread {spread:.1f}x "
                         f"({'balanced' if spread < 3 else 'UNEVEN - check the mix'})")
        if self.chrf:
            lines.append("  chrF++:")
            for k, v in self.chrf.items():
                lines.append(f"      {v:6.2f}  {k}")
        lines.append(f"  language purity    {self.language_purity:.1%} of generations are Kabyle")
        lines.append(f"  lexicon OOV        {self.oov_rate:.1%} of generated words are unknown")
        if self.mmlu_en is not None:
            lines.append(f"  MMLU (en)          {self.mmlu_en:.1%}")
            if self.mmlu_en_base is not None:
                drop = self.mmlu_en_base - self.mmlu_en
                verdict = "OK" if drop < 0.05 else "FORGETTING - raise the replay ratio"
                lines.append(f"      base {self.mmlu_en_base:.1%}, "
                             f"drop {drop:+.1%}  [{verdict}]")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                              encoding="utf-8")


# --- Perplexity ---------------------------------------------------------------

def perplexity(model, tokenizer, texts: list[str], seq_len: int = 1024,
               device: str = "cuda") -> float:
    """Token-length-weighted perplexity.

    Weighting by token count matters: averaging per-document perplexities lets a
    handful of short documents dominate and produces a number that moves for reasons
    unrelated to the model.
    """
    import torch

    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=seq_len)["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_nll += out.loss.item() * n
            total_tok += n
    return math.exp(total_nll / total_tok) if total_tok else float("inf")


def perplexity_by_domain(model, tokenizer, val_path: str | Path,
                         max_per_domain: int = 200, device: str = "cuda") -> dict[str, float]:
    by_domain: dict[str, list[str]] = {}
    with Path(val_path).open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            bucket = by_domain.setdefault(d.get("source", "?"), [])
            if len(bucket) < max_per_domain:
                bucket.append(d["text"])
    return {dom: perplexity(model, tokenizer, texts, device=device)
            for dom, texts in by_domain.items() if texts}


# --- Translation --------------------------------------------------------------

def chrf(hypotheses: list[str], references: list[str]) -> float:
    """chrF++ via sacrebleu.

    Character-level F-score, not BLEU. Kabyle is agglutinative -- `yettwanɣa` and
    `yettwanɣan` share no word-level n-gram but are obviously near-identical, and BLEU
    scores that as a total miss.
    """
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        log.warning("sacrebleu not installed; skipping chrF++")
        return 0.0
    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


# --- Generation quality -------------------------------------------------------

def language_purity(generations: list[str], lid=None) -> float:
    """Share of generations identified as Kabyle."""
    if not generations:
        return 0.0
    if lid is None:
        from .filters import LanguageID

        lid = LanguageID()
    return sum(1 for g in generations if lid.predict(g)[0].startswith("kab")) / len(generations)


def oov_rate(generations: list[str], lexicon: set[str]) -> float:
    """Share of generated words absent from the Kabyle dictionary.

    The cheapest honest check that the model is emitting real Kabyle rather than
    plausible-looking noise. Compare against the same figure computed on held-out
    human text -- the corpus itself typically sits around 15-25%, so that is the floor,
    not zero.
    """
    toks = [w.lower() for g in generations for w in words(normalize(g))]
    if not toks:
        return 1.0
    return sum(1 for w in toks if w not in lexicon) / len(toks)


def generate(model, tokenizer, prompts: list[str], *, max_new_tokens: int = 160,
             temperature: float = 0.7, top_p: float = 0.9, chat: bool = False,
             system: str | None = None, device: str = "cuda") -> list[str]:
    import torch

    model.eval()
    out: list[str] = []
    for prompt in prompts:
        if chat and getattr(tokenizer, "chat_template", None):
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
        else:
            text = prompt
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **ids, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        out.append(tokenizer.decode(gen[0][ids["input_ids"].shape[1]:],
                                    skip_special_tokens=True))
    return out


#: Fixed prompts for the manual pass. Small enough to read every answer, broad enough
#: to expose the usual failure modes: repetition, French drift, and confident nonsense.
VIBE_PROMPTS = [
    "Azul! Amek i telliḍ?",                             # greeting
    "D acu-t Tamurt n Leqbayel?",                       # factual, in-domain
    "Sefhem-iyi-d amek i txeddmeḍ tibuqalin n uzemmur.",  # procedural
    "Aru-yi-d tullist tamecṭuḥt ɣef yiwen uqcic d uɣyul-is.",  # creative
    "Suqel ɣer teqbaylit: The weather is beautiful today.",   # translation
    "D acu i d azal n 12 x 8?",                         # reasoning retention
    "Sefhem-iyi-d acuɣer i d-yekkat ugeffur.",          # explanation
    "Fk-iyi-d kraḍ n yizlan iqbayliyen yettwassnen.",   # cultural knowledge
    "Ṣeggem tira: aqchich yura tabrat i yemma-s.",      # orthography
    "Quelle est la capitale de l'Algerie ? Reponds en kabyle.",  # cross-lingual
]
