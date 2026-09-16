"""Tokenizer measurement and vocabulary extension for Kabyle.

Kabyle's distinctive letters -- ɣ (U+0263), ɛ (U+025B), ḍ, ṭ, ṣ, ẓ, ḥ, č, ǧ -- are all
two bytes in UTF-8. A byte-level BPE that has never seen Kabyle falls back to raw
bytes for each of them, so a word like `yettwanɣa` costs several times more tokens
than its English equivalent. That is a direct multiplier on training FLOPs *and* a
proportional reduction in usable context length, and it is the reason vocabulary
extension is mandatory rather than an optimization.

`fertility` measures the damage. `extend` repairs it, initializing each new embedding
row as the mean of the base-tokenizer pieces the new token decomposes into -- the
detail that decides whether continued pretraining starts from a usable model or from
a loss spike it never recovers from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .normalize import words

log = logging.getLogger(__name__)

#: English byte-level BPE sits near this. Anything much above ~2.0 for Kabyle means
#: the tokenizer is spending most of its budget on diacritics.
ENGLISH_BASELINE = 1.3

#: Gemma / Llama / Aya are gated: accept their licence on the Hub and run
#: `huggingface-cli login` first, or they are silently skipped.
CANDIDATES = [
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "google/gemma-3-1b-pt",
    "meta-llama/Llama-3.2-1B",
    "HuggingFaceTB/SmolLM3-3B",
    "mistralai/Mistral-7B-v0.3",
    "CohereForAI/aya-expanse-8b",
    # Not a candidate base model -- a reference point. NLLB-200 was trained *with*
    # kab_Latn, so its fertility is roughly the floor vocabulary extension can reach.
    "facebook/nllb-200-distilled-600M",
]

#: Models in CANDIDATES that exist only for comparison and must not be selected.
REFERENCE_ONLY = {"facebook/nllb-200-distilled-600M"}


@dataclass
class Fertility:
    name: str
    tokens_per_word: float
    tokens_per_char: float
    vocab_size: int
    unk_rate: float = 0.0
    #: Tokens/word for words containing a Kabyle-specific letter, divided by the same
    #: figure for words without one. 1.0 means diacritics are free; 2.0 means a word
    #: costs twice as much just for containing ɣ or ḍ. This is the number vocabulary
    #: extension has to move.
    diacritic_penalty: float = 1.0

    def __str__(self) -> str:
        return (
            f"{self.name:<34} {self.tokens_per_word:5.2f} tok/word  "
            f"{self.tokens_per_char:5.3f} tok/char  vocab {self.vocab_size:>7,}  "
            f"diacritic-penalty {self.diacritic_penalty:4.2f}x"
        )


def fertility(tokenizer, text: str, name: str = "") -> Fertility:
    """Measure how expensively `tokenizer` encodes Kabyle."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    toks = words(text)
    n_words = len(toks) or 1

    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk = ids.count(unk_id) / len(ids) if unk_id is not None and ids else 0.0

    # Cost of a diacritic, measured by encoding words individually and comparing the
    # two populations. Inspecting `convert_ids_to_tokens` output directly does not
    # work: byte-level BPE reports pieces in its own byte alphabet (ɣ appears as
    # "É£"), so searching those strings for "ɣ" always finds nothing regardless of
    # how the word was actually segmented.
    special = set("ɣɛḍṭṣẓṛḥčǧƔƐḌṬṢẒṚḤČǦ")
    with_d, without_d = [], []
    for w in toks[:20_000]:
        n = len(tokenizer(w, add_special_tokens=False)["input_ids"]) / max(1, len(w))
        (with_d if special & set(w) else without_d).append(n)
    penalty = (
        (sum(with_d) / len(with_d)) / (sum(without_d) / len(without_d))
        if with_d and without_d else 1.0
    )

    return Fertility(
        name=name or getattr(tokenizer, "name_or_path", "?"),
        tokens_per_word=len(ids) / n_words,
        tokens_per_char=len(ids) / max(1, len(text)),
        vocab_size=len(tokenizer),
        unk_rate=unk,
        diacritic_penalty=penalty,
    )


def benchmark(text: str, names: list[str] | None = None) -> list[Fertility]:
    """Compare candidate base tokenizers on the same Kabyle sample."""
    from transformers import AutoTokenizer

    out: list[Fertility] = []
    for name in names or CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception as exc:
            log.warning("%-34s unavailable: %s", name, str(exc)[:100])
            continue
        f = fertility(tok, text, name)
        out.append(f)
        log.info("%s", f)
    return sorted(out, key=lambda f: f.tokens_per_word)


# --- Vocabulary extension -----------------------------------------------------

def train_kabyle_bpe(
    corpus: list[str], vocab_size: int = 12_000, out_dir: str | Path = "data/interim/kab_bpe"
):
    """Train a Kabyle-only BPE model whose merges will be grafted onto the base."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # A Kabyle-specific piece is only worth a vocabulary slot if it recurs.
        min_frequency=5,
        show_progress=True,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    log.info("trained Kabyle BPE: %d pieces -> %s", tok.get_vocab_size(), out)
    return tok


def select_new_tokens(kab_tok, base_tok, max_new: int = 8_000, min_len: int = 2) -> list[str]:
    """Kabyle pieces worth adding: frequent, non-trivial, and not already covered."""
    base_vocab = set(base_tok.get_vocab())
    ranked = sorted(kab_tok.get_vocab().items(), key=lambda kv: kv[1])  # by merge order
    picked: list[str] = []
    for piece, _ in ranked:
        if piece in base_vocab or len(piece.lstrip("Ġ▁")) < min_len:
            continue
        picked.append(piece)
        if len(picked) >= max_new:
            break
    log.info("selected %d new tokens (of %d candidates)", len(picked), len(ranked))
    return picked


def extend(model, base_tok, new_tokens: list[str], *, mean_init: bool = True):
    """Add `new_tokens` to the tokenizer and grow the model's embeddings.

    Each new row is seeded with the mean of the embeddings of the pieces the base
    tokenizer would have used for that string. Random initialization puts the new
    tokens in a region of embedding space the model has never seen, producing an
    enormous initial loss that continued pretraining spends most of its budget
    undoing. Mean init starts them somewhere the model already understands.
    """
    import torch

    n_added = base_tok.add_tokens(new_tokens)
    if n_added == 0:
        log.warning("no tokens added; vocabulary unchanged")
        return model, base_tok

    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(base_tok))
    emb = model.get_input_embeddings().weight.data

    if mean_init:
        with torch.no_grad():
            for tok_str in new_tokens:
                new_id = base_tok.convert_tokens_to_ids(tok_str)
                if new_id is None or new_id < old_size:
                    continue
                surface = base_tok.convert_tokens_to_string([tok_str])
                parts = base_tok(surface, add_special_tokens=False)["input_ids"]
                parts = [p for p in parts if p < old_size]
                if parts:
                    emb[new_id] = emb[parts].mean(dim=0)

        out_emb = model.get_output_embeddings()
        tied = out_emb is not None and out_emb.weight.data_ptr() == emb.data_ptr()
        if out_emb is not None and not tied:
            # Untied head: seed it the same way, or the model emits noise for every
            # new token no matter how good the input embedding is.
            ow = out_emb.weight.data
            with torch.no_grad():
                for tok_str in new_tokens:
                    new_id = base_tok.convert_tokens_to_ids(tok_str)
                    if new_id is None or new_id < old_size:
                        continue
                    surface = base_tok.convert_tokens_to_string([tok_str])
                    parts = [p for p in base_tok(surface, add_special_tokens=False)["input_ids"]
                             if p < old_size]
                    if parts:
                        ow[new_id] = ow[parts].mean(dim=0)

    log.info("vocabulary %d -> %d (+%d), mean_init=%s",
             old_size, len(base_tok), n_added, mean_init)
    return model, base_tok


def save_report(results: list[Fertility], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8"
    )
