#!/usr/bin/env python
"""Phase 7: evaluate a Kabyle model.

    python scripts/07_eval.py --model out/cpt/final --base Qwen/Qwen3-1.7B-Base
    python scripts/07_eval.py --model out/sft/final --chat --vibe

`--base` enables the forgetting check, which is the gate on continued pretraining: if
English MMLU has fallen more than ~5 points, the replay ratio was too low and the
model has traded reasoning for Kabyle. Rerun Phase 4 with `--replay 0.5` rather than
continuing to SFT.

`--vibe` prints answers to ten fixed prompts. Read them. You are a native speaker and
no metric here can tell you whether the Kabyle sounds right.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import console, evaluate as ev, lexicon
from kabyl import tokenizer as tk

log = logging.getLogger("eval")


def mmlu_en(model, tokenizer, n: int = 400, device: str = "cuda") -> float | None:
    """Accuracy on an English MMLU sample, by comparing A/B/C/D option logprobs."""
    import torch
    from datasets import load_dataset

    try:
        ds = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
    except Exception as exc:
        log.warning("MMLU unavailable: %s", str(exc)[:100])
        return None

    letters = ["A", "B", "C", "D"]
    correct = 0
    model.eval()
    for row in ds:
        prompt = row["question"] + "\n" + "\n".join(
            f"{l}. {c}" for l, c in zip(letters, row["choices"])
        ) + "\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        scores = []
        for l in letters:
            tid = tokenizer.encode(" " + l, add_special_tokens=False)
            scores.append(logits[tid[0]].item() if tid else -1e9)
        if int(max(range(4), key=lambda i: scores[i])) == int(row["answer"]):
            correct += 1
    return correct / len(ds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default=None, help="for the forgetting comparison")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--chat", action="store_true", help="apply the chat template")
    ap.add_argument("--vibe", action="store_true", help="print the manual prompt set")
    ap.add_argument("--skip-mmlu", action="store_true")
    ap.add_argument("--max-per-domain", type=int, default=150)
    args = ap.parse_args()

    console.setup()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    res = ev.EvalResult(model=args.model)
    lex = lexicon.load()

    val = Path(args.corpus) / "val.jsonl"
    if val.exists():
        sample = "\n".join(json.loads(l)["text"] for _, l in
                           zip(range(400), val.open(encoding="utf-8")))
        res.fertility = tk.fertility(tokenizer, sample).tokens_per_word
        log.info("computing perplexity by domain...")
        res.perplexity = ev.perplexity_by_domain(
            model, tokenizer, val, args.max_per_domain, device
        )
    else:
        log.warning("no val split at %s; skipping perplexity", val)

    log.info("generating for language/OOV checks...")
    gens = ev.generate(model, tokenizer, ev.VIBE_PROMPTS, chat=args.chat,
                       system=None, device=device)
    res.language_purity = ev.language_purity(gens)
    res.oov_rate = ev.oov_rate(gens, lex)
    res.samples = [{"prompt": p, "generation": g} for p, g in zip(ev.VIBE_PROMPTS, gens)]

    # Reference OOV on real human text: the floor this metric can reach. Held-out
    # Kabyle typically sits at 15-25%, so the model's figure should be read against
    # that, never against zero.
    if val.exists():
        human = [json.loads(l)["text"] for _, l in zip(range(200), val.open(encoding="utf-8"))]
        print(f"reference OOV on held-out human text: {ev.oov_rate(human, lex):.1%}")

    if not args.skip_mmlu:
        log.info("MMLU (English) on the adapted model...")
        res.mmlu_en = mmlu_en(model, tokenizer, device=device)
        if args.base:
            del model
            torch.cuda.empty_cache()
            log.info("MMLU (English) on the untouched base...")
            btok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
            bmodel = AutoModelForCausalLM.from_pretrained(
                args.base,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(device)
            res.mmlu_en_base = mmlu_en(bmodel, btok, device=device)

    print("\n" + "=" * 70)
    print(res.summary())
    print("=" * 70)

    if args.vibe:
        print("\n" + "=" * 70)
        print("MANUAL REVIEW - read every answer")
        print("=" * 70)
        for s in res.samples:
            print(f"\nQ: {s['prompt']}")
            print(f"A: {s['generation'][:400]}")

    out = Path(args.out) / (Path(args.model).name + ".json")
    res.save(out)
    log.info("\nsaved -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
