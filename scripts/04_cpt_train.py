#!/usr/bin/env python
"""Phase 4: continued pretraining -- teach the base model Kabyle without erasing it.

    python scripts/04_cpt_train.py --corpus data/corpus --out out/cpt \
        --base Qwen/Qwen3-1.7B-Base --hub-repo you/kabyllm-cpt

Design decisions and why:

**LoRA, not full fine-tuning.** With ~35M Kabyle tokens against 1.7B parameters, full
fine-tuning overwrites far more than it teaches. High-rank LoRA moves enough weight to
learn a language while leaving the base distribution reachable.

**Embeddings train fully.** The tokens added in Phase 3 have embeddings seeded from a
mean of old pieces; LoRA on a frozen embedding matrix cannot move them. They get their
own much lower learning rate -- embeddings destabilize fast at the adapter LR.

**40% replay.** Batches mix French and English. Without it the model forgets how to
reason, and reasoning is the entire reason for adapting a pretrained model rather than
training from scratch. This is the parameter to raise first if English degrades.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import console, training

log = logging.getLogger("cpt")


def build_dataset(corpus_dir: Path, tokenizer, seq_len: int, replay_frac: float,
                  replay_limit: int, seed: int):
    """Kabyle corpus interleaved with replay data, packed into fixed-length blocks."""
    from datasets import Dataset, interleave_datasets, load_dataset

    from kabyl import sources

    train_file = corpus_dir / "train_weighted.jsonl"
    if not train_file.exists():
        train_file = corpus_dir / "train.jsonl"
    kab = load_dataset("json", data_files=str(train_file), split="train")
    log.info("kabyle: %d documents", len(kab))

    parts, probs = [kab.select_columns(["text"])], [1.0 - replay_frac]
    if replay_frac > 0:
        for spec in sources.REPLAY:
            try:
                ds = load_dataset(
                    spec["repo"], spec.get("config"), split="train", streaming=True
                )
                rows = [{"text": r["text"]} for _, r in
                        zip(range(replay_limit), ds) if r.get("text")]
                if rows:
                    parts.append(Dataset.from_list(rows))
                    probs.append(replay_frac * spec["share"])
                    log.info("replay %s: %d documents", spec["lang"], len(rows))
            except Exception as exc:
                log.warning("replay source %s unavailable: %s",
                            spec["repo"], str(exc)[:100])

    total = sum(probs)
    probs = [p / total for p in probs]
    mixed = (
        interleave_datasets(parts, probabilities=probs, seed=seed,
                            stopping_strategy="all_exhausted")
        if len(parts) > 1 else parts[0]
    )
    log.info("mix: %s", {n: f"{p:.0%}" for n, p in
                         zip(["kabyle"] + [s["lang"] for s in sources.REPLAY], probs)})

    def tok_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=True)

    tokenized = mixed.map(tok_fn, batched=True, remove_columns=mixed.column_names,
                          desc="tokenizing")

    def pack(batch):
        """Concatenate then split into `seq_len` blocks -- no padding waste."""
        ids = [i for seq in batch["input_ids"] for i in seq]
        n = (len(ids) // seq_len) * seq_len
        blocks = [ids[i:i + seq_len] for i in range(0, n, seq_len)]
        return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

    packed = tokenized.map(pack, batched=True, batch_size=1000,
                           remove_columns=tokenized.column_names, desc="packing")
    log.info("packed: %d blocks of %d tokens = %d total",
             len(packed), seq_len, len(packed) * seq_len)
    return packed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--extended", default=None,
                    help="path from 03_tokenizer.py --extend (preferred over --base)")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="out/cpt")
    ap.add_argument("--hub-repo", default=None, help="private repo for resumable state")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA adapters")
    ap.add_argument("--embed-lr", type=float, default=1e-5, help="embedding matrix")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=256)
    ap.add_argument("--replay", type=float, default=0.4)
    ap.add_argument("--replay-limit", type=int, default=60_000)
    ap.add_argument("--session-hours", type=float, default=11.0)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and budget, then exit")
    args = ap.parse_args()

    console.setup()
    print(training.describe_gpu(), "\n")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    model_path = args.extended or args.base
    log.info("base model: %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        log.error("corpus missing: %s -- run scripts/02_build_corpus.py", corpus_dir)
        return 1

    ds = build_dataset(corpus_dir, tokenizer, args.seq_len,
                       args.replay, args.replay_limit, args.seed)
    n_tokens = len(ds) * args.seq_len

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if not training.gpu_report()["bf16"] else torch.bfloat16,
        attn_implementation=training.attn_impl(),
        trust_remote_code=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print("\n" + training.budget_report(n_params, n_tokens, args.epochs) + "\n")

    if args.dry_run:
        log.info("dry run -- nothing trained")
        return 0

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # The new Kabyle vocabulary rows must be trainable in full. A LoRA delta on a
        # frozen embedding table cannot move a token that started as a mean of other
        # tokens, so without this the extended vocabulary never actually learns.
        modules_to_save=["embed_tokens"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Embeddings at the adapter LR diverge; they get their own group an order of
    # magnitude lower.
    decay, no_decay, embeds = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in name:
            embeds.append(p)
        elif p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optim_groups = [
        {"params": decay, "weight_decay": 0.01, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr},
        {"params": embeds, "weight_decay": 0.0, "lr": args.embed_lr},
    ]
    log.info("param groups: %d decay, %d no-decay, %d embedding",
             len(decay), len(no_decay), len(embeds))

    checkpointer = training.HubCheckpointer(args.hub_repo) if args.hub_repo else None
    guard = training.SessionGuard(args.session_hours)
    resume = checkpointer.latest(Path(args.out) / "resume") if checkpointer else None

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=2,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        report_to="none",
        seed=args.seed,
        **training.precision_kwargs(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        optimizers=(torch.optim.AdamW(optim_groups, lr=args.lr), None),
        callbacks=[training.make_stop_callback(guard, checkpointer)],
    )

    log.info("training: %d blocks, %.1fh session budget", len(ds), args.session_hours)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    out = Path(args.out) / "final"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(out)
    (out / "cpt_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log.info("saved -> %s", out)
    log.info("\nNext: scripts/07_eval.py --model %s --base %s", out, args.base)
    log.info("Check English MMLU against the base before going further; a large drop "
             "means the replay ratio is too low.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
