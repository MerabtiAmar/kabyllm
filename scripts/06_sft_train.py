#!/usr/bin/env python
"""Phase 6: instruction-tune the Kabyle-adapted model.

    python scripts/06_sft_train.py --model out/cpt/final --data data/sft --out out/sft

Loss is masked on the prompt: the model is graded on its answers, never on
reproducing the questions. Without masking, a dataset whose instructions are
templated (as most of ours are) teaches the model to recite templates.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import console, training

log = logging.getLogger("sft-train")

#: Fallback for base models that ship no chat template (many -Base checkpoints).
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="out/cpt/final")
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--out", default="out/sft")
    ap.add_argument("--hub-repo", default=None)
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--session-hours", type=float, default=11.0)
    ap.add_argument("--merge", action="store_true", help="merge adapters and save")
    args = ap.parse_args()

    console.setup()
    print(training.describe_gpu(), "\n")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        log.info("no chat template on the base; installing ChatML")
        tokenizer.chat_template = CHATML

    data_dir = Path(args.data)
    ds = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )
    log.info("sft data: %d train, %d val", len(ds["train"]), len(ds["validation"]))

    def encode(row):
        msgs = row["messages"]
        full = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                               add_generation_prompt=True)
        full_ids = tokenizer(full, truncation=True, max_length=args.seq_len)["input_ids"]
        prompt_ids = tokenizer(prompt, truncation=True,
                               max_length=args.seq_len)["input_ids"]
        labels = list(full_ids)
        # Mask the prompt: grade the answer, not the question.
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}

    ds = ds.map(encode, remove_columns=ds["train"].column_names, desc="encoding")
    ds = ds.filter(lambda r: any(l != -100 for l in r["labels"]),
                   desc="dropping fully-masked rows")

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            k = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad] * k)
            out["labels"].append(b["labels"] + [-100] * k)
            out["attention_mask"].append(b["attention_mask"] + [0] * k)
        return {k: torch.tensor(v) for k, v in out.items()}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if not training.gpu_report()["bf16"] else torch.bfloat16,
        attn_implementation=training.attn_impl(),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    checkpointer = training.HubCheckpointer(args.hub_repo) if args.hub_repo else None
    guard = training.SessionGuard(args.session_hours)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=250,
            save_steps=250,
            save_total_limit=2,
            max_grad_norm=1.0,
            gradient_checkpointing=True,
            optim="adamw_bnb_8bit",
            report_to="none",
            **training.precision_kwargs(),
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate,
        callbacks=[training.make_stop_callback(guard, checkpointer)],
    )
    trainer.train()

    out = Path(args.out) / "final"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(out)
    log.info("saved -> %s", out)

    if args.merge:
        log.info("merging adapters for inference/export")
        merged = model.merge_and_unload()
        mpath = Path(args.out) / "merged"
        merged.save_pretrained(mpath, safe_serialization=True)
        tokenizer.save_pretrained(mpath)
        log.info("merged -> %s (ready for scripts/08_export_gguf.py)", mpath)

    log.info("\nNext: scripts/07_eval.py --model %s --chat", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
