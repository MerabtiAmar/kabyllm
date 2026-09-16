#!/usr/bin/env python
"""Phase 8: merge adapters and export to GGUF for local inference.

    python scripts/08_export_gguf.py --model out/sft/final --base Qwen/Qwen3-1.7B-Base

A 1.7B model at Q4_K_M is roughly 1.1 GB and runs on a laptop CPU, which matters for a
language whose speakers are not all sitting next to an A100.

Requires llama.cpp checked out somewhere; pass `--llama-cpp` or let the script clone it.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabyl import console

log = logging.getLogger("export")


def merge(adapter: str, base: str, out: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading base %s", base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, trust_remote_code=True, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    # The adapter was trained after vocabulary extension, so the base must be grown to
    # match before the adapter will load.
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        log.info("resizing embeddings %d -> %d for the extended vocabulary",
                 model.get_input_embeddings().weight.shape[0], len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    log.info("applying adapter %s", adapter)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    log.info("merged -> %s", out)
    return out


def to_gguf(src: Path, llama_cpp: Path, quant: str) -> None:
    convert = llama_cpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        log.error("convert_hf_to_gguf.py not found under %s", llama_cpp)
        log.error("clone it:  git clone https://github.com/ggerganov/llama.cpp")
        return
    f16 = src.parent / f"{src.name}-f16.gguf"
    subprocess.run(
        [sys.executable, str(convert), str(src), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    log.info("f16 GGUF -> %s (%.1f GB)", f16, f16.stat().st_size / 1e9)

    quantize = next(
        (exe for d in (llama_cpp / "build" / "bin", llama_cpp)
         for exe in d.glob("llama-quantize*") if exe.is_file()),
        None,
    )
    if quantize is None:
        log.warning("llama-quantize not built; stopping at f16. Build llama.cpp to "
                    "produce the %s file.", quant)
        return
    qout = src.parent / f"{src.name}-{quant}.gguf"
    subprocess.run([str(quantize), str(f16), str(qout), quant], check=True)
    log.info("quantized -> %s (%.2f GB)", qout, qout.stat().st_size / 1e9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="adapter directory")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--out", default="out/export")
    ap.add_argument("--llama-cpp", default="llama.cpp")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--skip-gguf", action="store_true")
    args = ap.parse_args()

    console.setup()
    merged = merge(args.model, args.base, Path(args.out) / "merged")
    if not args.skip_gguf:
        to_gguf(merged, Path(args.llama_cpp), args.quant)

    print("\nRun it locally:")
    print(f"  llama-cli -m {args.out}/merged-{args.quant}.gguf -p 'Azul! Amek i telliḍ?' -cnv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
