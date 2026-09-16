"""Kaggle notebook source for Phases 3-4 (tokenizer extension + continued pretraining).

Paste each `# %% [markdown]` / `# %%` block into a Kaggle cell, or run
`jupytext --to notebook notebooks/kaggle_cpt.py`.

Settings > Accelerator > **GPU T4 x2**, and Settings > Internet > **On**.
Add your HF token as a Kaggle Secret named `HF_TOKEN` so checkpoints can be pushed --
without it a session that hits the 12-hour cap loses everything since the last save.
"""

# %% [markdown]
# # KabyLLM - continued pretraining
#
# Adapts a multilingual base model to Kabyle. The Kabyle corpus supplies the
# *language*; the base model already supplies the reasoning. That split is the whole
# reason this is feasible on ~35M tokens when training from scratch would need
# thousands of times more.

# %%
# !pip install -q -U transformers accelerate peft bitsandbytes datasets sentencepiece
# !pip install -q -U "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

# %%
import os
import subprocess
import sys

REPO = "https://github.com/YOURNAME/kabyllm"  # or upload the repo as a Kaggle Dataset
if not os.path.exists("kabyllm"):
    subprocess.run(["git", "clone", "--depth", "1", REPO], check=False)
sys.path.insert(0, "kabyllm")

from kaggle_secrets import UserSecretsClient  # noqa: E402

try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded - checkpoints will survive the session")
except Exception:
    print("WARNING: no HF_TOKEN secret. A 12h timeout will lose all progress.")

# %%
from kabyl import training  # noqa: E402

print(training.describe_gpu())
# Expect: 2x Tesla T4 (SM 7.5) -> fp16 + sdpa. bf16 and FlashAttention-2 need Ampere+.

# %% [markdown]
# ## 1. Corpus
#
# Build it here, or (much faster on repeat runs) attach the output of a previous run
# as a Kaggle Dataset and skip straight to the tokenizer step.

# %%
# !cd kabyllm && python scripts/02_build_corpus.py --out /kaggle/working/corpus

# %% [markdown]
# ## 2. Tokenizer
#
# Measure fertility first. Qwen3 encodes Kabyle at ~2.77 tokens/word against ~1.3 for
# English -- every training step costs roughly twice what the same content costs in
# English, and effective context halves. Vocabulary extension is what recovers that.

# %%
# !cd kabyllm && python scripts/03_tokenizer.py --bench --corpus /kaggle/working/corpus/train.jsonl

# %%
# !cd kabyllm && python scripts/03_tokenizer.py --extend \
#     --corpus /kaggle/working/corpus/train.jsonl \
#     --base Qwen/Qwen3-1.7B-Base --new 8000 --out /kaggle/working

# %% [markdown]
# ## 3. Budget check
#
# Confirm the run fits the weekly quota before starting it.

# %%
# !cd kabyllm && python scripts/04_cpt_train.py --dry-run \
#     --extended /kaggle/working/extended --corpus /kaggle/working/corpus

# %% [markdown]
# ## 4. Continued pretraining
#
# `--session-hours 11` stops voluntarily before Kaggle's 12-hour kill, so the final
# checkpoint is always written. Rerunning the same cell resumes from the Hub.
#
# `--replay 0.4` keeps 40% French/English in every batch. It is the difference between
# a bilingual assistant and a model that has forgotten how to reason.

# %%
# !cd kabyllm && python scripts/04_cpt_train.py \
#     --extended /kaggle/working/extended \
#     --corpus /kaggle/working/corpus \
#     --out /kaggle/working/cpt \
#     --hub-repo YOURNAME/kabyllm-cpt \
#     --replay 0.4 --epochs 2 --session-hours 11

# %% [markdown]
# ## 5. Gate: did it forget?
#
# If English MMLU dropped more than ~5 points, stop and rerun Phase 4 with a higher
# replay ratio. Do not proceed to SFT on a model that has lost its reasoning -- SFT
# will not give it back.

# %%
# !cd kabyllm && python scripts/07_eval.py \
#     --model /kaggle/working/cpt/final \
#     --base Qwen/Qwen3-1.7B-Base \
#     --corpus /kaggle/working/corpus --vibe
