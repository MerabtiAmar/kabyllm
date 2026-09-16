"""Shared training infrastructure for Kaggle.

Two constraints shape everything here and neither is negotiable:

**T4 is Turing (SM 7.5).** No bfloat16 and no FlashAttention-2 (which needs SM 8.0+).
Training runs in fp16 with a gradient scaler and PyTorch's `sdpa` attention. Passing
`bf16=True` or `attn_implementation="flash_attention_2"` on Kaggle's default GPU fails
at import or silently falls back.

**Sessions are capped at 12 hours** with a 30 h/week quota. The v0 run lost 8,207 steps
of progress to an interrupted session because nothing was checkpointed anywhere
durable. `HubCheckpointer` writes optimizer, scheduler, and dataloader position to a
private Hub repo so the next session resumes rather than restarts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


def gpu_report() -> dict:
    """Detect the accelerator and the precision it actually supports."""
    import torch

    info = {
        "cuda": torch.cuda.is_available(),
        "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "names": [],
        "capability": None,
        "bf16": False,
        "flash_attn": False,
    }
    if not info["cuda"]:
        return info
    info["names"] = [torch.cuda.get_device_name(i) for i in range(info["n_gpu"])]
    major, minor = torch.cuda.get_device_capability(0)
    info["capability"] = f"{major}.{minor}"
    info["bf16"] = major >= 8
    info["flash_attn"] = major >= 8
    return info


def describe_gpu() -> str:
    g = gpu_report()
    if not g["cuda"]:
        return "no CUDA device -- training will not run here"
    lines = [f"{g['n_gpu']}x {g['names'][0]}  (SM {g['capability']})"]
    if not g["bf16"]:
        lines.append("  SM < 8.0: using fp16 + GradScaler (bf16 unavailable on Turing)")
        lines.append("  SM < 8.0: using sdpa attention (FlashAttention-2 needs Ampere+)")
    else:
        lines.append("  bf16 and FlashAttention-2 available")
    return "\n".join(lines)


def precision_kwargs() -> dict:
    """TrainingArguments precision flags appropriate to the detected GPU."""
    g = gpu_report()
    if not g["cuda"]:
        return {}
    return {"bf16": True} if g["bf16"] else {"fp16": True}


def attn_impl() -> str:
    return "flash_attention_2" if gpu_report()["flash_attn"] else "sdpa"


# --- Cross-session checkpointing ---------------------------------------------

@dataclass
class HubCheckpointer:
    """Persist training state to a private Hub repo so a killed session can resume.

    Kaggle's `/kaggle/working` does not reliably survive a session ending, and its
    20 GB cap fills quickly with optimizer states. The Hub is the only durable store
    available for free, so checkpoints go there and the next session pulls them back.
    """

    repo_id: str
    every_steps: int = 500
    keep_last: int = 2
    private: bool = True
    token: str | None = None
    _last: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.token = self.token or os.environ.get("HF_TOKEN")

    def _api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def ensure_repo(self) -> None:
        from huggingface_hub import create_repo

        create_repo(self.repo_id, private=self.private, exist_ok=True, token=self.token)

    def push(self, local_dir: str | Path, step: int) -> None:
        self.ensure_repo()
        api = self._api()
        t0 = time.time()
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=self.repo_id,
            commit_message=f"checkpoint step {step}",
        )
        log.info("checkpoint step %d pushed to %s (%.0fs)",
                 step, self.repo_id, time.time() - t0)

    def latest(self, local_dir: str | Path) -> Path | None:
        """Download the most recent checkpoint, or None if there is none."""
        from huggingface_hub import snapshot_download

        try:
            path = snapshot_download(
                repo_id=self.repo_id, local_dir=str(local_dir), token=self.token
            )
        except Exception as exc:
            log.info("no resumable checkpoint (%s)", str(exc)[:80])
            return None
        p = Path(path)
        if not (p / "trainer_state.json").exists():
            return None
        state = json.loads((p / "trainer_state.json").read_text(encoding="utf-8"))
        log.info("resuming from step %s", state.get("global_step"))
        return p


class SessionGuard:
    """Stop cleanly before Kaggle kills the session.

    A hard kill loses whatever happened since the last checkpoint. Stopping
    voluntarily at the budget boundary means the final checkpoint is always written.
    """

    def __init__(self, budget_hours: float = 11.0):
        self.deadline = time.time() + budget_hours * 3600
        self.budget_hours = budget_hours

    @property
    def expired(self) -> bool:
        return time.time() >= self.deadline

    @property
    def remaining_h(self) -> float:
        return max(0.0, (self.deadline - time.time()) / 3600)


def make_stop_callback(guard: SessionGuard, checkpointer: HubCheckpointer | None = None):
    """TrainerCallback that halts training when the session budget runs out."""
    from transformers import TrainerCallback

    class _Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if guard.expired:
                log.warning(
                    "session budget of %.1fh reached at step %d - stopping cleanly "
                    "so the checkpoint is written; rerun to resume",
                    guard.budget_hours, state.global_step,
                )
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            if checkpointer is not None:
                ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if ckpt.exists():
                    try:
                        checkpointer.push(ckpt, state.global_step)
                    except Exception as exc:
                        log.warning("checkpoint push failed (training continues): %s",
                                    str(exc)[:120])
            return control

    return _Stop()


# --- Budgeting ----------------------------------------------------------------

def estimate_hours(
    n_params: float, n_tokens: float, *, tflops: float = 35.0, mfu: float = 1.0
) -> float:
    """Wall-clock estimate from the 6ND rule.

    `tflops` defaults to a realistic sustained figure for 2xT4 with LoRA and gradient
    checkpointing, not the 130 TFLOPS marketing number for the pair.
    """
    flops = 6.0 * n_params * n_tokens
    return flops / (tflops * 1e12 * mfu) / 3600


def budget_report(n_params: float, n_tokens: float, epochs: float = 1.0) -> str:
    total = n_tokens * epochs
    hours = estimate_hours(n_params, total)
    return (
        f"params {n_params / 1e9:.2f}B x tokens {total / 1e6:.0f}M "
        f"= {6 * n_params * total:.2e} FLOPs\n"
        f"estimated {hours:.1f} GPU-hours on 2xT4 "
        f"({math.ceil(hours / 11)} Kaggle session(s), "
        f"{hours / 30 * 100:.0f}% of a weekly quota)"
    )
