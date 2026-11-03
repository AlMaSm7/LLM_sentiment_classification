from __future__ import annotations

import math
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal


@contextmanager
def wandb_run(
    config: dict[str, Any],
    *,
    language: str,
    condition: str,
) -> Generator[Any, None, None]:
    """Start a run before training and close it, including on an exception."""

    if not config.get("use_wandb", False):
        yield None
        return

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install W&B with `uv add wandb`") from exc

    # disable automatic uploads of large model checkpoints
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    output_dir = Path("outputs/wandb").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_value = os.environ.get("WANDB_MODE") or config.get("wandb_mode", "online")

    if mode_value not in ("online", "offline"):
        raise ValueError(f"Invalid W&B mode: {mode_value}")

    mode: Literal["online", "offline"] = "offline" if mode_value == "offline" else "online"

    run = wandb.init(
        project=config.get("wandb_project", "peft_emotion_classification"),
        mode=mode,
        name=f"{language}-{condition}-{config['experiment']['name']}",
        group=f"{language}-peft",
        job_type="train",
        dir=str(output_dir),
        config={
            "language": language,
            "condition": condition,
            "experiment": config.get("experiment", {}),
            "model": config.get("model", {}),
            "peft": config.get("peft", {}),
            "training": config.get("training", {}),
        },
    )
    try:
        yield run
    finally:
        run.finish()


def log_evaluation(run: Any | None, metrics: dict[str, Any], *, split: str) -> None:
    """Log numerical evaluation scores without logging full prediction text."""

    if run is None:
        return

    values: dict[str, float] = {}

    def collect(data: Any, prefix: str) -> None:
        if isinstance(data, dict):
            for name, value in data.items():
                collect(value, f"{prefix}/{name}")
        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            if math.isfinite(float(data)):
                values[prefix] = float(data)

    collect(metrics, split)
    if values:
        run.log(values)
