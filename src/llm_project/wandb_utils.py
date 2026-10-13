from typing import Any


def init_wandb(
    config: dict[str, Any],
    run_name: str,
):
    """Start W&B if it is enabled."""

    if not config["wandb"]["enabled"]:
        return None

    import wandb

    return wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        mode=config["wandb"].get(
            "mode",
            "online",
        ),
        name=run_name,
        config=config,
    )


def log_metrics(
    run,
    metrics: dict[str, Any],
) -> None:
    """Log final evaluation metrics."""

    if run is None:
        return

    values = {}

    for language, language_metrics in metrics.items():
        for name, value in language_metrics.items():
            if name == "per_emotion_f1":
                for emotion, score in value.items():
                    values[f"{language}/f1_{emotion}"] = score

            else:
                values[f"{language}/{name}"] = value

    run.log(values)


def finish_wandb(
    run,
) -> None:
    """Finish W&B if a run is active."""

    if run is not None:
        run.finish()
