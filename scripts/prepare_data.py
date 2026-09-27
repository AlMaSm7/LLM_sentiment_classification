from pathlib import Path

from llm_project.data_preparation.data_pipeline import (
    run_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


if __name__ == "__main__":
    run_pipeline(project_root=PROJECT_ROOT, config_path=CONFIG_PATH)
