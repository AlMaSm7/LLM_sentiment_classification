"""RUN FILE ONLY ONCE"""

import os

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# model settings
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

# change to your username!!
CACHE = "/scratch/s6865704/llm/hf_cache"

os.environ["HF_HOME"] = CACHE
os.environ["HF_HUB_CACHE"] = f"{CACHE}/hub"
os.environ["HF_XET_CACHE"] = f"{CACHE}/xet"


# check cache before downloading
for model in MODELS:
    try:
        path = snapshot_download(repo_id=model, local_files_only=True)
        print(f"Already cached: {model}")

    except LocalEntryNotFoundError:
        print(f"Downloading: {model}")
        path = snapshot_download(repo_id=model)

    print(f"Model path: {path}")
