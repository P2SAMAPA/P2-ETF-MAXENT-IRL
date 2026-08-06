"""
push_results.py  —  HuggingFace upload wrapper for MaxEnt-IRL results.
"""

import os
from huggingface_hub import HfApi
import config


def upload_results(file1_path: str, file2_path: str, hf_token: str = None):
    """Upload the two JSON result files to HuggingFace."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required")

    api = HfApi(token=token)

    for path in [file1_path, file2_path]:
        if not os.path.exists(path):
            print(f"⚠️  Warning: {path} not found, skipping")
            continue

        filename = os.path.basename(path)
        print(f"   Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=config.RESULTS_REPO,
            token=token,
            repo_type="dataset"
        )

    print("✅ Upload complete!")
