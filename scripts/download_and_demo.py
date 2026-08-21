"""Download a Hugging Face checkpoint and launch the Gradio demo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_checkpoint(repo_id: str, filename: str, revision: str | None = None) -> Path:
    """Download one checkpoint from Hugging Face Hub and return its cached path."""
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


def main() -> None:
    """Parse download/demo options, load weights, and start Gradio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=os.environ.get("HF_MODEL_REPO_ID"),
        help="Hugging Face model repository, or set HF_MODEL_REPO_ID",
    )
    parser.add_argument("--filename", default="best.pt", help="Checkpoint filename in repository")
    parser.add_argument("--revision", help="Optional branch, tag, or commit SHA")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not args.repo_id:
        parser.error("repo_id is required (or set HF_MODEL_REPO_ID)")

    checkpoint = download_checkpoint(args.repo_id, args.filename, args.revision)

    import app

    app.CHECKPOINT_PATH = checkpoint
    app.get_detector(checkpoint)  # fail before opening server
    app.build_demo().launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
