"""
Shared environment setup for IR roadmap notebooks.

Idempotent: call setup_env() once at the top of any notebook.
- Locally: no-op beyond returning False for IN_COLAB.
- On Colab: mounts Drive, redirects ir_datasets + HuggingFace caches into Drive
  so downloads survive session disconnects.

The git clone + pip install must run BEFORE this module can be imported
on Colab (chicken/egg). See notebooks/00_setup.ipynb for the canonical
2-cell bootstrap pattern.
"""
import os
import sys

REPO_URL = "https://github.com/ZiaadNegm/Information-Retreival.git"
DRIVE_CACHE_ROOT = "/content/drive/MyDrive/ir-roadmap-cache"


def is_colab() -> bool:
    return "google.colab" in sys.modules


def setup_env(verbose: bool = True) -> bool:
    """Mount Drive (Colab only) and point ir_datasets / HuggingFace caches at it.

    Returns True if running on Colab, False locally. Safe to call multiple times.
    """
    in_colab = is_colab()
    if in_colab:
        from google.colab import drive  # type: ignore

        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        os.makedirs(f"{DRIVE_CACHE_ROOT}/ir_datasets", exist_ok=True)
        os.makedirs(f"{DRIVE_CACHE_ROOT}/huggingface", exist_ok=True)
        os.environ["IR_DATASETS_HOME"] = f"{DRIVE_CACHE_ROOT}/ir_datasets"
        os.environ["HF_HOME"] = f"{DRIVE_CACHE_ROOT}/huggingface"
        if verbose:
            print(f"[bootstrap] Colab — caches at {DRIVE_CACHE_ROOT}")
    elif verbose:
        print("[bootstrap] local — caches at ~/.ir_datasets/ and ~/.cache/huggingface/")
    return in_colab
