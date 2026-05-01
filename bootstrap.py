"""
Shared environment setup for IR roadmap notebooks.

Idempotent: call setup_env() once at the top of any notebook.
- Locally: no-op beyond returning False for IN_COLAB.
- On Colab: tries to mount Drive and redirect ir_datasets / HuggingFace caches
  into Drive so downloads survive session disconnects. If mount fails (the
  OAuth dialog is finicky in PyCharm-managed Colab), prints a warning and
  falls back to the ephemeral VM disk — data won't persist that session,
  but nothing crashes.

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
    if not in_colab:
        if verbose:
            print("[bootstrap] local — caches at ~/.ir_datasets/ and ~/.cache/huggingface/")
        return False

    from google.colab import drive  # type: ignore

    mounted = os.path.ismount("/content/drive")
    if not mounted:
        try:
            drive.mount("/content/drive")
            mounted = True
        except Exception as exc:
            print(f"[bootstrap] WARNING: Drive mount failed ({exc}).")
            print("[bootstrap] Falling back to ephemeral VM cache for this session.")
            print("[bootstrap] Data will NOT persist after disconnect.")
            print("[bootstrap] To fix: re-run this cell and complete the Google OAuth dialog,")
            print("[bootstrap] or run this notebook in browser Colab where the dialog is reliable.")

    if mounted:
        os.makedirs(f"{DRIVE_CACHE_ROOT}/ir_datasets", exist_ok=True)
        os.makedirs(f"{DRIVE_CACHE_ROOT}/huggingface", exist_ok=True)
        os.environ["IR_DATASETS_HOME"] = f"{DRIVE_CACHE_ROOT}/ir_datasets"
        os.environ["HF_HOME"] = f"{DRIVE_CACHE_ROOT}/huggingface"
        if verbose:
            print(f"[bootstrap] Colab — caches at {DRIVE_CACHE_ROOT}")
    else:
        # Fallback: use the VM's default cache locations (ephemeral).
        if verbose:
            print("[bootstrap] Colab (ephemeral) — caches at /root/.ir_datasets and /root/.cache/huggingface")

    return True
