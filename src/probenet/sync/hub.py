"""HuggingFace Hub coordination primitives for distributed training.

Provides checkpoint polling, dataset upload/download with atomic ``_complete``
markers, and rollout ID naming.  Adapted from lehome_solution's ``hf_sync.py``.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

ROLLOUT_ID_RE = r"^rollout_(\d+)_(.+?)_(\d{8}_\d{6})(?:_(.+))?$"


# ── Checkpoint helpers ───────────────────────────────────────────────────────


def get_latest_model_step(hf_model_repo: str) -> int | None:
    """Poll HF Hub for the most recent checkpoint step number."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return None

    try:
        path = hf_hub_download(
            hf_model_repo,
            filename="latest/assets/step_info.json",
            repo_type="model",
            force_download=True,
        )
        with open(path) as f:
            return json.load(f).get("step")
    except Exception:
        pass

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(hf_model_repo, repo_type="model"))
        steps = [int(e.path) for e in entries if e.path.isdigit() if hasattr(e, "path")]
        return max(steps) if steps else None
    except Exception:
        logger.debug("Could not list repo tree for %s", hf_model_repo, exc_info=True)
        return None


def download_latest_checkpoint(
    hf_model_repo: str,
    local_dir: str,
    *,
    include_train_state: bool = False,
) -> tuple[str, int] | None:
    """Download the latest checkpoint from HF Hub.

    Returns ``(checkpoint_dir, step)`` or ``None`` if nothing new.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    try:
        local_base = Path(local_dir)
        local_base.mkdir(parents=True, exist_ok=True)

        remote_step = get_latest_model_step(hf_model_repo)
        if remote_step is None:
            return None

        step_dir = local_base / f"step_{remote_step}"
        step_info = step_dir / "assets" / "step_info.json"
        if step_info.exists():
            logger.info("Checkpoint step %d already cached", remote_step)
            return str(step_dir), remote_step

        tmp_dir = local_base / f"_downloading_step_{remote_step}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        patterns = ["latest/params/**", "latest/assets/**"]
        if include_train_state:
            patterns.append("latest/train_state/**")

        logger.info("Downloading checkpoint step %d", remote_step)
        snapshot_download(
            hf_model_repo,
            allow_patterns=patterns,
            local_dir=str(tmp_dir),
            repo_type="model",
        )

        tmp_latest = tmp_dir / "latest"
        if not tmp_latest.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        tmp_latest.rename(step_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # keep only latest 3
        step_dirs = sorted(
            [d for d in local_base.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
            reverse=True,
        )
        for d in step_dirs[3:]:
            shutil.rmtree(d, ignore_errors=True)

        return str(step_dir), remote_step
    except Exception as e:
        logger.warning("Checkpoint download failed: %s", e)
        return None


# ── Dataset helpers ──────────────────────────────────────────────────────────


def list_available_rollouts(hf_dataset_repo: str) -> list[dict]:
    """Return metadata for all completed rollout datasets on HF Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(hf_dataset_repo, repo_type="dataset", recursive=False))
    except Exception as e:
        logger.warning("Failed to list %s: %s", hf_dataset_repo, e)
        return []

    rollout_dirs = [
        getattr(e, "path", "") for e in entries if getattr(e, "path", "").startswith("rollout_")
    ]
    if not rollout_dirs:
        return []

    complete_paths = [f"{d}/_complete" for d in rollout_dirs]
    try:
        path_infos = api.get_paths_info(hf_dataset_repo, complete_paths, repo_type="dataset")
    except Exception as e:
        logger.warning("_complete check failed: %s", e)
        return []

    complete_set = {info.path.rsplit("/", 1)[0] for info in path_infos}

    rollouts = []
    for path in rollout_dirs:
        if path not in complete_set:
            continue
        info = _parse_rollout_id(path)
        if info:
            rollouts.append(info)
    return rollouts


def download_rollout_dataset(hf_dataset_repo: str, rollout_id: str, local_dir: str) -> str | None:
    """Download a single rollout's eval_dataset from HF Hub."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return None

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(hf_dataset_repo, path_in_repo=rollout_id, recursive=True, repo_type="dataset"))
    except Exception as e:
        logger.warning("Failed to list rollout %s: %s", rollout_id, e)
        return None

    file_paths = [e.path for e in entries if hasattr(e, "size")]
    complete_marker = f"{rollout_id}/_complete"
    if complete_marker not in file_paths:
        return None

    downloadable = [p for p in file_paths if not p.endswith("/_complete")]
    try:
        for fpath in downloadable:
            hf_hub_download(hf_dataset_repo, filename=fpath, local_dir=local_dir, repo_type="dataset")
        eval_ds = Path(local_dir) / rollout_id / "eval_dataset"
        if eval_ds.exists():
            return str(eval_ds)
        rollout_dir = Path(local_dir) / rollout_id
        if any(rollout_dir.glob("data/chunk-*/*.parquet")):
            return str(rollout_dir)
        return None
    except Exception as e:
        logger.warning("Rollout download failed %s: %s", rollout_id, e)
        return None


def upload_rollout_dataset(eval_run_dir: str, rollout_id: str, hf_dataset_repo: str) -> bool:
    """Upload a rollout's eval dataset to HF Hub with atomic ``_complete`` marker."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    api = HfApi()
    run_path = Path(eval_run_dir)
    if not run_path.exists():
        return False

    api.create_repo(hf_dataset_repo, repo_type="dataset", exist_ok=True)
    try:
        api.upload_folder(
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            folder_path=str(run_path),
            path_in_repo=rollout_id,
            commit_message=f"Rollout: {rollout_id}",
        )
        api.upload_file(
            path_or_fileobj=io.BytesIO(b"ok"),
            path_in_repo=f"{rollout_id}/_complete",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"Complete: {rollout_id}",
        )
        logger.info("Uploaded rollout %s to %s", rollout_id, hf_dataset_repo)
        return True
    except Exception as e:
        logger.warning("Rollout upload failed %s: %s", rollout_id, e)
        return False


def upload_checkpoint_step(checkpoint_dir: str, step: int, hf_model_repo: str) -> bool:
    """Upload a training checkpoint as ``latest/`` and a numbered directory."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    api = HfApi()
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return False

    api.create_repo(hf_model_repo, repo_type="model", exist_ok=True)
    try:
        # numbered step dir
        api.upload_folder(
            repo_id=hf_model_repo,
            repo_type="model",
            folder_path=str(ckpt_path),
            path_in_repo=str(step),
            commit_message=f"Checkpoint step {step}",
        )
        # convenience latest/ symlink
        api.upload_folder(
            repo_id=hf_model_repo,
            repo_type="model",
            folder_path=str(ckpt_path),
            path_in_repo="latest",
            commit_message=f"Latest checkpoint: {step}",
        )
        return True
    except Exception as e:
        logger.warning("Checkpoint upload failed: %s", e)
        return False


# ── Rollout ID ────────────────────────────────────────────────────────────────


def make_rollout_id(model_step: int, strategy: str, worker_id: str | None = None) -> str:
    """Generate a unique rollout ID.

    Format: ``rollout_{step}_{strategy}_{YYYYMMDD_HHMMSS}[_{worker_id}]``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"rollout_{model_step}_{strategy}_{ts}"
    if worker_id:
        base = f"{base}_{worker_id}"
    return base


# ── Internal ──────────────────────────────────────────────────────────────────


def _parse_rollout_id(rollout_id: str) -> dict | None:
    import re

    m = re.match(r"^rollout_(\d+)_(.+?)_(\d{8}_\d{6})(?:_(.+))?$", rollout_id)
    if not m:
        return None
    try:
        model_step = int(m.group(1))
    except ValueError:
        return None
    return {
        "rollout_id": rollout_id,
        "model_step": model_step,
        "strategy": m.group(2),
        "timestamp": m.group(3),
        "worker_id": m.group(4),
    }
