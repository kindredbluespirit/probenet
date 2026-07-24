import os
from pathlib import Path


def _detect_git_root() -> Path:
    try:
        from git import Repo

        repo = Repo(os.getcwd(), search_parent_directories=True)
        return Path(repo.git.rev_parse("--show-toplevel"))
    except Exception:
        return Path(__file__).resolve().parents[3]


def _resolve_assets_root() -> str:
    env_root = os.environ.get("PROBENET_ASSETS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve().as_posix()

    return (_detect_git_root() / "assets").resolve().as_posix()


ASSETS_ROOT = _resolve_assets_root()
