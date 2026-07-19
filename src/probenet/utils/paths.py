"""Path utilities for locating third-party assets."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def menagerie_path() -> Path:
    """Return the path to the cloned mujoco_menagerie repository.

    Raises:
        FileNotFoundError: if the repository has not been cloned.
    """
    path = REPO_ROOT / "third_party" / "mujoco_menagerie"
    if not path.exists():
        raise FileNotFoundError(
            "mujoco_menagerie not found. "
            "Clone it with: git clone --depth=1 "
            "https://github.com/google-deepmind/mujoco_menagerie.git "
            "third_party/mujoco_menagerie"
        )
    return path


def so101_scene_path() -> Path:
    """Return the path to the robotstudio_so101 scene.xml file."""
    return menagerie_path() / "robotstudio_so101" / "scene.xml"


def so101_model_path() -> Path:
    """Return the path to the robotstudio_so101 so101.xml file."""
    return menagerie_path() / "robotstudio_so101" / "so101.xml"
