"""ProbeNet dataset: collection helpers and PyTorch Dataset (episode-sequential)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def save_episode(
    output_dir: str | Path,
    episode_id: int,
    rgb: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    metadata: dict,
) -> Path:
    """Save a single episode to disk."""
    root = Path(output_dir)
    ep_dir = root / f"episode_{episode_id:05d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.save(ep_dir / "rgb.npy", rgb)
    np.save(ep_dir / "state.npy", state)
    np.save(ep_dir / "action.npy", action)
    with open(ep_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, default=str)
    return ep_dir


def load_episode(episode_dir: str | Path) -> dict:
    ep_dir = Path(episode_dir)
    return {
        "rgb": np.load(ep_dir / "rgb.npy"),
        "state": np.load(ep_dir / "state.npy"),
        "action": np.load(ep_dir / "action.npy"),
        "metadata": json.loads((ep_dir / "metadata.json").read_text()),
    }


def _vis_phys_from_metadata(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    vp = meta.get("visual_params", {})
    vis = np.array([vp.get(k, 0.0) for k in ("shape", "size", "gloss", "material")], dtype=np.float32)
    pp = meta.get("physical_params", {})
    phys = np.array([pp.get(k, 0.0) for k in ("mass", "compliance", "friction")], dtype=np.float32)
    return vis, phys


class EpisodeDataset(Dataset):
    """A single episode as a Dataset (each item = one frame in the episode)."""

    def __init__(self, episode_dir: Path, device: torch.device) -> None:
        self.ep_data = load_episode(episode_dir)
        meta = self.ep_data["metadata"]
        self.vis, self.phys = _vis_phys_from_metadata(meta)
        self.device = device

    def __len__(self) -> int:
        return len(self.ep_data["rgb"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rgb = torch.from_numpy(self.ep_data["rgb"][idx]).float().to(self.device) / 255.0
        rgb = rgb.permute(2, 0, 1)
        return {
            "rgb": rgb,
            "state": torch.from_numpy(self.ep_data["state"][idx]).float().to(self.device),
            "action": torch.from_numpy(self.ep_data["action"][idx]).float().to(self.device),
            "visual_params": torch.from_numpy(self.vis).float().to(self.device),
            "physical_params": torch.from_numpy(self.phys).float().to(self.device),
        }


class ProbeNetDataset(Dataset):
    """Dataset that aggregates all episodes into memory sequentially.

    Loads all episodes on init (one at a time, sequentially) and concatenates
    them into flat arrays. This avoids the random-access cost of lazy loading.

    Requires enough CPU RAM (~8 GB for 100 episodes). If that is a problem,
    use ``EpisodicDataLoader`` or reduce the number of episodes.
    """

    def __init__(self, data_dir: str | Path, device: torch.device | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.device = device or torch.device("cpu")

        episode_dirs = sorted(self.data_dir.glob("episode_*"))
        if not episode_dirs:
            raise FileNotFoundError(f"No episodes found in {data_dir}")

        # Load all episodes sequentially into memory.
        rgbs: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        vises: list[np.ndarray] = []
        phy_s: list[np.ndarray] = []

        for ep_dir in episode_dirs:
            ep = load_episode(ep_dir)
            rgbs.append(ep["rgb"])
            states.append(ep["state"])
            actions.append(ep["action"])
            vis, phys = _vis_phys_from_metadata(ep["metadata"])
            vises.append(np.tile(vis, (len(ep["rgb"]), 1)))
            phy_s.append(np.tile(phys, (len(ep["rgb"]), 1)))

        self.rgb = np.concatenate(rgbs, axis=0)
        self.state = np.concatenate(states, axis=0)
        self.action = np.concatenate(actions, axis=0)
        self.visual_params = np.concatenate(vises, axis=0)
        self.physical_params = np.concatenate(phy_s, axis=0)

    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rgb = torch.from_numpy(self.rgb[idx]).float().to(self.device) / 255.0
        rgb = rgb.permute(2, 0, 1)
        return {
            "rgb": rgb,
            "state": torch.from_numpy(self.state[idx]).float().to(self.device),
            "action": torch.from_numpy(self.action[idx]).float().to(self.device),
            "visual_params": torch.from_numpy(self.visual_params[idx]).float().to(self.device),
            "physical_params": torch.from_numpy(self.physical_params[idx]).float().to(self.device),
        }


class EpisodicDataLoader:
    """Iterate over episodes, yielding mini-batches from one episode at a time.

    This avoids the random-access overhead of the flat Dataset while still
    shuffling at the episode level and frame level within each episode.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 64,
        shuffle: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or torch.device("cpu")

        episode_dirs = sorted(Path(data_dir).glob("episode_*"))
        if not episode_dirs:
            raise FileNotFoundError(f"No episodes found in {data_dir}")
        self.episode_dirs = episode_dirs
        self.episode_datasets: list[EpisodeDataset] = []

    def __len__(self) -> int:
        # Estimate number of batches.
        total = sum(len(ep) for ep in self.episode_datasets) if self.episode_datasets else 0
        return max(1, total // self.batch_size)

    def __iter__(self):
        # Shuffle episode order.
        order = list(range(len(self.episode_dirs)))
        if self.shuffle:
            np.random.shuffle(order)

        for ep_idx in order:
            ep_path = self.episode_dirs[ep_idx]
            ep_ds = EpisodeDataset(ep_path, self.device)
            frame_order = list(range(len(ep_ds)))
            if self.shuffle:
                np.random.shuffle(frame_order)

            batch_rgb: list[torch.Tensor] = []
            batch_state: list[torch.Tensor] = []
            batch_action: list[torch.Tensor] = []
            batch_vis: list[torch.Tensor] = []
            batch_phys: list[torch.Tensor] = []

            for fi in frame_order:
                item = ep_ds[fi]
                batch_rgb.append(item["rgb"])
                batch_state.append(item["state"])
                batch_action.append(item["action"])
                batch_vis.append(item["visual_params"])
                batch_phys.append(item["physical_params"])

                if len(batch_rgb) == self.batch_size:
                    yield {
                        "rgb": torch.stack(batch_rgb),
                        "state": torch.stack(batch_state),
                        "action": torch.stack(batch_action),
                        "visual_params": torch.stack(batch_vis),
                        "physical_params": torch.stack(batch_phys),
                    }
                    batch_rgb, batch_state, batch_action, batch_vis, batch_phys = [], [], [], [], []

            # Don't drop the last incomplete batch (it may be small, but that's fine).
            if batch_rgb:
                yield {
                    "rgb": torch.stack(batch_rgb),
                    "state": torch.stack(batch_state),
                    "action": torch.stack(batch_action),
                    "visual_params": torch.stack(batch_vis),
                    "physical_params": torch.stack(batch_phys),
                }


def create_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    train_split: float = 0.8,
    device: torch.device | None = None,
) -> tuple[EpisodicDataLoader, EpisodicDataLoader]:
    """Create train/val data loaders.

    Splits episodes by episode (not by frame), so no episode leaks across splits.
    """
    episode_dirs = sorted(Path(data_dir).glob("episode_*"))
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found in {data_dir}")

    n = len(episode_dirs)
    n_train = max(1, int(n * train_split))
    np.random.shuffle(episode_dirs)

    train_dirs = episode_dirs[:n_train]
    val_dirs = episode_dirs[n_train:]

    def _make_loader(dirs: list[Path]) -> EpisodicDataLoader:
        loader = EpisodicDataLoader(data_dir, batch_size=batch_size, shuffle=True, device=device)
        loader.episode_dirs = dirs
        return loader

    return _make_loader(train_dirs), _make_loader(val_dirs)
