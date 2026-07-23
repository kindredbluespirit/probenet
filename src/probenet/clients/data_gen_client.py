"""Abstract data generator client and implementations.

Connects to episode_gen servers (Isaac Sim or real robot) via WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod

import numpy as np
import websockets

logger = logging.getLogger(__name__)


class DataGenClient(ABC):
    """Abstract interface for communicating with a data generator server."""

    @abstractmethod
    def reset(self, seed: int = 0) -> dict:
        """Reset the environment and return initial observation."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool]:
        """Apply action, return (obs, reward, done)."""

    @abstractmethod
    def close(self):
        """Close the connection."""


class _BaseWebSocketClient(DataGenClient):
    """Base class for WebSocket-based data generator clients."""

    def __init__(self, url: str):
        self._url = url
        self._loop = asyncio.new_event_loop()

    def reset(self, seed: int = 0) -> dict:
        return self._send_recv({"type": "reset", "seed": seed})

    def step(self, action: np.ndarray) -> tuple[dict, float, bool]:
        result = self._send_recv({"type": "step", "action": action.tolist()})
        return result["obs"], result.get("reward", 0.0), result.get("done", False)

    def close(self):
        self._loop.close()

    def _send_recv(self, payload: dict) -> dict:
        return self._loop.run_until_complete(self._async_send_recv(payload))

    async def _async_send_recv(self, payload: dict) -> dict:
        async with websockets.connect(self._url) as ws:
            await ws.send(json.dumps(payload))
            return json.loads(await ws.recv())


class SimClient(_BaseWebSocketClient):
    """Connects to an Isaac Sim data generator server."""

    def __init__(self, url: str = "ws://localhost:8226"):
        super().__init__(url)


class RealRobotClient(_BaseWebSocketClient):
    """Connects to a real SO-101 robot data generator server."""

    def __init__(self, url: str = "ws://localhost:8227"):
        super().__init__(url)


def create_data_gen_client(source: str, url: str | None = None) -> DataGenClient:
    """Factory: create a data generator client for the given source."""
    if source == "sim":
        return SimClient(url or "ws://localhost:8226")
    if source == "so101":
        return RealRobotClient(url or "ws://localhost:8227")
    raise ValueError(f"Unknown data source: {source}")
