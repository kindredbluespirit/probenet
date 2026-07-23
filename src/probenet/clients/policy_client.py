"""Abstract policy client and WebSocket implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod

import numpy as np
import websockets

logger = logging.getLogger(__name__)


class PolicyClient(ABC):
    """Abstract interface for communicating with a policy inference server."""

    @abstractmethod
    def infer(self, obs: dict, initial_actions: np.ndarray | None = None) -> dict:
        """Send observation, receive actions."""


class WebSocketPolicyClient(PolicyClient):
    """Connects to an openpi WebSocket inference server."""

    def __init__(self, url: str = "ws://localhost:8000"):
        self._url = url
        self._websocket = None
        self._loop = asyncio.new_event_loop()

    def infer(self, obs: dict, initial_actions: np.ndarray | None = None) -> dict:
        payload = {"type": "infer_chunk", **self._serialize_obs(obs)}
        if initial_actions is not None:
            payload["initial_actions"] = initial_actions.tolist()
        return self._send_recv(payload)

    def _serialize_obs(self, obs: dict) -> dict:
        return obs

    def _send_recv(self, payload: dict) -> dict:
        return self._loop.run_until_complete(self._async_send_recv(payload))

    async def _async_send_recv(self, payload: dict) -> dict:
        async with websockets.connect(self._url) as ws:
            await ws.send(json.dumps(payload))
            response = json.loads(await ws.recv())
        return response

    def close(self):
        self._loop.close()


def create_policy_client(backend: str, url: str | None = None) -> PolicyClient:
    """Factory: create a policy client for the given backend."""
    if backend == "openpi":
        return WebSocketPolicyClient(url or "ws://localhost:8000")
    if backend == "gr00t":
        msg = "GR00T client not yet implemented"
        raise NotImplementedError(msg)
    raise ValueError(f"Unknown policy backend: {backend}")
