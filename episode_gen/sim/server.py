"""Isaac Sim data generator — WebSocket server.

Runs inside an Isaac Sim SimulationApp process. Exposes a WebSocket
interface for the probenet orchestrator to drive episode collection.

Protocol:
    → {"type": "reset", "seed": 42}
    ← {"obs": {...}, "info": {...}}

    → {"type": "step", "action": [...]}
    ← {"obs": {...}, "reward": 0.0, "done": false, "info": {...}}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import websockets

logger = logging.getLogger(__name__)


class IsaacSimServer:
    """WebSocket server wrapping an Isaac Sim environment."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8226, headless: bool = True):
        self._host = host
        self._port = port
        self._headless = headless
        self._env = None

    async def handle_client(self, websocket):
        async for message in websocket:
            request = json.loads(message)
            msg_type = request.get("type")

            if msg_type == "reset":
                response = self._handle_reset(request)
            elif msg_type == "step":
                response = self._handle_step(request)
            elif msg_type == "get_obs":
                response = self._handle_get_obs()
            elif msg_type == "close":
                response = {"status": "ok"}
                await websocket.send(json.dumps(response))
                break
            else:
                response = {"status": "error", "message": f"Unknown type: {msg_type}"}

            await websocket.send(json.dumps(response))

    def _handle_reset(self, request: dict) -> dict:
        seed = request.get("seed", 0)
        if self._env is None:
            self._env = self._create_env()
        obs, info = self._env.reset(seed=seed)
        return {"status": "ok", "obs": self._serialize_obs(obs), "info": info}

    def _handle_step(self, request: dict) -> dict:
        action = np.array(request["action"], dtype=np.float64)
        obs, reward, done, truncated, info = self._env.step(action)
        return {
            "status": "ok",
            "obs": self._serialize_obs(obs),
            "reward": float(reward),
            "done": bool(done or truncated),
            "info": info,
        }

    def _handle_get_obs(self) -> dict:
        obs = self._env._get_observations()
        return {"status": "ok", "obs": self._serialize_obs(obs)}

    def _serialize_obs(self, obs) -> dict:
        return obs

    def _create_env(self):
        raise NotImplementedError("Subclass must implement _create_env")

    async def run(self):
        logger.info("Starting Isaac Sim server on %s:%d", self._host, self._port)
        async with websockets.serve(self.handle_client, self._host, self._port):
            await asyncio.Future()

    def close(self):
        if self._env is not None:
            self._env.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Isaac Sim data generator server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8226)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    server = IsaacSimServer(host=args.host, port=args.port, headless=args.headless)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
