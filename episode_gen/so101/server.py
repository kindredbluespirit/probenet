"""Real SO-101 robot data generator — WebSocket server with DAgger support.

Connects to a physical SO-101 arm via lerobot's Follower class and
exposes a WebSocket interface for the probenet orchestrator.

Two modes:
  1. Autonomous: orchestrator sends actions, robot executes them.
  2. DAgger/teleop: human uses leader arms, robot follows, data is recorded.

Protocol (same as sim/):
    → {"type": "reset"}
    ← {"obs": {...}, "info": {...}}

    → {"type": "step", "action": [...]}
    ← {"obs": {...}, "reward": 0.0, "done": false}
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


class SO101Server:
    """WebSocket server wrapping a real SO-101 robot."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8227):
        self._host = host
        self._port = port
        self._robot = None
        self._episode = []

    async def handle_client(self, websocket):
        async for message in websocket:
            request = json.loads(message)
            msg_type = request.get("type")

            if msg_type == "reset":
                response = self._handle_reset(request)
            elif msg_type == "step":
                response = self._handle_step(request)
            elif msg_type == "switch_mode":
                response = self._handle_switch_mode(request)
            elif msg_type == "close":
                response = {"status": "ok"}
                await websocket.send(json.dumps(response))
                break
            else:
                response = {"status": "error", "message": f"Unknown type: {msg_type}"}

            await websocket.send(json.dumps(response))

    def _handle_reset(self, request: dict) -> dict:
        if self._robot is None:
            self._robot = self._connect_robot()
        self._episode = []
        obs = self._robot.get_observation()
        return {"status": "ok", "obs": self._serialize_obs(obs), "info": {}}

    def _handle_step(self, request: dict) -> dict:
        action = np.array(request["action"], dtype=np.float32)
        self._robot.send_action(action)
        obs = self._robot.get_observation()
        self._episode.append({"obs": obs, "action": action.tolist()})
        return {"status": "ok", "obs": self._serialize_obs(obs), "reward": 0.0, "done": False}

    def _handle_switch_mode(self, request: dict) -> dict:
        mode = request.get("mode", "auto")
        logger.info("Switching to mode: %s", mode)
        return {"status": "ok", "mode": mode}

    def _serialize_obs(self, obs) -> dict:
        return obs

    def _connect_robot(self):
        raise NotImplementedError("Subclass must implement _connect_robot")

    async def run(self):
        logger.info("Starting SO101 server on %s:%d", self._host, self._port)
        async with websockets.serve(self.handle_client, self._host, self._port):
            await asyncio.Future()

    def close(self):
        if self._robot is not None:
            self._robot.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="SO-101 real robot data generator server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8227)
    parser.add_argument("--port-config", type=str, help="Path to robot port config YAML")
    args = parser.parse_args()

    server = SO101Server(host=args.host, port=args.port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
