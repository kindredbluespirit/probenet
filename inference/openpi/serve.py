"""ProbeNet openpi (π₀.₅) inference server.

Loads a fine-tuned checkpoint and serves policy inference over WebSocket.

Protocol:
    → {"type": "infer", "obs": {...}}
    ← {"actions": [...], "metadata": {...}}

Uses openpi's built-in WebsocketPolicyServer or a thin wrapper.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="openpi π₀.₅ inference server")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--config-name", type=str, default="pi05_so101", help="Registered config name")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", type=str, default="pick and place the object")
    args = parser.parse_args()

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _train_config

    train_config = _train_config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=train_config.policy_metadata or {},
    )
    logger.info("Starting openpi inference server on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
