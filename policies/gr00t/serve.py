"""GR00T inference server — placeholder.

Will use GR00T's built-in PolicyServer (ZeroMQ).
See backends/isaac-gr00t/gr00t/eval/run_gr00t_server.py for reference.

Usage (future):
    uv run python serve.py \
        --model-path outputs/checkpoints/gr00t_so101/baseline_v1 \
        --embodiment-tag SO101 \
        --port 5555
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="GR00T N1.7 inference server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--embodiment-tag", type=str, default="SO101")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    raise NotImplementedError("GR00T inference server — Phase E")


if __name__ == "__main__":
    main()
