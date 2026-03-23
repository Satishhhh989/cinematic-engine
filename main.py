"""
Cinematic Motion HUD Engine — CLI Entry Point.

Usage:
    python main.py [input_video] [output_video]
"""

from __future__ import annotations

import logging
import sys
import time

from engine import run_engine


def main() -> None:
    """CLI wrapper: parses argv and runs the engine."""
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

    input_video = sys.argv[1] if len(sys.argv) >= 2 else "input/input.mp4"
    output_video = sys.argv[2] if len(sys.argv) >= 3 else "output/output_cinematic.mp4"

    print("╔══════════════════════════════════════════════════════╗")
    print("║        CINEMATIC MOTION HUD ENGINE                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Input:  {input_video}")
    print(f"  Output: {output_video}")
    print()

    t0 = time.time()
    result = run_engine(input_video, output_video)
    elapsed = time.time() - t0

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  Done in {elapsed:.1f}s → {result:<33}║")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
