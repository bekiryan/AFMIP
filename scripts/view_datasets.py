#!/usr/bin/env python3
"""
Launch the Streamlit dataset viewer.

Usage:
    python scripts/view_datasets.py
    python scripts/view_datasets.py --port 9000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PATH = Path(__file__).resolve().parent / "view_datasets_streamlit.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch dataset explorer (Streamlit)")
    parser.add_argument("--port", type=int, default=8502, help="HTTP port (default 8502)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"Launching Streamlit viewer at http://localhost:{args.port}")
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
    except FileNotFoundError:
        print("Failed to start Streamlit. Install it with: pip install streamlit")
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Streamlit exited with code {exc.returncode}")
        raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()
