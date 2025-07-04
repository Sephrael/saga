# main.py
"""CLI entry point for the SAGA novel generation system."""

from __future__ import annotations

import argparse

from orchestration.cli_runner import run


def main() -> None:
    """Parse command-line arguments and start SAGA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", default=None, help="Path to text file to ingest")
    args = parser.parse_args()
    run(args.ingest)


if __name__ == "__main__":
    main()
