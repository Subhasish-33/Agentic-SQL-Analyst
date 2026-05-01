"""CLI entrypoint for offline or CI evaluation of the SQL analyst."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from app.evaluation.runner import main as run_evaluation_main


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the Agentic SQL Analyst.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N evaluation queries.")
    return parser.parse_args()


def main() -> None:
    """Execute the evaluation CLI."""

    args = parse_args()
    logging.basicConfig(level=logging.ERROR)
    result = asyncio.run(run_evaluation_main(limit=args.limit))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
