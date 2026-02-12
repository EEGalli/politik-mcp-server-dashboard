#!/usr/bin/env python3
"""
Fristående script för att kontrollera sociala medier-flöden.
Kör manuellt eller via cron:

    # Varje dag kl 07:00
    0 7 * * * cd /path/to/politik-mcp-server && python run_feed_check.py

    # Kontrollera en specifik politiker
    python run_feed_check.py --politician ulf_kristersson

    # Fler jämförelser per inlägg
    python run_feed_check.py --max-comparisons 5
"""

import argparse
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from src.feed_monitor import check_feeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kontrollera sociala medier-flöden via RSSHub"
    )
    parser.add_argument(
        "--politician",
        default="",
        help="Kontrollera bara denna politiker_key (default: alla)",
    )
    parser.add_argument(
        "--max-comparisons",
        type=int,
        default=3,
        help="Max antal löften att jämföra per nytt inlägg (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Aktivera debug-loggning",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = asyncio.run(
        check_feeds(
            politician_key=args.politician,
            max_comparisons_per_item=args.max_comparisons,
        )
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
