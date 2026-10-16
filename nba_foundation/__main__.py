"""Run with python -m nba_foundation --help."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from urllib.error import URLError

from .adapters import BOX, GENERIC, PBP
from .fetch import fetch_game
from .reconcile import reconcile_game
from .store import Archive, DATA_CLASSES, ENTITIES


def parser():
    root = argparse.ArgumentParser(description="Archive and reconcile NBA data without inventing missing inputs")
    root.add_argument("--db", default="nba_archive.sqlite3", help="SQLite archive path")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialize archive schema")
    fetch = commands.add_parser("fetch-game", help="Archive official NBA Live box score and play-by-play")
    fetch.add_argument("game_id")
    ingest = commands.add_parser("ingest", help="Archive a saved source response or complete records.v1 snapshot")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--feed", choices=[BOX, PBP, GENERIC], required=True)
    ingest.add_argument("--scope", required=True, help="Game ID or stable provider snapshot scope")
    ingest.add_argument("--source-uri", required=True, help="Origin of the saved source response")
    ingest.add_argument("--data-class", choices=sorted(DATA_CLASSES), default="unverified",
                        help="Use real only for an authenticated real-data source; local files default to unverified")
    ingest.add_argument("--source-published-at", help="Documented source publication time, with timezone")
    query = commands.add_parser("query", help="Export only the complete snapshot known at a cutoff")
    query.add_argument("--source", required=True)
    query.add_argument("--feed", required=True)
    query.add_argument("--scope", required=True)
    query.add_argument("--cutoff", required=True)
    query.add_argument("--entity", choices=sorted(ENTITIES))
    query.add_argument("--max-age-seconds", type=float)
    reconcile = commands.add_parser("reconcile", help="Check an archived final game; exit 2 if gate fails")
    reconcile.add_argument("game_id")
    reconcile.add_argument("--cutoff", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        with Archive(args.db) as archive:
            exit_code = 0
            if args.command == "init":
                result = {"initialized": str(args.db), "schema_version": 1}
            elif args.command == "fetch-game":
                result = {"snapshots": fetch_game(archive, args.game_id)}
            elif args.command == "ingest":
                snapshot_id = archive.ingest(args.path.read_bytes(), source=args.source, feed=args.feed,
                    scope=args.scope, data_class=args.data_class, source_uri=args.source_uri,
                    source_published_at=args.source_published_at)
                result = archive.metadata(snapshot_id)
                exit_code = 2 if result["normalization_error"] else 0
            elif args.command == "query":
                result = archive.records_as_of(source=args.source, feed=args.feed, scope=args.scope,
                    cutoff=args.cutoff, entity=args.entity, max_age_seconds=args.max_age_seconds)
            else:
                result = reconcile_game(archive, args.game_id, args.cutoff)
                exit_code = 0 if result["passed"] else 2
        print(json.dumps(result, indent=2, allow_nan=False))
        return exit_code
    except (ValueError, KeyError, TypeError, OSError, URLError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc), "status": "withheld"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
