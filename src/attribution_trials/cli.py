"""Command-line entrypoint.

Subcommands:

* ``ingest`` -- parse a Lichess archive into a persisted trajectory dataset.

Run as ``python -m attribution_trials.cli ingest <archive> --out <dir>``.
"""

from __future__ import annotations

import argparse

from attribution_trials import __version__


def _cmd_ingest(args: argparse.Namespace) -> int:
    from attribution_trials.data.ingest import run_ingest

    run_ingest(
        args.archive,
        args.out,
        speed=None if args.speed == "all" else args.speed,
        min_games=args.min_games,
        min_sessions=args.min_sessions,
        max_players=args.max_players,
        gap_threshold_seconds=args.gap_threshold,
        workers=args.workers,
        batch_size=args.batch_size,
        max_games=args.max_games,
        max_games_per_player=args.max_games_per_player,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m attribution_trials.cli", description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"attribution-trials {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser(
        "ingest",
        help="parse a Lichess .pgn(.zst) archive into a persisted dataset",
    )
    ing.add_argument("archive", help="path to a .pgn or .pgn.zst archive")
    ing.add_argument(
        "--out",
        required=True,
        help="output directory (dataset.jsonl.gz + manifest.json)",
    )
    ing.add_argument(
        "--speed",
        default="blitz",
        choices=[
            "ultrabullet",
            "bullet",
            "blitz",
            "rapid",
            "classical",
            "correspondence",
            "all",
        ],
        help="single time-control class to keep (default blitz; 'all' mixes)",
    )
    ing.add_argument("--min-games", type=int, default=50)
    ing.add_argument("--min-sessions", type=int, default=3)
    ing.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="cap the cohort to the top-N players by game count",
    )
    ing.add_argument(
        "--gap-threshold",
        type=float,
        default=1800.0,
        help="session gap threshold in seconds (default 1800 = 30 min)",
    )
    ing.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes for the pass-2 parse (the bottleneck); 1 = serial",
    )
    ing.add_argument("--batch-size", type=int, default=512)
    ing.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="cap games read per pass",
    )
    ing.add_argument(
        "--max-games-per-player",
        type=int,
        default=None,
        help="cap each player to their earliest N games",
    )
    ing.set_defaults(func=_cmd_ingest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
