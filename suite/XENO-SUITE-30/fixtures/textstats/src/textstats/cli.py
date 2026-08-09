"""Command-line entry point for :mod:`textstats`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from textstats.report import render_text_report
from textstats.stats import top_n

EXIT_OK = 0
EXIT_USAGE = 2

PROG = "textstats"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    The only output shape available today is the plain-text report.
    """
    parser = argparse.ArgumentParser(prog=PROG, description="Summarise a text file.")
    parser.add_argument("path", help="path to a UTF-8 text file")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="also print the N most frequent words, one per line",
    )
    return parser


def read_text(path: Path) -> str:
    """Read ``path`` as UTF-8 text.

    Raises:
        OSError: if the file cannot be read.
    """
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    path = Path(args.path)
    try:
        text = read_text(path)
    except OSError as exc:
        print(f"{PROG}: cannot read {path}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(render_text_report(text))

    if args.top > 0:
        for word, count in top_n(text, args.top):
            print(f"{word}\t{count}")

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
