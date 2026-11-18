"""Concurrent-rank-safe CLI for the unchanged epoch-tail diagnostic."""
import argparse
import json
from pathlib import Path

from tools.reports.context_macro_tail_candidate import run


def main(argv=None, *, runner=run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    # The lifecycle owns a fresh attempt root. Both torchrun ranks share this
    # parent; the native trainer still exclusively creates each rank directory.
    args.output.mkdir(parents=True, exist_ok=True)
    return runner(json.loads(args.config.read_text()), args.output)


if __name__ == '__main__':
    main()
