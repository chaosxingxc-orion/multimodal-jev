#!/usr/bin/env python3
"""Run one experiment package from the committed Wiki ledger."""
from pathlib import Path
import os
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from tools.reports.experiment_package import main

    raise SystemExit(main())
