"""Public CLI placeholder for dataset audit entry point.

The full audit report is preserved in reports/data_audit.md. This public entry
point intentionally exposes help without running any data operation by default.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Optional manifest path for a future read-only audit run.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    print("Audit implementation is documented in reports/data_audit.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
