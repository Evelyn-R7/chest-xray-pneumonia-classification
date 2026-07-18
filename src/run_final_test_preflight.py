"""Run final test preflight without reading test.csv contents or test images."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.final_test.preflight import run_preflight
from src.final_test.safety import EXPECTED_PROTOCOL_SHA256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-protocol-sha256", default=EXPECTED_PROTOCOL_SHA256)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/final_test_preflight")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_preflight(
        project_root=args.project_root,
        protocol_path=args.project_root / "results/final_protocol/final_protocol.json",
        frozen_marker_path=args.project_root / "results/final_protocol/PROTOCOL_FROZEN",
        output_dir=args.output_dir,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    report = args.project_root / "reports/final_test_preflight.md"
    report.write_text(
        "# Final test preflight\n\n"
        f"- final_protocol SHA-256: `{payload['final_protocol_sha256']}`\n"
        f"- test manifest content read: `{payload['test_manifest_content_read']}`\n"
        f"- test images read: `{payload['test_images_read']}`\n"
        f"- started marker created: `{payload['started_marker_created']}`\n",
        encoding="utf-8",
    )
    print("PREFLIGHT_JSON=" + str((args.output_dir / "preflight.json").resolve()))
    print("PREFLIGHT_REPORT=" + str(report.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
