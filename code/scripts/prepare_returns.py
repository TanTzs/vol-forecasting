"""Build five-minute return files from the raw quote CSV files."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from volatility_forecasting_dqn.data import clean_quote_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw Chinese A-share quotes and calculate log returns."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "chinese",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chinese_return",
    )
    args = parser.parse_args()

    written_files = clean_quote_directory(args.raw_dir, args.output_dir)
    print(f"Generated {len(written_files)} return files in {args.output_dir}")


if __name__ == "__main__":
    main()
