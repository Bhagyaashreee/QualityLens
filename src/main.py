"""QualityLens CLI - Command-line demonstration for data quality and drift monitoring.

Usage:
    python -m src.main
    python -m src.main --mode quality
    python -m src.main --mode drift
    python -m src.main --mode all
    python -m src.main --file path/to/dataset.csv
    python -m src.main --reference path/to/ref.csv --current path/to/curr.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.quality_engine import DataQualityEngine
from src.drift_engine import DataDriftEngine


def run_quality_pipeline(csv_path: Path) -> int:
    """Load a dataset, execute quality checks, and print summary report.

    Args:
        csv_path: Path to target CSV file.

    Returns:
        Exit code: 0 on success, 1 on file/read errors.
    """
    print(f"\n[QualityLens] >>> MILESTONE 1: DATA QUALITY AUDIT <<<")
    print(f"[QualityLens] Target Dataset: {csv_path}")

    # Graceful file existence checks
    if not csv_path.exists():
        print(f"\n[ERROR] Target file not found: '{csv_path}'", file=sys.stderr)
        print("Please verify the file path and ensure the dataset exists.\n", file=sys.stderr)
        return 1

    if not csv_path.is_file():
        print(f"\n[ERROR] Specified path is not a regular file: '{csv_path}'\n", file=sys.stderr)
        return 1

    # Graceful read check
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"\n[ERROR] Failed to read CSV: '{csv_path}' is completely empty.\n", file=sys.stderr)
        return 1
    except pd.errors.ParserError as pe:
        print(f"\n[ERROR] Malformed CSV structure in '{csv_path}': {pe}\n", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error opening file '{csv_path}': {e}\n", file=sys.stderr)
        return 1

    print(f"[QualityLens] Successfully loaded {len(df)} rows and {len(df.columns)} columns.")

    # Configure validation rules
    expected_schema = {
        "student_id": "int",
        "first_name": "string",
        "last_name": "string",
        "age": "numeric",
        "enrollment_status": "string",
        "gpa": "numeric",
        "major": "string",
    }

    numeric_ranges = {
        "age": {"min": 16, "max": 100},
        "gpa": {"min": 0.0, "max": 4.0},
    }

    categorical_allowed_values = {
        "enrollment_status": ["Full-Time", "Part-Time", "Graduated", "On Leave"]
    }

    engine = DataQualityEngine(
        missing_threshold=0.15,
        expected_schema=expected_schema,
        numeric_ranges=numeric_ranges,
        categorical_allowed_values=categorical_allowed_values,
    )

    report = engine.evaluate(df)
    print()
    print(report.summary_text())
    print()
    return 0


def run_drift_pipeline(reference_path: Path, current_path: Path) -> int:
    """Load reference baseline and current dataset, execute drift checks, and print report.

    Args:
        reference_path: Path to reference baseline CSV.
        current_path: Path to current batch CSV.

    Returns:
        Exit code: 0 on success, 1 on file/read errors.
    """
    print(f"\n[QualityLens] >>> MILESTONE 2: DATA DRIFT DETECTION <<<")
    print(f"[QualityLens] Reference Baseline : {reference_path}")
    print(f"[QualityLens] Current Batch      : {current_path}")

    # Validate file existence
    for label, path in [("Reference", reference_path), ("Current", current_path)]:
        if not path.exists():
            print(f"\n[ERROR] {label} dataset not found: '{path}'", file=sys.stderr)
            return 1
        if not path.is_file():
            print(f"\n[ERROR] {label} path is not a file: '{path}'", file=sys.stderr)
            return 1

    try:
        ref_df = pd.read_csv(reference_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to read Reference CSV '{reference_path}': {e}\n", file=sys.stderr)
        return 1

    try:
        curr_df = pd.read_csv(current_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to read Current CSV '{current_path}': {e}\n", file=sys.stderr)
        return 1

    print(f"[QualityLens] Loaded Reference ({len(ref_df)} rows) and Current ({len(curr_df)} rows).")

    features_to_check = [c for c in ["age", "gpa", "study_hours", "major", "enrollment_status"] if c in ref_df.columns]
    drift_engine = DataDriftEngine(
        numerical_p_value_threshold=0.05,
        categorical_psi_threshold=0.20,
        dataset_drift_share_threshold=0.33,
        features=features_to_check if features_to_check else None,
    )

    drift_report = drift_engine.detect_drift(ref_df, curr_df)
    print()
    print(drift_report.summary_text())
    print()
    return 0


def main() -> None:
    """CLI Entry point supporting Quality, Drift, and Combined modes."""
    parser = argparse.ArgumentParser(
        description="QualityLens CLI: Evaluate data hygiene, schema, duplicates, and statistical drift."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "quality", "drift"],
        default="all",
        help="Audit mode to execute: 'quality' (Milestone 1), 'drift' (Milestone 2), or 'all' (default)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data") / "sample" / "students_synthetic.csv",
        help="Path to CSV dataset for Quality Audit (default: data/sample/students_synthetic.csv)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data") / "sample" / "students_reference.csv",
        help="Path to Reference CSV for Drift Detection (default: data/sample/students_reference.csv)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("data") / "sample" / "students_current.csv",
        help="Path to Current CSV for Drift Detection (default: data/sample/students_current.csv)",
    )

    args = parser.parse_args()

    exit_code = 0
    if args.mode in ["all", "quality"]:
        exit_code = run_quality_pipeline(args.file)
        if exit_code != 0 and args.mode == "quality":
            sys.exit(exit_code)

    if args.mode in ["all", "drift"]:
        drift_exit = run_drift_pipeline(args.reference, args.current)
        if drift_exit != 0:
            exit_code = drift_exit

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
