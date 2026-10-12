"""Comprehensive automated unit tests for DataQualityEngine.

Covers:
1. Clean dataset baseline
2. Missing values detection and threshold exceedance
3. Duplicate records detection
4. Invalid numeric ranges (min/max bounds)
5. Unexpected categorical values
6. Missing and unexpected schema columns
7. Empty DataFrames (0 rows, 0 columns)
8. Null values and mixed python data types
9. Deterministic score calculation and breakdown
10. Non-mutation guarantee (original DataFrame remains unchanged)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.quality_engine import DataQualityEngine, QualityFinding, QualityReport


@pytest.fixture
def clean_dataframe() -> pd.DataFrame:
    """Fixture returning a pristine, valid dataset with 5 rows."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Evan"],
        "age": [20, 22, 21, 23, 24],
        "status": ["active", "active", "pending", "active", "pending"],
        "score": [85.5, 92.0, 78.0, 88.5, 95.0],
    })


def test_clean_dataset(clean_dataframe: pd.DataFrame) -> None:
    """A clean dataset conforming to all rules should receive 100.0 score."""
    engine = DataQualityEngine(
        missing_threshold=0.10,
        expected_schema={
            "id": "int",
            "name": "string",
            "age": "int",
            "status": "string",
            "score": "float",
        },
        numeric_ranges={
            "age": {"min": 18, "max": 65},
            "score": {"min": 0.0, "max": 100.0},
        },
        categorical_allowed_values={
            "status": ["active", "pending"]
        },
    )

    report = engine.evaluate(clean_dataframe)

    assert report.row_count == 5
    assert report.column_count == 5
    assert report.overall_score == 100.0
    assert report.missing_summary["total_missing_cells"] == 0
    assert report.duplicate_summary["duplicate_rows"] == 0
    assert len(report.missing_summary["columns_exceeding_threshold"]) == 0
    assert len(report.schema_summary["missing_columns"]) == 0
    assert len(report.schema_summary["unexpected_columns"]) == 0
    assert len(report.schema_summary["type_mismatches"]) == 0

    grouped = report.findings_by_severity()
    assert len(grouped["ERROR"]) == 0
    assert len(grouped["WARNING"]) == 0


def test_missing_values() -> None:
    """Engine should accurately count missing values and flag columns exceeding threshold."""
    df = pd.DataFrame({
        "col_a": [1, 2, np.nan, 4, 5],            # 1 missing out of 5 (20%)
        "col_b": [np.nan, np.nan, np.nan, 4, 5],  # 3 missing out of 5 (60%)
        "col_c": ["x", "y", "z", "w", "v"],      # 0 missing (0%)
    })

    # Threshold set at 25%
    engine = DataQualityEngine(missing_threshold=0.25)
    report = engine.evaluate(df)

    assert report.missing_summary["total_cells"] == 15
    assert report.missing_summary["total_missing_cells"] == 4
    assert report.missing_summary["by_column"]["col_a"]["missing_count"] == 1
    assert report.missing_summary["by_column"]["col_a"]["exceeds_threshold"] is False
    assert report.missing_summary["by_column"]["col_b"]["missing_count"] == 3
    assert report.missing_summary["by_column"]["col_b"]["exceeds_threshold"] is True
    assert "col_b" in report.missing_summary["columns_exceeding_threshold"]
    assert "col_a" not in report.missing_summary["columns_exceeding_threshold"]

    # col_b missing >= 50% triggers ERROR finding
    errors = [f for f in report.findings if f.severity == "ERROR" and f.column == "col_b"]
    assert len(errors) == 1


def test_duplicate_records() -> None:
    """Engine should detect duplicate rows and compute accurate percentage."""
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": ["A", "B", "B", "C"],
    })  # 4 rows, 1 duplicate row -> 25%

    engine = DataQualityEngine()
    report = engine.evaluate(df)

    assert report.duplicate_summary["duplicate_rows"] == 1
    assert report.duplicate_summary["duplicate_percentage"] == 25.0

    warnings = [f for f in report.findings if f.category == "duplicates" and f.severity == "WARNING"]
    assert len(warnings) == 1
    assert "Found 1 duplicate row(s)" in warnings[0].message


def test_invalid_numeric_ranges() -> None:
    """Engine should flag values outside min/max bounds without mutating data."""
    df = pd.DataFrame({
        "age": [15, 20, 25, 120, -5],  # min: 18, max: 65 -> 15, 120, -5 are invalid (3 out of 5)
        "grade": [1, 2, 3, 4, 5],
    })

    engine = DataQualityEngine(
        numeric_ranges={
            "age": {"min": 18, "max": 65}
        }
    )
    report = engine.evaluate(df)

    num_val = report.validation_summary["numeric_validations"]["age"]
    assert num_val["total_evaluated"] == 5
    assert num_val["invalid_count"] == 3
    assert num_val["invalid_percentage"] == 60.0

    warnings = [f for f in report.findings if f.column == "age" and f.category == "value_validation"]
    assert len(warnings) == 1
    assert "out of valid range" in warnings[0].message


def test_unexpected_categorical_values() -> None:
    """Engine should detect categorical values not in the allowed set."""
    df = pd.DataFrame({
        "status": ["active", "suspended", "active", "UNKNOWN_STATUS", "deleted"],
    })

    engine = DataQualityEngine(
        categorical_allowed_values={
            "status": ["active", "suspended", "pending"]
        }
    )
    report = engine.evaluate(df)

    cat_val = report.validation_summary["categorical_validations"]["status"]
    assert cat_val["total_evaluated"] == 5
    assert cat_val["invalid_count"] == 2  # UNKNOWN_STATUS and deleted
    assert cat_val["invalid_percentage"] == 40.0
    assert "UNKNOWN_STATUS" in cat_val["unexpected_values"]

    warnings = [f for f in report.findings if f.column == "status" and f.category == "value_validation"]
    assert len(warnings) == 1
    assert "2 disallowed categorical record(s)" in warnings[0].message


def test_missing_and_unexpected_schema_columns() -> None:
    """Engine should identify missing expected columns and unexpected extra columns."""
    df = pd.DataFrame({
        "existing_col": [1, 2, 3],
        "extra_col": ["a", "b", "c"],
    })

    engine = DataQualityEngine(
        expected_schema={
            "existing_col": "int",
            "required_col_b": "string",
            "required_col_c": "float",
        }
    )
    report = engine.evaluate(df)

    schema = report.schema_summary
    assert "required_col_b" in schema["missing_columns"]
    assert "required_col_c" in schema["missing_columns"]
    assert "extra_col" in schema["unexpected_columns"]

    errors = [f for f in report.findings if f.category == "schema" and f.severity == "ERROR"]
    # 2 missing columns -> 2 ERROR findings
    assert len(errors) == 2

    warnings = [f for f in report.findings if f.category == "schema" and f.severity == "WARNING"]
    # 1 unexpected column -> 1 WARNING finding
    assert len(warnings) == 1


def test_empty_dataframe() -> None:
    """Engine should handle empty DataFrames gracefully without crashing."""
    empty_no_cols = pd.DataFrame()
    empty_with_cols = pd.DataFrame(columns=["a", "b", "c"])

    engine = DataQualityEngine(expected_schema={"a": "int", "b": "string"})

    report_1 = engine.evaluate(empty_no_cols)
    assert report_1.row_count == 0
    assert report_1.column_count == 0
    assert report_1.overall_score == 0.0
    assert any(f.severity == "ERROR" and "empty" in f.message.lower() for f in report_1.findings)

    report_2 = engine.evaluate(empty_with_cols)
    assert report_2.row_count == 0
    assert report_2.column_count == 3
    assert report_2.overall_score == 0.0
    assert any(f.severity == "ERROR" and "empty" in f.message.lower() for f in report_2.findings)


def test_null_values_and_mixed_data_types() -> None:
    """Engine should detect object columns containing mixed Python types."""
    df = pd.DataFrame({
        "mixed_col": [10, "string_value", 3.14, None],  # ints, strings, floats
        "clean_str": ["apple", "banana", None, "cherry"],
    })

    engine = DataQualityEngine()
    report = engine.evaluate(df)

    schema = report.schema_summary
    assert "mixed_col" in schema["mixed_type_columns"]
    assert "clean_str" not in schema["mixed_type_columns"]

    mixed_findings = [f for f in report.findings if f.column == "mixed_col" and "mixed python data types" in f.message]
    assert len(mixed_findings) == 1


def test_score_calculation() -> None:
    """Score calculation should follow deterministic, documented mathematical rules."""
    # Construct a dataset with 4 rows where:
    # 1. col_a has a missing cell exceeding threshold (1/4 = 25% > 10%)
    # 2. Rows 0 and 1 are duplicate rows (1 duplicate row out of 4 = 25%)
    df = pd.DataFrame({
        "col_a": [1, 1, None, 4],
        "col_b": ["A", "A", "C", "D"],
    })

    engine = DataQualityEngine(
        missing_threshold=0.10,
        expected_schema={"col_a": "int", "col_b": "string"},
    )
    report = engine.evaluate(df)

    score_summary = report.score_summary
    assert score_summary["base_score"] == 100.0
    assert "deductions" in score_summary
    assert score_summary["deductions"]["completeness"] > 0
    assert score_summary["deductions"]["uniqueness"] > 0
    assert 0.0 <= report.overall_score <= 100.0
    assert "disclaimer" in score_summary
    assert len(score_summary["disclaimer"]) > 20


def test_ensuring_original_dataframe_is_not_modified() -> None:
    """Engine must not mutate the input DataFrame under any circumstance."""
    original_df = pd.DataFrame({
        "id": [1, 2, 2, 4],
        "age": [20, 150, np.nan, 25],
        "category": ["A", "INVALID", "B", "A"],
    })

    # Keep a deep copy and snapshot of original dataframe
    expected_snapshot = original_df.copy(deep=True)

    engine = DataQualityEngine(
        missing_threshold=0.10,
        expected_schema={"id": "int", "age": "float", "category": "string"},
        numeric_ranges={"age": {"min": 18, "max": 65}},
        categorical_allowed_values={"category": ["A", "B"]},
    )

    report = engine.evaluate(original_df)

    # Assert that original_df remains perfectly identical
    pd.testing.assert_frame_equal(original_df, expected_snapshot)
    assert id(original_df) != id(expected_snapshot)
    assert len(original_df) == 4
    assert list(original_df.columns) == ["id", "age", "category"]


def test_invalid_engine_threshold() -> None:
    """Initializing DataQualityEngine with invalid threshold should raise ValueError."""
    with pytest.raises(ValueError):
        DataQualityEngine(missing_threshold=1.5)

    with pytest.raises(ValueError):
        DataQualityEngine(missing_threshold=-0.1)


def test_report_serialization_and_summary_text(clean_dataframe: pd.DataFrame) -> None:
    """Report should serialize cleanly to dictionary and generate readable text."""
    engine = DataQualityEngine()
    report = engine.evaluate(clean_dataframe)

    report_dict = report.to_dict()
    assert isinstance(report_dict, dict)
    assert "overall_score" in report_dict
    assert "findings" in report_dict
    assert report_dict["row_count"] == 5

    text_summary = report.summary_text()
    assert isinstance(text_summary, str)
    assert "QUALITYLENS DATA QUALITY AUDIT REPORT" in text_summary
    assert "100.0 / 100.0" in text_summary


def test_manual_test_csv_unconfigured_rules_root_cause() -> None:
    """Demonstrate root cause: without configured numeric ranges, custom uploads score 100/100."""
    manual_df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [20, 22, 145],
        "score": [85, 90, 70],
    })

    # When no numeric_ranges are passed (default behavior of unconfigured custom upload)
    engine = DataQualityEngine(missing_threshold=0.15)
    report = engine.evaluate(manual_df)

    assert report.overall_score == 100.0
    warnings = [f for f in report.findings if f.severity == "WARNING"]
    assert len(warnings) == 0
    assert len(report.validation_summary["numeric_validations"]) == 0


def test_manual_test_csv_with_configured_age_bounds() -> None:
    """When user configures age bounds (16 to 100), Charlie's age 145 is flagged and score is penalized."""
    manual_df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [20, 22, 145],
        "score": [85, 90, 70],
    })

    engine = DataQualityEngine(
        missing_threshold=0.15,
        numeric_ranges={"age": {"min": 16, "max": 100}},
    )
    report = engine.evaluate(manual_df)

    # Score should be penalized due to value validity deduction
    assert report.overall_score < 100.0
    assert report.overall_score == 95.0

    # Validation summary should accurately reflect Charlie's violation
    age_val = report.validation_summary["numeric_validations"]["age"]
    assert age_val["total_evaluated"] == 3
    assert age_val["invalid_count"] == 1
    assert age_val["invalid_percentage"] == 33.33
    assert age_val["sample_violations"] == [145]
    assert age_val["min_allowed"] == 16
    assert age_val["max_allowed"] == 100

    # Warning finding must be present
    age_warnings = [
        f for f in report.findings
        if f.column == "age" and f.category == "value_validation" and f.severity == "WARNING"
    ]
    assert len(age_warnings) == 1
    assert "1 numeric value(s) out of valid range [16, 100]" in age_warnings[0].message
    assert age_warnings[0].details["samples"] == [145]


def test_custom_csv_configurable_age_without_hardcoded_100_assumption() -> None:
    """Engine must not assume age maximum is 100; user-defined bounds must dictate validity."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [20, 22, 145],
        "score": [85, 90, 70],
    })

    # Case 1: Centenarian / historical study allows age up to 150
    engine_wide = DataQualityEngine(
        numeric_ranges={"age": {"min": 0, "max": 150}}
    )
    report_wide = engine_wide.evaluate(df)
    assert report_wide.overall_score == 100.0
    assert report_wide.validation_summary["numeric_validations"]["age"]["invalid_count"] == 0

    # Case 2: Pediatric cohort where age max is 18 -> Alice (20), Bob (22), Charlie (145) all invalid
    engine_pediatric = DataQualityEngine(
        numeric_ranges={"age": {"min": 0, "max": 18}}
    )
    report_pediatric = engine_pediatric.evaluate(df)
    ped_val = report_pediatric.validation_summary["numeric_validations"]["age"]
    assert ped_val["invalid_count"] == 3
    assert ped_val["invalid_percentage"] == 100.0
    assert report_pediatric.overall_score < 100.0


def test_custom_csv_multi_column_validation_and_reporting() -> None:
    """Multiple configurable numeric ranges evaluate independently and report invalid samples."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [20, 22, 145],
        "score": [85, 90, 70],
    })

    engine = DataQualityEngine(
        numeric_ranges={
            "age": {"min": 16, "max": 100},
            "score": {"min": 0, "max": 80},  # Alice (85) and Bob (90) violate this
        }
    )
    report = engine.evaluate(df)

    num_vals = report.validation_summary["numeric_validations"]
    assert "age" in num_vals
    assert "score" in num_vals

    assert num_vals["age"]["invalid_count"] == 1
    assert num_vals["age"]["sample_violations"] == [145]

    assert num_vals["score"]["invalid_count"] == 2
    assert num_vals["score"]["sample_violations"] == [85, 90]

    val_warnings = [f for f in report.findings if f.category == "value_validation"]
    assert len(val_warnings) == 2

