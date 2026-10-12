"""Comprehensive automated unit tests for DataDriftEngine.

Covers:
1. Baseline no-drift case on identical distributions
2. Detected numerical drift via Kolmogorov-Smirnov test
3. Detected categorical drift via Population Stability Index (PSI)
4. Unseen and missing categorical values handling
5. Missing columns in current dataset
6. Incompatible data types between reference and current
7. Empty DataFrames (0 rows)
8. High null rates and small sample sizes
9. Non-mutation guarantee (reference and current remain unchanged)
10. Report serialization and summary text generation
11. Selected feature subsets
12. End-to-end drift detection on synthetic student sample files
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.drift_engine import DataDriftEngine, DriftReport, FeatureDriftResult


@pytest.fixture
def baseline_dataframe() -> pd.DataFrame:
    """Fixture providing a deterministic reference DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": list(range(1, 51)),
        "age": np.random.normal(loc=22.0, scale=2.0, size=50),
        "score": np.random.normal(loc=75.0, scale=10.0, size=50),
        "department": ["Engineering"] * 25 + ["Business"] * 15 + ["Arts"] * 10,
    })


def test_no_drift_identical_datasets(baseline_dataframe: pd.DataFrame) -> None:
    """Comparing identical datasets should detect zero drift and output STABLE status."""
    engine = DataDriftEngine(
        numerical_p_value_threshold=0.05,
        categorical_psi_threshold=0.20,
    )
    report = engine.detect_drift(baseline_dataframe, baseline_dataframe.copy())

    assert report.reference_row_count == 50
    assert report.current_row_count == 50
    assert report.drifted_features_count == 0
    assert report.drift_share == 0.0
    assert report.is_dataset_drifted is False
    assert report.overall_status == "STABLE"

    # Age KS p-value should be ~1.0
    age_res = report.feature_results["age"]
    assert age_res.is_drift_detected is False
    assert age_res.severity == "NONE"
    assert age_res.p_value is not None and age_res.p_value > 0.90

    # Department PSI should be 0.0
    dept_res = report.feature_results["department"]
    assert dept_res.is_drift_detected is False
    assert dept_res.severity == "NONE"
    assert dept_res.statistic is not None and dept_res.statistic < 0.01


def test_numerical_drift_detection() -> None:
    """Engine should flag significant distribution shifts in numeric features via KS test."""
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "metric": np.random.normal(loc=10.0, scale=2.0, size=100)
    })
    curr_df = pd.DataFrame({
        # Shifted mean from 10.0 to 18.0 (severe distribution divergence)
        "metric": np.random.normal(loc=18.0, scale=2.0, size=100)
    })

    engine = DataDriftEngine(numerical_p_value_threshold=0.05)
    report = engine.detect_drift(ref_df, curr_df)

    res = report.feature_results["metric"]
    assert res.is_drift_detected is True
    assert res.severity in ["MODERATE", "HIGH"]
    assert res.test_name == "kolmogorov_smirnov"
    assert res.p_value is not None and res.p_value < 0.001
    assert res.statistic is not None and res.statistic > 0.50
    assert report.is_dataset_drifted is True


def test_categorical_drift_detection() -> None:
    """Engine should flag distribution shifts in categorical features via PSI."""
    ref_df = pd.DataFrame({
        "tier": ["Gold"] * 70 + ["Silver"] * 20 + ["Bronze"] * 10
    })
    curr_df = pd.DataFrame({
        # Inverted proportions
        "tier": ["Gold"] * 10 + ["Silver"] * 30 + ["Bronze"] * 60
    })

    engine = DataDriftEngine(categorical_psi_threshold=0.20)
    report = engine.detect_drift(ref_df, curr_df)

    res = report.feature_results["tier"]
    assert res.is_drift_detected is True
    assert res.severity in ["MODERATE", "HIGH"]
    assert res.test_name == "population_stability_index"
    assert res.statistic is not None and res.statistic >= 0.20
    assert res.details["total_variation_distance"] > 0.30


def test_unseen_and_missing_categories() -> None:
    """Engine should handle unseen categories in current and missing categories gracefully."""
    ref_df = pd.DataFrame({
        "category": ["Alpha", "Beta", "Beta", "Beta"]
    })
    curr_df = pd.DataFrame({
        # Beta present, Alpha missing, Omega is brand new unseen category
        "category": ["Beta", "Beta", "Omega", "Omega"]
    })

    engine = DataDriftEngine(categorical_psi_threshold=0.20)
    report = engine.detect_drift(ref_df, curr_df)

    res = report.feature_results["category"]
    assert "Omega" in res.details["unseen_categories"]
    assert "Alpha" in res.details["missing_categories"]
    assert res.statistic is not None  # Successfully computed with epsilon smoothing without crash


def test_missing_columns_in_current_dataset() -> None:
    """Missing column in current dataset should be flagged as ERROR and counted as drifted."""
    ref_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    curr_df = pd.DataFrame({"col_a": [1, 2, 3]})  # col_b missing

    engine = DataDriftEngine()
    report = engine.detect_drift(ref_df, curr_df)

    assert "col_b" in report.missing_in_current
    res = report.feature_results["col_b"]
    assert res.is_drift_detected is True
    assert res.severity == "ERROR"
    assert "missing in current" in res.message


def test_incompatible_data_types() -> None:
    """Column with different data types between ref and curr should be flagged as ERROR."""
    ref_df = pd.DataFrame({"field": [10.5, 20.2, 30.1]})
    curr_df = pd.DataFrame({"field": ["text_a", "text_b", "text_c"]})

    engine = DataDriftEngine()
    report = engine.detect_drift(ref_df, curr_df)

    res = report.feature_results["field"]
    assert res.is_drift_detected is True
    assert res.severity == "ERROR"
    assert "Incompatible data types" in res.message


def test_empty_dataframes() -> None:
    """Empty DataFrames should be handled gracefully without exceptions."""
    empty_ref = pd.DataFrame(columns=["a", "b"])
    valid_curr = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    engine = DataDriftEngine()
    report_1 = engine.detect_drift(empty_ref, valid_curr)
    assert report_1.overall_status == "ERROR"
    assert report_1.reference_row_count == 0

    report_2 = engine.detect_drift(valid_curr, empty_ref)
    assert report_2.overall_status == "ERROR"
    assert report_2.current_row_count == 0


def test_null_values_and_small_sample_size() -> None:
    """Engine should handle null values and warn when sample sizes are below minimum."""
    ref_df = pd.DataFrame({"metric": [1.0, 2.0, None, None]})  # only 2 valid
    curr_df = pd.DataFrame({"metric": [1.5, None, 2.5, None]})  # only 2 valid

    engine = DataDriftEngine(min_sample_size=5)
    report = engine.detect_drift(ref_df, curr_df)

    res = report.feature_results["metric"]
    assert res.is_drift_detected is False
    assert "Insufficient non-null observations" in res.message
    assert res.reference_stats["missing_percentage"] == 50.0


def test_ensuring_original_dataframes_not_modified() -> None:
    """Engine must not mutate input DataFrames under any circumstance."""
    ref_df = pd.DataFrame({"val": [10.0, 20.0, 30.0], "cat": ["A", "B", "A"]})
    curr_df = pd.DataFrame({"val": [15.0, 25.0, 35.0], "cat": ["B", "B", "A"]})

    ref_snapshot = ref_df.copy(deep=True)
    curr_snapshot = curr_df.copy(deep=True)

    engine = DataDriftEngine()
    engine.detect_drift(ref_df, curr_df)

    pd.testing.assert_frame_equal(ref_df, ref_snapshot)
    pd.testing.assert_frame_equal(curr_df, curr_snapshot)


def test_drift_report_serialization_and_summary_text(baseline_dataframe: pd.DataFrame) -> None:
    """Report should serialize cleanly to dictionary and generate readable text."""
    engine = DataDriftEngine()
    report = engine.detect_drift(baseline_dataframe, baseline_dataframe)

    report_dict = report.to_dict()
    assert isinstance(report_dict, dict)
    assert "overall_status" in report_dict
    assert "features" in report_dict
    assert report_dict["reference_row_count"] == 50

    summary_text = report.summary_text()
    assert isinstance(summary_text, str)
    assert "QUALITYLENS DATA DRIFT MONITORING REPORT" in summary_text
    assert "INTERPRETATION GUIDANCE" in summary_text


def test_selected_feature_subsets(baseline_dataframe: pd.DataFrame) -> None:
    """Engine should respect features parameter filter."""
    engine = DataDriftEngine(features=["age"])
    report = engine.detect_drift(baseline_dataframe, baseline_dataframe)

    assert len(report.feature_results) == 1
    assert "age" in report.feature_results
    assert "score" not in report.feature_results


def test_synthetic_sample_datasets_drift() -> None:
    """End-to-end verification using the sample reference and current CSVs."""
    sample_dir = Path("data") / "sample"
    ref_path = sample_dir / "students_reference.csv"
    curr_path = sample_dir / "students_current.csv"

    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path)

    engine = DataDriftEngine(
        numerical_p_value_threshold=0.05,
        categorical_psi_threshold=0.20,
    )
    report = engine.detect_drift(ref_df, curr_df)

    # age and gpa should be stable
    assert report.feature_results["age"].is_drift_detected is False
    assert report.feature_results["gpa"].is_drift_detected is False

    # study_hours should be drifted (mean shifted from ~15 to ~29 hrs)
    study_res = report.feature_results["study_hours"]
    assert study_res.is_drift_detected is True
    assert study_res.test_name == "kolmogorov_smirnov"
    assert study_res.p_value < 0.001

    # enrollment_status should be drifted (new Remote-Online category introduced)
    status_res = report.feature_results["enrollment_status"]
    assert status_res.is_drift_detected is True
    assert status_res.test_name == "population_stability_index"
    assert "Remote-Online" in status_res.details["unseen_categories"]
