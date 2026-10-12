"""Automated unit tests for dashboard helper utilities (src/dashboard_utils.py).

Tests:
1. Safe CSV parsing for valid files, missing files, and empty buffers
2. Quality score radial gauge generation across score tiers
3. Missing values bar chart generation
4. Numerical drift distribution charts (histogram & box plot)
5. Categorical drift distribution charts
6. Report serialization and export formats (JSON & CSV)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.dashboard_utils import (
    load_csv_safely,
    apply_dark_theme,
    build_quality_score_gauge,
    build_missing_values_chart,
    build_numerical_drift_chart,
    build_categorical_drift_chart,
    format_quality_export,
    format_drift_export,
    extract_out_of_bounds_rows,
    format_active_rules_explanation,
)
from src.quality_engine import DataQualityEngine
from src.drift_engine import DataDriftEngine


def test_load_csv_safely_valid_file() -> None:
    """Loading a valid CSV file path should return a DataFrame and no error."""
    sample_path = Path("data") / "sample" / "students_synthetic.csv"
    df, err = load_csv_safely(sample_path)

    assert err is None
    assert df is not None
    assert len(df) > 0
    assert "student_id" in df.columns


def test_load_csv_safely_missing_file() -> None:
    """Attempting to load a non-existent file should return None and a friendly error."""
    df, err = load_csv_safely(Path("data") / "sample" / "nonexistent_dataset.csv")

    assert df is None
    assert err is not None
    assert "File not found" in err


def test_load_csv_safely_empty_buffer() -> None:
    """Loading an empty buffer should return None and an error."""
    empty_buf = io.StringIO("")
    df, err = load_csv_safely(empty_buf)

    assert df is None
    assert err is not None
    assert "empty" in err.lower()


def test_build_quality_score_gauge() -> None:
    """Gauge generator should return a dark-themed plotly Figure with score tiers preserved."""
    fig_high = build_quality_score_gauge(95.0)
    assert isinstance(fig_high, go.Figure)
    assert fig_high.layout.paper_bgcolor == "#0E1117"
    assert fig_high.layout.plot_bgcolor == "#0E1117"
    assert fig_high.layout.font.color == "#F8FAFC"
    assert fig_high.data[0].number.font.color == "#F8FAFC"
    assert fig_high.data[0].title.font.color == "#F8FAFC"
    assert fig_high.data[0].gauge.axis.tickfont.color == "#94A3B8"
    assert fig_high.data[0].gauge.bgcolor == "#1E293B"
    assert fig_high.data[0].gauge.bar.color == "#10B981"  # Emerald for high score
    assert len(fig_high.data[0].gauge.steps) == 3        # Red, amber, green tiers preserved

    fig_mid = build_quality_score_gauge(65.0)
    assert isinstance(fig_mid, go.Figure)
    assert fig_mid.data[0].gauge.bar.color == "#F59E0B"   # Amber for medium score

    fig_low = build_quality_score_gauge(25.0)
    assert isinstance(fig_low, go.Figure)
    assert fig_low.data[0].gauge.bar.color == "#EF4444"   # Red for low score


def test_build_missing_values_chart() -> None:
    """Missing values chart generator should return a dark-themed chart with visible threshold line."""
    summary = {
        "threshold_percentage": 20.0,
        "by_column": {
            "col_a": {"missing_percentage": 10.0, "missing_count": 2, "exceeds_threshold": False},
            "col_b": {"missing_percentage": 45.0, "missing_count": 9, "exceeds_threshold": True},
        },
    }
    fig = build_missing_values_chart(summary)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"

    # Dark theme layout properties
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#161B22"
    assert fig.layout.font.color == "#F8FAFC"
    assert fig.layout.xaxis.title.font.color == "#F8FAFC"
    assert fig.layout.yaxis.title.font.color == "#F8FAFC"
    assert fig.layout.xaxis.tickfont.color == "#94A3B8"
    assert fig.layout.yaxis.tickfont.color == "#94A3B8"
    assert fig.layout.hoverlabel.bgcolor == "#1E293B"

    # Clearly visible threshold line and annotation
    assert len(fig.layout.shapes) > 0
    assert fig.layout.shapes[0].line.color == "#F87171"
    assert len(fig.layout.annotations) > 0
    assert fig.layout.annotations[0].font.color == "#FCA5A5"


def test_build_numerical_drift_chart() -> None:
    """Numerical drift chart should generate dark-themed histogram and box plots."""
    ref_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    curr_series = pd.Series([3.0, 4.0, 5.0, 6.0, 7.0])

    hist_fig = build_numerical_drift_chart("metric", ref_series, curr_series, chart_type="histogram")
    assert isinstance(hist_fig, go.Figure)
    assert len(hist_fig.data) == 2
    assert hist_fig.layout.paper_bgcolor == "#0E1117"
    assert hist_fig.layout.plot_bgcolor == "#161B22"
    assert hist_fig.layout.font.color == "#F8FAFC"
    assert hist_fig.layout.xaxis.title.font.color == "#F8FAFC"
    assert hist_fig.layout.yaxis.title.font.color == "#F8FAFC"
    assert hist_fig.layout.legend.font.color == "#F8FAFC"
    assert hist_fig.layout.hoverlabel.bgcolor == "#1E293B"

    box_fig = build_numerical_drift_chart("metric", ref_series, curr_series, chart_type="box")
    assert isinstance(box_fig, go.Figure)
    assert len(box_fig.data) == 2
    assert box_fig.layout.paper_bgcolor == "#0E1117"
    assert box_fig.layout.plot_bgcolor == "#161B22"
    assert box_fig.layout.font.color == "#F8FAFC"
    assert box_fig.layout.yaxis.title.font.color == "#F8FAFC"
    assert box_fig.layout.legend.font.color == "#F8FAFC"


def test_build_categorical_drift_chart() -> None:
    """Categorical drift chart should generate a dark-themed grouped bar chart."""
    ref_series = pd.Series(["A", "A", "B", "C"])
    curr_series = pd.Series(["A", "B", "B", "D"])

    fig = build_categorical_drift_chart("category", ref_series, curr_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.layout.barmode == "group"
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#161B22"
    assert fig.layout.font.color == "#F8FAFC"
    assert fig.layout.xaxis.title.font.color == "#F8FAFC"
    assert fig.layout.yaxis.title.font.color == "#F8FAFC"
    assert fig.layout.xaxis.tickfont.color == "#94A3B8"
    assert fig.layout.yaxis.tickfont.color == "#94A3B8"
    assert fig.layout.legend.font.color == "#F8FAFC"
    assert fig.layout.hoverlabel.bgcolor == "#1E293B"


def test_format_quality_export_json_and_csv() -> None:
    """Quality report formatter should serialize to valid JSON and tabular CSV strings."""
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    engine = DataQualityEngine()
    report = engine.evaluate(df)

    # JSON export
    json_str = format_quality_export(report, "json")
    parsed_json = json.loads(json_str)
    assert "overall_score" in parsed_json
    assert parsed_json["row_count"] == 3

    # CSV export
    csv_str = format_quality_export(report, "csv")
    assert "severity" in csv_str
    assert "category" in csv_str


def test_format_drift_export_json_and_csv() -> None:
    """Drift report formatter should serialize to valid JSON and tabular CSV strings."""
    ref_df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
    curr_df = pd.DataFrame({"val": [5.0, 6.0, 7.0]})
    engine = DataDriftEngine()
    report = engine.detect_drift(ref_df, curr_df)

    # JSON export
    json_str = format_drift_export(report, "json")
    parsed_json = json.loads(json_str)
    assert "overall_status" in parsed_json
    assert "features" in parsed_json

    # CSV export
    csv_str = format_drift_export(report, "csv")
    assert "feature" in csv_str
    assert "drift_detected" in csv_str


def test_apply_dark_theme_helper() -> None:
    """Helper should consistently configure dark background, light fonts, and hover colors."""
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    styled_fig = apply_dark_theme(fig, title="Custom Title")

    assert styled_fig.layout.paper_bgcolor == "#0E1117"
    assert styled_fig.layout.plot_bgcolor == "#161B22"
    assert styled_fig.layout.font.color == "#F8FAFC"
    assert styled_fig.layout.title.text == "Custom Title"
    assert styled_fig.layout.title.font.color == "#F8FAFC"
    assert styled_fig.layout.hoverlabel.bgcolor == "#1E293B"
    assert styled_fig.layout.xaxis.gridcolor == "#2D3748"
    assert styled_fig.layout.yaxis.gridcolor == "#2D3748"
    assert styled_fig.layout.xaxis.tickfont.color == "#94A3B8"


def test_extract_out_of_bounds_rows() -> None:
    """Helper should accurately isolate records violating numeric bounds."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [20, 22, 145],
        "score": [85, 90, 70],
    })

    # Age between 16 and 100 flags Charlie (145)
    invalid_age = extract_out_of_bounds_rows(df, "age", min_val=16, max_val=100)
    assert len(invalid_age) == 1
    assert invalid_age.iloc[0]["name"] == "Charlie"
    assert invalid_age.iloc[0]["age"] == 145

    # Age between 16 and 150 flags nothing
    valid_age = extract_out_of_bounds_rows(df, "age", min_val=16, max_val=150)
    assert len(valid_age) == 0

    # Score max 80 flags Alice (85) and Bob (90)
    invalid_score = extract_out_of_bounds_rows(df, "score", min_val=0, max_val=80)
    assert len(invalid_score) == 2
    assert list(invalid_score["name"]) == ["Alice", "Bob"]

    # Edge cases
    assert extract_out_of_bounds_rows(df, "nonexistent", min_val=0, max_val=10).empty
    assert extract_out_of_bounds_rows(df, "age", min_val=None, max_val=None).empty
    assert extract_out_of_bounds_rows(pd.DataFrame(), "age", min_val=10, max_val=50).empty


def test_format_active_rules_explanation() -> None:
    """Helper should generate clear, transparent explanations of all active validation rules."""
    # Configured rules
    rules = format_active_rules_explanation(
        missing_threshold=0.15,
        numeric_ranges={"age": {"min": 16, "max": 100}, "score": {"min": 0, "max": 100}},
        expected_schema={"name": "string", "age": "numeric"},
        categorical_allowed={"status": ["Active", "Pending"]},
    )
    explanation = "\n".join(rules)
    assert "15.0%" in explanation
    assert "age" in explanation
    assert "[16, 100]" in explanation
    assert "score" in explanation
    assert "Expected Schema" in explanation
    assert "status" in explanation

    # Default / unconfigured rules
    default_rules = format_active_rules_explanation(missing_threshold=0.20)
    default_text = "\n".join(default_rules)
    assert "20.0%" in default_text
    assert "No numeric bounds configured" in default_text
    assert "Flexible schema" in default_text
    assert "No categorical domain restrictions active" in default_text

