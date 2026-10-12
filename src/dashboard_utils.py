"""QualityLens - Reusable Dashboard Utilities and Visualization Helpers.

Provides helper functions for:
- Safe CSV parsing with user-friendly error messages
- Plotly interactive chart generation (Quality score gauge, missing values bar chart,
  overlaid numerical distribution plots, grouped categorical frequency bar charts)
- Data export serialization (JSON and tabular CSV)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.quality_engine import QualityReport
from src.drift_engine import DriftReport, FeatureDriftResult


def load_csv_safely(file_or_path: Any) -> tuple[pd.DataFrame | None, str | None]:
    """Parse a CSV file safely from a Path, string, or file-like buffer.

    Args:
        file_or_path: File path or file-like object (e.g. Streamlit UploadedFile).

    Returns:
        Tuple of (DataFrame or None, error_message or None).
    """
    if file_or_path is None:
        return None, "No file provided."

    try:
        # Check if Path-like
        if isinstance(file_or_path, (str, Path)):
            p = Path(file_or_path)
            if not p.exists():
                return None, f"File not found: '{p}'"
            if not p.is_file():
                return None, f"Path is not a regular file: '{p}'"
            df = pd.read_csv(p)
        else:
            # File-like object (e.g., Streamlit UploadedFile)
            df = pd.read_csv(file_or_path)

        if df.empty:
            return df, "Uploaded dataset contains 0 records."

        return df, None

    except pd.errors.EmptyDataError:
        return None, "The uploaded CSV file is empty (0 bytes or header-only)."
    except pd.errors.ParserError as pe:
        return None, f"CSV parsing error: {pe}"
    except UnicodeDecodeError:
        return None, "File encoding error: Please ensure the CSV is UTF-8 encoded."
    except Exception as e:
        return None, f"Unable to open file: {e}"


# -----------------------------------------------------------------------------
# Shared Dark Theme Tokens & Layout Helper
# -----------------------------------------------------------------------------
DARK_THEME_PAPER_BG = "#0E1117"
DARK_THEME_PLOT_BG = "#161B22"
DARK_THEME_TEXT_COLOR = "#F8FAFC"
DARK_THEME_MUTED_TEXT = "#94A3B8"
DARK_THEME_GRID_COLOR = "#2D3748"
DARK_THEME_BORDER_COLOR = "#334155"

DARK_THEME_LAYOUT_SETTINGS: dict[str, Any] = {
    "paper_bgcolor": DARK_THEME_PAPER_BG,
    "plot_bgcolor": DARK_THEME_PLOT_BG,
    "font": dict(family="sans-serif", color=DARK_THEME_TEXT_COLOR),
    "hoverlabel": dict(
        bgcolor="#1E293B",
        font=dict(color=DARK_THEME_TEXT_COLOR, size=12),
        bordercolor=DARK_THEME_BORDER_COLOR,
    ),
    "legend": dict(
        font=dict(color=DARK_THEME_TEXT_COLOR, size=11),
        bgcolor="rgba(14, 17, 23, 0.8)",
        bordercolor=DARK_THEME_BORDER_COLOR,
    ),
}


def apply_dark_theme(
    fig: go.Figure,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Apply consistent, high-contrast dark theme styling across all dashboard charts."""
    fig.update_layout(**DARK_THEME_LAYOUT_SETTINGS)

    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color=DARK_THEME_TEXT_COLOR, size=15),
            )
        )

    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)

    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    # Style axes ensuring readable light titles, gridlines, and tick labels
    fig.update_xaxes(
        gridcolor=DARK_THEME_GRID_COLOR,
        zerolinecolor=DARK_THEME_GRID_COLOR,
        tickfont=dict(color=DARK_THEME_MUTED_TEXT, size=11),
        title_font=dict(color=DARK_THEME_TEXT_COLOR, size=12),
    )
    fig.update_yaxes(
        gridcolor=DARK_THEME_GRID_COLOR,
        zerolinecolor=DARK_THEME_GRID_COLOR,
        tickfont=dict(color=DARK_THEME_MUTED_TEXT, size=11),
        title_font=dict(color=DARK_THEME_TEXT_COLOR, size=12),
    )

    return fig


def build_quality_score_gauge(score: float) -> go.Figure:
    """Build an interactive radial gauge chart visualizing the 0-100 quality score in dark theme."""
    clamped_score = max(0.0, min(100.0, float(score)))

    # Determine bar color based on score tier
    if clamped_score >= 80.0:
        bar_color = "#10B981"  # Emerald
    elif clamped_score >= 50.0:
        bar_color = "#F59E0B"  # Amber
    else:
        bar_color = "#EF4444"  # Rose/Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped_score,
            number={"suffix": " / 100", "font": {"size": 28, "color": DARK_THEME_TEXT_COLOR}},
            title={"text": "Quality Score", "font": {"size": 18, "color": DARK_THEME_TEXT_COLOR}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": DARK_THEME_MUTED_TEXT,
                    "tickfont": {"color": DARK_THEME_MUTED_TEXT},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": DARK_THEME_BORDER_COLOR,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.35)"},    # Red zone
                    {"range": [50, 80], "color": "rgba(245, 158, 11, 0.35)"},  # Amber zone
                    {"range": [80, 100], "color": "rgba(16, 185, 129, 0.35)"}, # Green zone
                ],
            },
        )
    )
    apply_dark_theme(fig)
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=DARK_THEME_PAPER_BG,
        plot_bgcolor=DARK_THEME_PAPER_BG,
    )
    return fig


def build_missing_values_chart(missing_summary: dict[str, Any]) -> go.Figure | None:
    """Build a horizontal bar chart displaying missing percentage per column in dark theme."""
    by_col = missing_summary.get("by_column", {})
    if not by_col:
        return None

    cols = list(by_col.keys())
    pcts = [by_col[c].get("missing_percentage", 0.0) for c in cols]
    counts = [by_col[c].get("missing_count", 0) for c in cols]
    exceeded = [by_col[c].get("exceeds_threshold", False) for c in cols]

    # Vibrant contrast colors for dark background:
    # Exceeded threshold: Vibrant Red (#EF4444)
    # Normal: High-contrast Sky Blue (#38BDF8)
    colors = ["#EF4444" if exc else "#38BDF8" for exc in exceeded]

    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=cols,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0F172A", width=0.5)),
            customdata=counts,
            hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}% (%{customdata} rows)<extra></extra>",
        )
    )

    threshold_pct = missing_summary.get("threshold_percentage", 20.0)
    fig.add_vline(
        x=threshold_pct,
        line_dash="dash",
        line_color="#F87171",
        line_width=2,
        annotation_text=f"Threshold ({threshold_pct:.1f}%)",
        annotation_position="top right",
        annotation_font=dict(color="#FCA5A5", size=11),
    )

    fig.update_layout(
        height=max(260, len(cols) * 32),
        margin=dict(l=40, r=40, t=40, b=30),
    )
    apply_dark_theme(
        fig,
        title="Missing Values by Column (%)",
        xaxis_title="Missing (%)",
        yaxis_title="Column",
    )
    fig.update_xaxes(range=[0, max(100.0, max(pcts, default=0.0) + 10)])
    return fig


def build_numerical_drift_chart(
    col: str,
    ref_series: pd.Series,
    curr_series: pd.Series,
    chart_type: str = "histogram",
) -> go.Figure:
    """Build an overlaid histogram or box plot comparing Reference vs. Current numeric distributions in dark theme."""
    ref_clean = pd.to_numeric(ref_series.dropna(), errors="coerce").dropna()
    curr_clean = pd.to_numeric(curr_series.dropna(), errors="coerce").dropna()

    fig = go.Figure()

    if chart_type == "box":
        fig.add_trace(
            go.Box(
                y=ref_clean,
                name="Reference Baseline",
                marker_color="#60A5FA",
                boxmean=True,
                line=dict(width=1.5),
            )
        )
        fig.add_trace(
            go.Box(
                y=curr_clean,
                name="Current Batch",
                marker_color="#FB923C",
                boxmean=True,
                line=dict(width=1.5),
            )
        )
        chart_title = f"Box Plot Comparison: [{col}]"
        x_title = ""
        y_title = col
        fig.update_layout(boxmode="group")
    else:  # histogram
        fig.add_trace(
            go.Histogram(
                x=ref_clean,
                name="Reference Baseline",
                opacity=0.70,
                marker_color="#3B82F6",
                histnorm="probability density",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=curr_clean,
                name="Current Batch",
                opacity=0.70,
                marker_color="#F97316",
                histnorm="probability density",
            )
        )
        chart_title = f"Distribution Density: [{col}] (Reference vs. Current)"
        x_title = col
        y_title = "Density"
        fig.update_layout(barmode="overlay")

    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_dark_theme(
        fig,
        title=chart_title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )

    return fig


def build_categorical_drift_chart(
    col: str,
    ref_series: pd.Series,
    curr_series: pd.Series,
) -> go.Figure:
    """Build a grouped bar chart comparing category frequency percentages in dark theme."""
    ref_counts = ref_series.dropna().astype(str).value_counts(normalize=True) * 100.0
    curr_counts = curr_series.dropna().astype(str).value_counts(normalize=True) * 100.0

    all_categories = sorted(list(set(ref_counts.index).union(set(curr_counts.index))))
    ref_vals = [ref_counts.get(c, 0.0) for c in all_categories]
    curr_vals = [curr_counts.get(c, 0.0) for c in all_categories]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_categories,
            y=ref_vals,
            name="Reference Baseline (%)",
            marker_color="#3B82F6",
            marker_line=dict(color="#1E293B", width=0.5),
        )
    )
    fig.add_trace(
        go.Bar(
            x=all_categories,
            y=curr_vals,
            name="Current Batch (%)",
            marker_color="#F97316",
            marker_line=dict(color="#1E293B", width=0.5),
        )
    )

    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_dark_theme(
        fig,
        title=f"Category Distribution (%): [{col}]",
        xaxis_title="Category",
        yaxis_title="Proportion (%)",
    )
    return fig


def format_quality_export(report: QualityReport, export_format: str = "json") -> str:
    """Serialize QualityReport to formatted JSON or CSV string."""
    if export_format.lower() == "csv":
        # Create tabular findings summary
        findings_data = []
        for f in report.findings:
            findings_data.append({
                "severity": f.severity,
                "category": f.category,
                "column": f.column or "N/A",
                "message": f.message,
            })
        df_findings = pd.DataFrame(findings_data)
        if df_findings.empty:
            df_findings = pd.DataFrame([{
                "severity": "INFO",
                "category": "general",
                "column": "all",
                "message": "Dataset conforms to all rules. Score: 100.0",
            }])
        return df_findings.to_csv(index=False)

    # Default: JSON format
    return json.dumps(report.to_dict(), indent=2)


def format_drift_export(report: DriftReport, export_format: str = "csv") -> str:
    """Serialize DriftReport to formatted CSV table or JSON string."""
    if export_format.lower() == "csv":
        rows = []
        for feat_name, res in report.feature_results.items():
            rows.append({
                "feature": feat_name,
                "feature_type": res.feature_type,
                "drift_detected": res.is_drift_detected,
                "severity": res.severity,
                "test_name": res.test_name,
                "statistic": res.statistic,
                "p_value": res.p_value,
                "threshold": res.threshold,
                "message": res.message,
            })
        df_drift = pd.DataFrame(rows)
        return df_drift.to_csv(index=False)

    return json.dumps(report.to_dict(), indent=2)


def extract_out_of_bounds_rows(
    df: pd.DataFrame,
    column: str,
    min_val: float | int | None = None,
    max_val: float | int | None = None,
) -> pd.DataFrame:
    """Extract rows where a numeric column violates min/max bounds.

    Args:
        df: Input DataFrame.
        column: Column name to evaluate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        DataFrame containing only rows violating the bounds, or empty DataFrame.
    """
    if min_val is None and max_val is None:
        return pd.DataFrame()

    if column not in df.columns:
        return pd.DataFrame()

    series = df[column]
    non_null = series.dropna()
    if non_null.empty:
        return pd.DataFrame()

    numeric_vals = pd.to_numeric(non_null, errors="coerce")
    coercion_failed = numeric_vals.isna()

    out_of_bounds = pd.Series(False, index=non_null.index)
    if min_val is not None:
        out_of_bounds = out_of_bounds | (numeric_vals < min_val)
    if max_val is not None:
        out_of_bounds = out_of_bounds | (numeric_vals > max_val)

    invalid_mask = coercion_failed | out_of_bounds
    invalid_indices = non_null[invalid_mask].index
    return df.loc[invalid_indices]


def format_active_rules_explanation(
    missing_threshold: float,
    numeric_ranges: dict[str, dict[str, float | int | None]] | None = None,
    expected_schema: dict[str, str] | None = None,
    categorical_allowed: dict[str, Any] | None = None,
) -> list[str]:
    """Generate human-readable explanations of all active validation rules.

    Args:
        missing_threshold: Missing cell threshold ratio (0.0 - 1.0).
        numeric_ranges: Optional dictionary of column min/max ranges.
        expected_schema: Optional dictionary of expected column types.
        categorical_allowed: Optional dictionary of allowed categorical values.

    Returns:
        List of formatted explanation strings describing active rules.
    """
    rules: list[str] = [
        f"**Missing Value Threshold**: Columns with > {missing_threshold * 100:.1f}% missing cells are flagged."
    ]

    if numeric_ranges:
        for col, bounds in numeric_ranges.items():
            min_v = bounds.get("min")
            max_v = bounds.get("max")
            if min_v is not None and max_v is not None:
                rules.append(
                    f"**Numeric Range [`{col}`]**: Valid range is [{min_v}, {max_v}]. "
                    f"Values < {min_v} or > {max_v} are flagged."
                )
            elif min_v is not None:
                rules.append(
                    f"**Numeric Range [`{col}`]**: Minimum allowed is {min_v}. "
                    f"Values < {min_v} are flagged."
                )
            elif max_v is not None:
                rules.append(
                    f"**Numeric Range [`{col}`]**: Maximum allowed is {max_v}. "
                    f"Values > {max_v} are flagged."
                )
    else:
        rules.append("**Numeric Range Validation**: No numeric bounds configured (all numeric values accepted).")

    if expected_schema:
        cols_str = ", ".join(f"`{c}` ({t})" for c, t in expected_schema.items())
        rules.append(f"**Expected Schema**: Contract enforces columns: {cols_str}.")
    else:
        rules.append("**Schema Conformance**: Flexible schema (no required column list enforced).")

    if categorical_allowed:
        for col, allowed in categorical_allowed.items():
            vals_str = ", ".join(repr(v) for v in sorted(list(allowed)))
            rules.append(f"**Categorical Domain [`{col}`]**: Allowed values: [{vals_str}].")
    else:
        rules.append("**Categorical Domain**: No categorical domain restrictions active.")

    return rules

