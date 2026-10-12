"""QualityLens - Interactive Data Quality and Drift Dashboard.

Streamlit web application for interactive data quality evaluation and statistical
drift detection.

Launch command:
    python -m streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

from src.quality_engine import DataQualityEngine, QualityReport
from src.drift_engine import DataDriftEngine, DriftReport
from src.dashboard_utils import (
    load_csv_safely,
    build_quality_score_gauge,
    build_missing_values_chart,
    build_numerical_drift_chart,
    build_categorical_drift_chart,
    format_quality_export,
    format_drift_export,
    format_active_rules_explanation,
    extract_out_of_bounds_rows,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QualityLens | Quality & Drift Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DIR = Path("data") / "sample"
SAMPLE_SYNTHETIC = SAMPLE_DIR / "students_synthetic.csv"
SAMPLE_REFERENCE = SAMPLE_DIR / "students_reference.csv"
SAMPLE_CURRENT = SAMPLE_DIR / "students_current.csv"


# -----------------------------------------------------------------------------
# Custom CSS for Modern, Clean Aesthetics
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #F8FAFC;
    }
    .badge-error {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-stable {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar_navigation() -> str:
    """Render the sidebar branding and view navigation."""
    with st.sidebar:
        st.markdown("## 🔍 **QualityLens**")
        st.caption("Data Quality & Drift Monitoring")
        st.divider()

        view = st.radio(
            "Select Module:",
            options=["Data Quality Audit", "Data Drift Detection", "Documentation & Guide"],
            index=0,
        )

        st.divider()
        st.markdown("### 📌 **System Information**")
        st.markdown("- **Engine**: Python 3.12")
        st.markdown("- **Framework**: Streamlit & Plotly")
        st.markdown("- **Deployment**: 100% Free & Local")
        st.caption("v1.3.0 | Milestones 1, 2, and 3")

    return view


# -----------------------------------------------------------------------------
# View 1: Data Quality Audit
# -----------------------------------------------------------------------------
def render_quality_audit_view() -> None:
    """Render the Data Quality Audit dashboard view."""
    st.markdown('<div class="main-title">📋 Data Quality Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Inspect single-dataset hygiene, missingness, schema adherence, '
        'and deterministic 0-100 quality scoring.</div>',
        unsafe_allow_html=True,
    )

    col_input, col_config = st.columns([1.1, 1.3])

    with col_input:
        st.markdown("#### 1. Dataset Input")
        use_sample = st.checkbox(
            "Load built-in synthetic student dataset (`students_synthetic.csv`)",
            value=True,
            help="Includes intentional missing values, duplicate rows, invalid age, and unexpected categorical value.",
        )

        uploaded_file = None
        if not use_sample:
            uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    # Load DataFrame safely
    df: pd.DataFrame | None = None
    err_msg: str | None = None

    if use_sample:
        if SAMPLE_SYNTHETIC.exists():
            df, err_msg = load_csv_safely(SAMPLE_SYNTHETIC)
        else:
            err_msg = f"Sample file not found at '{SAMPLE_SYNTHETIC}'."
    elif uploaded_file is not None:
        df, err_msg = load_csv_safely(uploaded_file)
    else:
        with col_config:
            st.markdown("#### 2. Audit Rules & Thresholds")
            st.info("👋 Upload a CSV file or check 'Load built-in synthetic student dataset' to begin.")
        return

    if err_msg:
        with col_config:
            st.markdown("#### 2. Audit Rules & Thresholds")
            st.error(f"❌ {err_msg}")
        return

    if df is None or df.empty:
        with col_config:
            st.markdown("#### 2. Audit Rules & Thresholds")
            st.warning("⚠️ The loaded dataset is empty (0 records).")
        return

    # Prepare Engine Configurations
    expected_schema: dict[str, str] | None = None
    numeric_ranges: dict[str, dict[str, float | int | None]] = {}
    categorical_allowed: dict[str, Any] | None = None

    with col_config:
        st.markdown("#### 2. Audit Rules & Thresholds")
        with st.expander("Configure Validation Parameters", expanded=(not use_sample)):
            missing_threshold = st.slider(
                "Missing Values Threshold per Column (%)",
                min_value=1,
                max_value=50,
                value=15,
                step=1,
                help="Columns with missing percentage exceeding this threshold are flagged.",
            ) / 100.0

            enable_student_schema = False
            if use_sample:
                enable_student_schema = st.checkbox(
                    "Apply Student Domain Schema & Bounds (Recommended for Sample Data)",
                    value=True,
                )
                if enable_student_schema:
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
                    categorical_allowed = {
                        "enrollment_status": ["Full-Time", "Part-Time", "Graduated", "On Leave"]
                    }

            # Custom validation rules for uploaded CSV or when sample schema is disabled
            if not use_sample or not enable_student_schema:
                candidate_numeric_cols = [
                    str(c)
                    for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    or pd.to_numeric(df[c], errors="coerce").notna().any()
                ]

                st.markdown("---")
                st.markdown("##### 📏 Numeric Range Validation")
                st.caption(
                    "Configure acceptable bounds [min, max] for numeric columns in your uploaded dataset. "
                    "Values outside bounds will be flagged as warnings and deducted from the quality score."
                )

                if candidate_numeric_cols:
                    selected_range_cols = st.multiselect(
                        "Select columns to validate ranges:",
                        options=candidate_numeric_cols,
                        default=[],
                        help="Choose columns to validate against minimum and/or maximum bounds.",
                    )

                    for col in selected_range_cols:
                        st.markdown(f"**Bounds for `{col}`**")
                        obs_series = pd.to_numeric(df[col], errors="coerce").dropna()
                        obs_min = f"{float(obs_series.min()):g}" if not obs_series.empty else "N/A"
                        obs_max = f"{float(obs_series.max()):g}" if not obs_series.empty else "N/A"
                        is_int = pd.api.types.is_integer_dtype(df[col])
                        step_val = 1.0 if is_int else 0.1

                        c_min, c_max = st.columns(2)
                        min_input = c_min.number_input(
                            f"Min allowed ({col})",
                            value=None,
                            step=step_val,
                            key=f"custom_min_{col}",
                            placeholder="None (no min)",
                            help=f"Observed in dataset: min = {obs_min}",
                        )
                        max_input = c_max.number_input(
                            f"Max allowed ({col})",
                            value=None,
                            step=step_val,
                            key=f"custom_max_{col}",
                            placeholder="None (no max)",
                            help=f"Observed in dataset: max = {obs_max}",
                        )

                        if min_input is not None and max_input is not None and min_input > max_input:
                            st.error(f"⚠️ Min ({min_input}) cannot be greater than Max ({max_input}) for '{col}'!")
                        else:
                            bounds: dict[str, float | int | None] = {}
                            if min_input is not None:
                                bounds["min"] = min_input
                            if max_input is not None:
                                bounds["max"] = max_input
                            if bounds:
                                numeric_ranges[col] = bounds
                else:
                    st.caption("No numeric columns detected in this dataset.")

                # Optional categorical allowed values
                candidate_cat_cols = [
                    str(c)
                    for c in df.columns
                    if c not in candidate_numeric_cols or df[c].dtype == "object"
                ]
                if candidate_cat_cols:
                    with st.expander("🏷️ Categorical Allowed Values (Optional)", expanded=False):
                        selected_cat_cols = st.multiselect(
                            "Select categorical columns:",
                            options=candidate_cat_cols,
                            default=[],
                            key="cat_cols_select",
                        )
                        for c_col in selected_cat_cols:
                            allowed_text = st.text_input(
                                f"Allowed values for `{c_col}` (comma-separated):",
                                key=f"cat_vals_{c_col}",
                                placeholder="e.g. Active, Inactive, Pending",
                            )
                            if allowed_text.strip():
                                parsed_vals = [v.strip() for v in allowed_text.split(",") if v.strip()]
                                if categorical_allowed is None:
                                    categorical_allowed = {}
                                categorical_allowed[c_col] = parsed_vals

        # Active Rules Explanation in config panel
        active_rules_list = format_active_rules_explanation(
            missing_threshold=missing_threshold,
            numeric_ranges=numeric_ranges if numeric_ranges else None,
            expected_schema=expected_schema,
            categorical_allowed=categorical_allowed,
        )

        with st.expander("📌 Active Validation Rules", expanded=True):
            for rule_text in active_rules_list:
                st.markdown(f"- {rule_text}")

    engine = DataQualityEngine(
        missing_threshold=missing_threshold,
        expected_schema=expected_schema,
        numeric_ranges=numeric_ranges if numeric_ranges else None,
        categorical_allowed_values=categorical_allowed,
    )

    # Execute Evaluation
    report: QualityReport = engine.evaluate(df)
    grouped_findings = report.findings_by_severity()

    st.divider()

    # 1. KPI Top Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Quality Score", f"{report.overall_score:.1f} / 100")
    with kpi2:
        st.metric("Total Rows", f"{report.row_count:,}")
    with kpi3:
        st.metric("Total Columns", f"{report.column_count}")
    with kpi4:
        st.metric(
            "Missing Cells",
            f"{report.missing_summary.get('total_missing_cells', 0):,}",
            f"{report.missing_summary.get('overall_missing_percentage', 0.0):.1f}%",
            delta_color="inverse",
        )
    with kpi5:
        st.metric(
            "Duplicate Rows",
            f"{report.duplicate_summary.get('duplicate_rows', 0):,}",
            f"{report.duplicate_summary.get('duplicate_percentage', 0.0):.1f}%",
            delta_color="inverse",
        )

    # 2. Charts Row
    chart_col1, chart_col2 = st.columns([1, 1.3])
    with chart_col1:
        gauge_fig = build_quality_score_gauge(report.overall_score)
        st.plotly_chart(gauge_fig, width="stretch")

    with chart_col2:
        miss_fig = build_missing_values_chart(report.missing_summary)
        if miss_fig:
            st.plotly_chart(miss_fig, width="stretch")
        else:
            st.success("🎉 No missing values found in this dataset!")

    # 3. Score Breakdown & Findings
    st.markdown("#### 3. Score Breakdown & Audit Findings")
    st.info(f"💡 **Scoring Explanation**: {report.score_summary.get('explanation', '')}")

    with st.expander("📋 View Active Rules & Validation Contracts", expanded=False):
        for r_text in active_rules_list:
            st.markdown(f"- {r_text}")

    tab_err, tab_warn, tab_info, tab_raw = st.tabs(
        [
            f"Errors ({len(grouped_findings['ERROR'])})",
            f"Warnings ({len(grouped_findings['WARNING'])})",
            f"Info ({len(grouped_findings['INFO'])})",
            "Dataset Preview",
        ]
    )

    with tab_err:
        if grouped_findings["ERROR"]:
            for f in grouped_findings["ERROR"]:
                st.error(f"**[{f.category.upper()}]** {f.message}")
        else:
            st.success("No critical error blockers detected.")

    with tab_warn:
        if grouped_findings["WARNING"]:
            for f in grouped_findings["WARNING"]:
                st.warning(f"**[{f.category.upper()}]** {f.message}")
                if f.category == "value_validation" and f.column:
                    samples = f.details.get("samples") or f.details.get("unexpected_values")
                    if samples:
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;🚩 **Detected Invalid Sample Values**: "
                            f"`{', '.join(str(s) for s in samples)}`"
                        )
                    if f.column in df.columns:
                        bounds = (numeric_ranges or {}).get(f.column, {})
                        invalid_rows = extract_out_of_bounds_rows(
                            df, f.column, min_val=bounds.get("min"), max_val=bounds.get("max")
                        )
                        if not invalid_rows.empty:
                            with st.expander(
                                f"🔍 Inspect {len(invalid_rows)} Invalid Record(s) in `{f.column}`",
                                expanded=True,
                            ):
                                st.dataframe(invalid_rows, width="stretch")
        else:
            st.success("No warnings detected.")

    with tab_info:
        for f in grouped_findings["INFO"]:
            st.info(f"**[{f.category.upper()}]** {f.message}")

    with tab_raw:
        st.dataframe(df.head(20), width="stretch")

    # 4. Report Exports
    st.divider()
    st.markdown("#### 4. Export Audit Report")
    exp_col1, exp_col2, _ = st.columns([1, 1, 2])

    with exp_col1:
        json_report = format_quality_export(report, "json")
        st.download_button(
            label="📥 Download JSON Report",
            data=json_report,
            file_name="qualitylens_audit_report.json",
            mime="application/json",
            width="stretch",
        )

    with exp_col2:
        csv_report = format_quality_export(report, "csv")
        st.download_button(
            label="📥 Download Findings CSV",
            data=csv_report,
            file_name="qualitylens_findings.csv",
            mime="text/csv",
            width="stretch",
        )


# -----------------------------------------------------------------------------
# View 2: Data Drift Detection
# -----------------------------------------------------------------------------
def render_drift_detection_view() -> None:
    """Render the Data Drift Detection dashboard view."""
    st.markdown('<div class="main-title">📈 Data Drift Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Compare baseline reference and current batch datasets using '
        '2-sample Kolmogorov-Smirnov tests (numeric) and Population Stability Index (categorical).</div>',
        unsafe_allow_html=True,
    )

    col_in, col_set = st.columns([1.2, 1])

    with col_in:
        st.markdown("#### 1. Datasets Input")
        use_sample_drift = st.checkbox(
            "Load built-in student cohorts (`students_reference.csv` vs `students_current.csv`)",
            value=True,
            help="Features stable age/gpa, shifted study hours, and new Remote-Online categorical status.",
        )

        ref_file = None
        curr_file = None
        if not use_sample_drift:
            ref_file = st.file_uploader("Upload Reference Baseline CSV", type=["csv"], key="ref_csv")
            curr_file = st.file_uploader("Upload Current Batch CSV", type=["csv"], key="curr_csv")

    with col_set:
        st.markdown("#### 2. Drift Thresholds")
        with st.expander("Configure Statistical Parameters", expanded=False):
            p_val_thresh = st.slider(
                "Numerical KS Significance Level (alpha)",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Drift is flagged if Kolmogorov-Smirnov p-value < alpha.",
            )
            psi_thresh = st.slider(
                "Categorical PSI Threshold",
                min_value=0.05,
                max_value=0.50,
                value=0.20,
                step=0.05,
                help="Standard rule of thumb: < 0.10 stable, >= 0.20 significant drift.",
            )
            dataset_share_thresh = st.slider(
                "Dataset Drift Share Threshold (%)",
                min_value=10,
                max_value=100,
                value=33,
                step=5,
                help="Percentage of drifted features required to flag overall dataset as DRIFT_DETECTED.",
            ) / 100.0

    # Load DataFrames safely
    ref_df: pd.DataFrame | None = None
    curr_df: pd.DataFrame | None = None
    ref_err: str | None = None
    curr_err: str | None = None

    if use_sample_drift:
        if SAMPLE_REFERENCE.exists() and SAMPLE_CURRENT.exists():
            ref_df, ref_err = load_csv_safely(SAMPLE_REFERENCE)
            curr_df, curr_err = load_csv_safely(SAMPLE_CURRENT)
        else:
            st.error("Sample reference or current file not found in data/sample/.")
            return
    elif ref_file and curr_file:
        ref_df, ref_err = load_csv_safely(ref_file)
        curr_df, curr_err = load_csv_safely(curr_file)
    else:
        st.info("👋 Upload both Reference and Current CSVs or check 'Load built-in student cohorts' to proceed.")
        return

    if ref_err:
        st.error(f"Reference File Error: {ref_err}")
        return
    if curr_err:
        st.error(f"Current File Error: {curr_err}")
        return

    if ref_df is None or curr_df is None or ref_df.empty or curr_df.empty:
        st.warning("⚠️ One or both datasets are empty.")
        return

    # Feature selection
    common_cols = sorted(list(set(ref_df.columns).intersection(set(curr_df.columns))))
    # Filter out student_id by default if sample dataset is used
    default_selected = [c for c in common_cols if c != "student_id"] if use_sample_drift else common_cols

    selected_features = st.multiselect(
        "Select Features to Evaluate for Drift:",
        options=common_cols,
        default=default_selected,
        help="Select which features to monitor. Exclude ID or timestamp index columns.",
    )

    if not selected_features:
        st.warning("Please select at least one feature to evaluate.")
        return

    # Run Drift Engine
    engine = DataDriftEngine(
        numerical_p_value_threshold=p_val_thresh,
        categorical_psi_threshold=psi_thresh,
        dataset_drift_share_threshold=dataset_share_thresh,
        features=selected_features,
    )
    drift_report: DriftReport = engine.detect_drift(ref_df, curr_df)

    st.divider()

    # 1. Overall Status KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if drift_report.overall_status == "DRIFT_DETECTED":
            st.error(f"🚨 Status: {drift_report.overall_status}")
        elif drift_report.overall_status == "WARNING":
            st.warning(f"⚠️ Status: {drift_report.overall_status}")
        else:
            st.success(f"✅ Status: {drift_report.overall_status}")

    with k2:
        st.metric("Features Evaluated", f"{drift_report.total_features_evaluated}")
    with k3:
        st.metric(
            "Drifted Features",
            f"{drift_report.drifted_features_count} / {drift_report.total_features_evaluated}",
            f"{drift_report.drift_share:.1f}%",
            delta_color="inverse",
        )
    with k4:
        st.metric("Reference / Current Rows", f"{drift_report.reference_row_count:,} / {drift_report.current_row_count:,}")

    # Schema discrepancies notes
    if drift_report.missing_in_current or drift_report.new_in_current:
        with st.expander("Schema Differences Detected", expanded=True):
            if drift_report.missing_in_current:
                st.warning(f"Columns missing in Current: {', '.join(drift_report.missing_in_current)}")
            if drift_report.new_in_current:
                st.info(f"New columns in Current: {', '.join(drift_report.new_in_current)}")

    st.divider()

    # 2. Interactive Feature Drilldown
    st.markdown("#### 3. Interactive Feature Distribution Inspector")

    drill_col, chart_type_col = st.columns([2, 1])
    with drill_col:
        inspected_feat = st.selectbox(
            "Select Feature to Inspect:",
            options=list(drift_report.feature_results.keys()),
        )
    with chart_type_col:
        chart_mode = st.radio("Chart Type", ["Histogram", "Box Plot"], horizontal=True)

    if inspected_feat:
        feat_res = drift_report.feature_results[inspected_feat]

        # Feature Summary Banner
        c_left, c_right = st.columns([1.5, 2.5])
        with c_left:
            st.markdown(f"**Feature**: `{inspected_feat}` ({feat_res.feature_type})")
            if feat_res.is_drift_detected:
                st.error(f"🔴 DRIFT DETECTED | Severity: **{feat_res.severity}**")
            else:
                st.success(f"🟢 STABLE | Severity: **{feat_res.severity}**")

            if feat_res.test_name == "kolmogorov_smirnov":
                st.markdown(f"- **Test**: 2-Sample Kolmogorov-Smirnov")
                st.markdown(f"- **KS Statistic**: `{feat_res.statistic:.4f}`")
                st.markdown(f"- **p-value**: `{feat_res.p_value:.4e}` (threshold: `{feat_res.threshold}`)")
            elif feat_res.test_name == "population_stability_index":
                st.markdown(f"- **Test**: Population Stability Index (PSI)")
                st.markdown(f"- **PSI**: `{feat_res.statistic:.4f}` (threshold: `{feat_res.threshold}`)")
                tvd = feat_res.details.get("total_variation_distance")
                if tvd is not None:
                    st.markdown(f"- **Total Variation Distance (TVD)**: `{tvd:.4f}`")

            if feat_res.message:
                st.caption(f"📝 {feat_res.message}")

            unseen = feat_res.details.get("unseen_categories", [])
            if unseen:
                st.warning(f"Unseen Categories in Current: {unseen}")

        with c_right:
            # Render Distribution Chart
            if feat_res.feature_type == "numerical":
                fig = build_numerical_drift_chart(
                    inspected_feat,
                    ref_df[inspected_feat],
                    curr_df[inspected_feat],
                    chart_type="box" if chart_mode == "Box Plot" else "histogram",
                )
                st.plotly_chart(fig, width="stretch")
            elif feat_res.feature_type == "categorical":
                fig = build_categorical_drift_chart(
                    inspected_feat,
                    ref_df[inspected_feat],
                    curr_df[inspected_feat],
                )
                st.plotly_chart(fig, width="stretch")

    # 3. Full Feature Table
    st.divider()
    st.markdown("#### 4. Complete Feature Drift Summary Table")

    table_data = []
    for f_name, r in drift_report.feature_results.items():
        table_data.append({
            "Feature": f_name,
            "Type": r.feature_type,
            "Drift Detected": "YES" if r.is_drift_detected else "NO",
            "Severity": r.severity,
            "Test": r.test_name,
            "Statistic": r.statistic,
            "p-value": r.p_value,
            "Threshold": r.threshold,
            "Message": r.message,
        })
    st.dataframe(pd.DataFrame(table_data), width="stretch")

    # 4. Exports
    st.divider()
    st.markdown("#### 5. Export Drift Report")
    d_exp1, d_exp2, _ = st.columns([1, 1, 2])

    with d_exp1:
        drift_json = format_drift_export(drift_report, "json")
        st.download_button(
            label="📥 Download Drift JSON",
            data=drift_json,
            file_name="qualitylens_drift_report.json",
            mime="application/json",
            width="stretch",
        )

    with d_exp2:
        drift_csv = format_drift_export(drift_report, "csv")
        st.download_button(
            label="📥 Download Drift Summary CSV",
            data=drift_csv,
            file_name="qualitylens_drift_summary.csv",
            mime="text/csv",
            width="stretch",
        )


# -----------------------------------------------------------------------------
# View 3: Documentation & Guide
# -----------------------------------------------------------------------------
def render_documentation_view() -> None:
    """Render framework documentation, metric definitions, and user guides."""
    st.markdown('<div class="main-title">ℹ️ QualityLens Guide & Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Reference documentation for data hygiene rules, scoring formulas, '
        'and statistical drift testing.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("1. Quality Checks vs. Data Drift Detection", expanded=True):
        st.markdown(
            """
            - **Data Quality Audit (Milestone 1)**: Evaluates a single dataset in isolation against explicit
              contracts (null rates, duplicate records, range bounds, schema types). A low score indicates
              **technical defects** in the dataset.
            - **Data Drift Detection (Milestone 2)**: Compares incoming batch data against a reference baseline.
              A drift alert indicates that **statistical distributions have changed**. Drift does not mean the
              data is invalid; it signals that machine learning models or statistical assumptions should be reviewed.
            """
        )

    with st.expander("2. Statistical Drift Metrics Explained", expanded=True):
        st.markdown(
            """
            #### A. Two-Sample Kolmogorov-Smirnov (KS) Test (Numerical)
            - **Method**: Evaluates the maximum vertical separation ($D$) between the cumulative distribution
              functions of the reference baseline and current batch:
              $$D = \\sup_x |F_{\\text{ref}}(x) - F_{\\text{curr}}(x)|$$
            - **Interpretation**: If $p\\text{-value} < \\alpha$ (default $\\alpha = 0.05$), the null hypothesis
              of identical distributions is rejected, and drift is flagged.

            #### B. Population Stability Index (PSI) (Categorical)
            - **Formula**:
              $$\\text{PSI} = \\sum_{i=1}^k (P_{\\text{curr}, i} - P_{\\text{ref}, i}) \\times \\ln\\left(\\frac{P_{\\text{curr}, i}}{P_{\\text{ref}, i}}\\right)$$
            - **Standard Rule of Thumb**:
              - $\\text{PSI} < 0.10$: **Stable** (insignificant change).
              - $0.10 \\le \\text{PSI} < 0.20$: **Moderate shift** (monitor).
              - $\\text{PSI} \\ge 0.20$: **Significant drift** (alert).

            #### C. Total Variation Distance (TVD)
            - Measures total probability mass that shifted across categories:
              $$\\text{TVD} = \\frac{1}{2} \\sum_{i=1}^k |P_{\\text{curr}, i} - P_{\\text{ref}, i}|$$
            """
        )

    with st.expander("3. Deterministic Quality Score Formulation", expanded=True):
        st.markdown(
            """
            - **Base Score**: 100.0 points.
            - **Completeness Deductions**: Up to 30 pts (overall missing ratio + threshold penalties).
            - **Uniqueness Deductions**: Up to 20 pts (duplicate row ratio).
            - **Schema Conformance**: Up to 25 pts (missing expected columns & data type mismatches).
            - **Value Validity**: Up to 25 pts (out-of-range numeric & disallowed categories).
            - **Empty Dataset**: Automatic 0.0 score.
            """
        )


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main() -> None:
    selected_view = render_sidebar_navigation()

    if selected_view == "Data Quality Audit":
        render_quality_audit_view()
    elif selected_view == "Data Drift Detection":
        render_drift_detection_view()
    else:
        render_documentation_view()


if __name__ == "__main__":
    main()
