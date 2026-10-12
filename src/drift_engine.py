"""QualityLens - Data Drift Detection Engine.

This module provides the DataDriftEngine and DriftReport classes for comparing
reference (baseline) and current datasets to detect distribution shifts across
numerical and categorical features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FeatureDriftResult:
    """Statistical drift evaluation result for a single feature.

    Severities:
        - "NONE": No statistically significant drift detected.
        - "LOW": Slight shift detected below critical alert thresholds.
        - "MODERATE": Measurable distribution shift (e.g., PSI between 0.10 and 0.25).
        - "HIGH": Severe distribution shift (e.g., KS p-value < 0.001 or PSI >= 0.25).
        - "ERROR": Column missing, empty, or data type incompatible between datasets.
    """

    feature_name: str
    feature_type: str  # "numerical", "categorical", "unsupported"
    is_drift_detected: bool
    severity: str  # "NONE", "LOW", "MODERATE", "HIGH", "ERROR"
    test_name: str  # "kolmogorov_smirnov", "population_stability_index", "none"
    statistic: float | None
    p_value: float | None
    threshold: float
    reference_stats: dict[str, Any] = field(default_factory=dict)
    current_stats: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)


@dataclass
class DriftReport:
    """Structured report containing feature-level metrics and dataset-level drift status."""

    reference_row_count: int
    current_row_count: int
    total_features_evaluated: int
    drifted_features_count: int
    drift_share: float  # Percentage of evaluated features that drifted (0.0 to 100.0)
    is_dataset_drifted: bool
    overall_status: str  # "STABLE", "DRIFT_DETECTED", "WARNING", "ERROR"
    feature_results: dict[str, FeatureDriftResult] = field(default_factory=dict)
    missing_in_current: list[str] = field(default_factory=list)
    new_in_current: list[str] = field(default_factory=list)
    disclaimer: str = (
        "Data drift indicates a statistical shift in feature distributions between the "
        "baseline and current datasets. Drift does NOT inherently indicate that the current "
        "data is corrupted or invalid. It reflects changing patterns, real-world dynamics, "
        "or upstream pipeline changes that may affect downstream analytical models."
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert full report into a serializable dictionary."""
        return {
            "reference_row_count": self.reference_row_count,
            "current_row_count": self.current_row_count,
            "total_features_evaluated": self.total_features_evaluated,
            "drifted_features_count": self.drifted_features_count,
            "drift_share": round(self.drift_share, 2),
            "is_dataset_drifted": self.is_dataset_drifted,
            "overall_status": self.overall_status,
            "missing_in_current": self.missing_in_current,
            "new_in_current": self.new_in_current,
            "disclaimer": self.disclaimer,
            "features": {k: v.to_dict() for k, v in self.feature_results.items()},
        }

    def summary_text(self) -> str:
        """Generate a concise, professional human-readable drift report."""
        lines = [
            "=" * 74,
            " QUALITYLENS DATA DRIFT MONITORING REPORT",
            "=" * 74,
            f"Reference Dataset : {self.reference_row_count:,} records",
            f"Current Dataset   : {self.current_row_count:,} records",
            f"Overall Status    : {self.overall_status}",
            f"Features Analyzed : {self.total_features_evaluated}",
            f"Drifted Features  : {self.drifted_features_count} / {self.total_features_evaluated} "
            f"({self.drift_share:.1f}%)",
            "-" * 74,
        ]

        if self.missing_in_current or self.new_in_current:
            lines.append("SCHEMA COMPARISON:")
            if self.missing_in_current:
                lines.append(f"   - Columns missing in current : {', '.join(self.missing_in_current)}")
            if self.new_in_current:
                lines.append(f"   - New columns in current     : {', '.join(self.new_in_current)}")
            lines.append("-" * 74)

        lines.append("FEATURE DRIFT BREAKDOWN:")

        for feat_name, res in self.feature_results.items():
            drift_flag = "DRIFT DETECTED" if res.is_drift_detected else "STABLE"
            lines.append(
                f" - [{feat_name}] ({res.feature_type}) -> {drift_flag} [Severity: {res.severity}]"
            )

            if res.test_name == "kolmogorov_smirnov":
                stat_val = f"{res.statistic:.4f}" if res.statistic is not None else "N/A"
                pval_val = f"{res.p_value:.4e}" if res.p_value is not None else "N/A"
                lines.append(
                    f"   Test: 2-Sample Kolmogorov-Smirnov | KS-Stat: {stat_val} | "
                    f"p-value: {pval_val} (Threshold: p < {res.threshold})"
                )
                r_stats = res.reference_stats
                c_stats = res.current_stats
                if "mean" in r_stats and "mean" in c_stats:
                    lines.append(
                        f"   Ref Mean: {r_stats['mean']:.2f} (std: {r_stats['std']:.2f}, med: {r_stats['median']:.2f}) | "
                        f"Curr Mean: {c_stats['mean']:.2f} (std: {c_stats['std']:.2f}, med: {c_stats['median']:.2f})"
                    )

            elif res.test_name == "population_stability_index":
                stat_val = f"{res.statistic:.4f}" if res.statistic is not None else "N/A"
                tvd_val = res.details.get("total_variation_distance")
                tvd_str = f" | TVD: {tvd_val:.4f}" if tvd_val is not None else ""
                lines.append(
                    f"   Test: Population Stability Index (PSI) | PSI: {stat_val}{tvd_str} "
                    f"(Threshold: PSI >= {res.threshold})"
                )
                unseen = res.details.get("unseen_categories", [])
                if unseen:
                    lines.append(f"   Unseen Categories in Current: {unseen}")

            if res.message:
                lines.append(f"   Note: {res.message}")

            lines.append("")

        lines.extend([
            "-" * 74,
            "INTERPRETATION GUIDANCE:",
            f" - {self.disclaimer}",
            "=" * 74,
        ])

        return "\n".join(lines)


class DataDriftEngine:
    """Reusable engine for evaluating data distribution shift between two datasets.

    Supports:
    - Numerical features: 2-sample Kolmogorov-Smirnov test, Wasserstein distance,
      and summary statistics comparison (mean, std, quantiles).
    - Categorical features: Population Stability Index (PSI) and Total Variation
      Distance (TVD), with explicit handling of unseen categories and missing values.
    - Graceful handling of empty frames, schema discrepancies, and small samples.

    Guarantees non-destructive evaluation (input DataFrames are never altered).
    """

    def __init__(
        self,
        numerical_p_value_threshold: float = 0.05,
        categorical_psi_threshold: float = 0.20,
        dataset_drift_share_threshold: float = 0.33,
        min_sample_size: int = 5,
        features: Sequence[str] | None = None,
    ) -> None:
        """Initialize the drift detection engine.

        Args:
            numerical_p_value_threshold: Significance level alpha for Kolmogorov-Smirnov
                test (default 0.05). Drift is flagged if p-value < threshold.
            categorical_psi_threshold: Threshold for Population Stability Index (default 0.20).
                PSI < 0.10 is stable, 0.10-0.25 is moderate shift, >= 0.20 is flagged drift.
            dataset_drift_share_threshold: Proportion of features that must drift to mark
                the entire dataset as drifted (default 0.33 or 33%).
            min_sample_size: Minimum number of non-null observations required to run
                statistical hypothesis testing (default 5).
            features: Optional list of specific column names to analyze. If None, all
                common columns will be evaluated.
        """
        if not (0.0 < numerical_p_value_threshold < 1.0):
            raise ValueError(f"numerical_p_value_threshold must be between 0.0 and 1.0, got {numerical_p_value_threshold}")
        if categorical_psi_threshold <= 0.0:
            raise ValueError(f"categorical_psi_threshold must be positive, got {categorical_psi_threshold}")
        if not (0.0 <= dataset_drift_share_threshold <= 1.0):
            raise ValueError(f"dataset_drift_share_threshold must be between 0.0 and 1.0, got {dataset_drift_share_threshold}")

        self.numerical_p_value_threshold = numerical_p_value_threshold
        self.categorical_psi_threshold = categorical_psi_threshold
        self.dataset_drift_share_threshold = dataset_drift_share_threshold
        self.min_sample_size = min_sample_size
        self.target_features = list(features) if features is not None else None

    def detect_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> DriftReport:
        """Evaluate data drift between reference baseline and current batch.

        Args:
            reference_df: Baseline reference pandas DataFrame.
            current_df: Current target pandas DataFrame to compare against reference.

        Returns:
            DriftReport containing detailed feature-level and dataset-level drift results.
        """
        if not isinstance(reference_df, pd.DataFrame) or not isinstance(current_df, pd.DataFrame):
            raise TypeError("Both reference_df and current_df must be pandas DataFrames.")

        ref_rows = len(reference_df)
        curr_rows = len(current_df)

        # Handle empty dataset edge cases
        if ref_rows == 0 or curr_rows == 0:
            return self._handle_empty_datasets(reference_df, current_df, ref_rows, curr_rows)

        ref_cols = set(reference_df.columns)
        curr_cols = set(current_df.columns)

        missing_in_current = sorted(list(ref_cols - curr_cols))
        new_in_current = sorted(list(curr_cols - ref_cols))

        # Determine features to evaluate
        if self.target_features is not None:
            cols_to_check = [c for c in self.target_features if c in ref_cols]
        else:
            cols_to_check = sorted(list(ref_cols))

        feature_results: dict[str, FeatureDriftResult] = {}
        drifted_count = 0

        for col in cols_to_check:
            # If feature is completely missing from current
            if col not in current_df.columns:
                res = FeatureDriftResult(
                    feature_name=col,
                    feature_type="missing",
                    is_drift_detected=True,
                    severity="ERROR",
                    test_name="none",
                    statistic=None,
                    p_value=None,
                    threshold=0.0,
                    message=f"Column '{col}' is present in reference but missing in current dataset.",
                )
                feature_results[col] = res
                drifted_count += 1
                continue

            ref_series = reference_df[col]
            curr_series = current_df[col]

            # Incompatible types check
            is_ref_numeric = pd.api.types.is_numeric_dtype(ref_series)
            is_curr_numeric = pd.api.types.is_numeric_dtype(curr_series)

            if is_ref_numeric != is_curr_numeric:
                res = FeatureDriftResult(
                    feature_name=col,
                    feature_type="incompatible",
                    is_drift_detected=True,
                    severity="ERROR",
                    test_name="none",
                    statistic=None,
                    p_value=None,
                    threshold=0.0,
                    message=(
                        f"Incompatible data types for '{col}': reference is '{ref_series.dtype}', "
                        f"current is '{curr_series.dtype}'."
                    ),
                )
                feature_results[col] = res
                drifted_count += 1
                continue

            if is_ref_numeric:
                res = self._evaluate_numerical_drift(col, ref_series, curr_series)
            else:
                res = self._evaluate_categorical_drift(col, ref_series, curr_series)

            feature_results[col] = res
            if res.is_drift_detected:
                drifted_count += 1

        total_evaluated = len(feature_results)
        drift_share = (drifted_count / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
        is_dataset_drifted = drift_share >= (self.dataset_drift_share_threshold * 100.0)

        if total_evaluated == 0:
            overall_status = "ERROR"
        elif is_dataset_drifted:
            overall_status = "DRIFT_DETECTED"
        elif drifted_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "STABLE"

        return DriftReport(
            reference_row_count=ref_rows,
            current_row_count=curr_rows,
            total_features_evaluated=total_evaluated,
            drifted_features_count=drifted_count,
            drift_share=round(drift_share, 2),
            is_dataset_drifted=is_dataset_drifted,
            overall_status=overall_status,
            feature_results=feature_results,
            missing_in_current=missing_in_current,
            new_in_current=new_in_current,
        )

    # -------------------------------------------------------------------------
    # Numerical Feature Drift
    # -------------------------------------------------------------------------

    def _evaluate_numerical_drift(
        self, col: str, ref_series: pd.Series, curr_series: pd.Series
    ) -> FeatureDriftResult:
        """Run 2-sample Kolmogorov-Smirnov test and compare distribution statistics."""
        # Clean nulls safely without mutating input Series
        ref_valid = pd.to_numeric(ref_series.dropna(), errors="coerce").dropna()
        curr_valid = pd.to_numeric(curr_series.dropna(), errors="coerce").dropna()

        ref_stats = self._compute_numerical_stats(ref_series, ref_valid)
        curr_stats = self._compute_numerical_stats(curr_series, curr_valid)

        # Handle low sample size or all-null cases
        if len(ref_valid) < self.min_sample_size or len(curr_valid) < self.min_sample_size:
            return FeatureDriftResult(
                feature_name=col,
                feature_type="numerical",
                is_drift_detected=False,
                severity="LOW",
                test_name="kolmogorov_smirnov",
                statistic=None,
                p_value=None,
                threshold=self.numerical_p_value_threshold,
                reference_stats=ref_stats,
                current_stats=curr_stats,
                message=(
                    f"Insufficient non-null observations (ref: {len(ref_valid)}, curr: {len(curr_valid)}) "
                    f"to perform Kolmogorov-Smirnov test (min required: {self.min_sample_size})."
                ),
            )

        # Two-sample Kolmogorov-Smirnov test
        ks_res = stats.ks_2samp(ref_valid, curr_valid)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # Optional physical distance metric (Wasserstein distance / Earth Mover's Distance)
        try:
            wasserstein_dist = float(stats.wasserstein_distance(ref_valid, curr_valid))
        except Exception:
            wasserstein_dist = None

        is_drift = p_val < self.numerical_p_value_threshold

        # Determine severity
        if not is_drift:
            severity = "NONE"
            message = "Distribution is statistically consistent with reference."
        elif ks_stat >= 0.35 or p_val < 0.001:
            severity = "HIGH"
            message = f"Statistically significant divergence detected (KS stat: {ks_stat:.4f}, p-val: {p_val:.4e})."
        elif ks_stat >= 0.15:
            severity = "MODERATE"
            message = f"Moderate distribution shift detected (KS stat: {ks_stat:.4f}, p-val: {p_val:.4e})."
        else:
            severity = "LOW"
            message = f"Mild distribution shift detected (p-val: {p_val:.4e})."

        details = {
            "wasserstein_distance": round(wasserstein_dist, 4) if wasserstein_dist is not None else None,
            "ref_sample_size": len(ref_valid),
            "curr_sample_size": len(curr_valid),
        }

        return FeatureDriftResult(
            feature_name=col,
            feature_type="numerical",
            is_drift_detected=is_drift,
            severity=severity,
            test_name="kolmogorov_smirnov",
            statistic=round(ks_stat, 4),
            p_value=round(p_val, 6),
            threshold=self.numerical_p_value_threshold,
            reference_stats=ref_stats,
            current_stats=curr_stats,
            details=details,
            message=message,
        )

    def _compute_numerical_stats(
        self, original_series: pd.Series, clean_series: pd.Series
    ) -> dict[str, Any]:
        """Compute comprehensive descriptive summary statistics."""
        total_len = len(original_series)
        missing_cnt = int(original_series.isna().sum())
        missing_pct = round((missing_cnt / total_len * 100.0), 2) if total_len > 0 else 0.0

        if len(clean_series) == 0:
            return {
                "count": 0,
                "missing_count": missing_cnt,
                "missing_percentage": missing_pct,
                "mean": None,
                "std": None,
                "median": None,
                "q25": None,
                "q75": None,
                "min": None,
                "max": None,
            }

        return {
            "count": int(len(clean_series)),
            "missing_count": missing_cnt,
            "missing_percentage": missing_pct,
            "mean": round(float(clean_series.mean()), 4),
            "std": round(float(clean_series.std()), 4) if len(clean_series) > 1 else 0.0,
            "median": round(float(clean_series.median()), 4),
            "q25": round(float(clean_series.quantile(0.25)), 4),
            "q75": round(float(clean_series.quantile(0.75)), 4),
            "min": round(float(clean_series.min()), 4),
            "max": round(float(clean_series.max()), 4),
        }

    # -------------------------------------------------------------------------
    # Categorical Feature Drift
    # -------------------------------------------------------------------------

    def _evaluate_categorical_drift(
        self, col: str, ref_series: pd.Series, curr_series: pd.Series
    ) -> FeatureDriftResult:
        """Run Population Stability Index (PSI) and Total Variation Distance."""
        # Convert non-null values to strings
        ref_valid = ref_series.dropna().astype(str)
        curr_valid = curr_series.dropna().astype(str)

        total_ref = len(ref_series)
        total_curr = len(curr_series)

        ref_missing_pct = round(int(ref_series.isna().sum()) / total_ref * 100.0, 2) if total_ref > 0 else 0.0
        curr_missing_pct = round(int(curr_series.isna().sum()) / total_curr * 100.0, 2) if total_curr > 0 else 0.0

        ref_stats = {
            "count": len(ref_valid),
            "missing_percentage": ref_missing_pct,
            "unique_categories": int(ref_valid.nunique()),
        }
        curr_stats = {
            "count": len(curr_valid),
            "missing_percentage": curr_missing_pct,
            "unique_categories": int(curr_valid.nunique()),
        }

        if len(ref_valid) == 0 or len(curr_valid) == 0:
            return FeatureDriftResult(
                feature_name=col,
                feature_type="categorical",
                is_drift_detected=False,
                severity="LOW",
                test_name="population_stability_index",
                statistic=None,
                p_value=None,
                threshold=self.categorical_psi_threshold,
                reference_stats=ref_stats,
                current_stats=curr_stats,
                message="Cannot evaluate categorical drift because one or both datasets have 0 valid entries.",
            )

        ref_counts = ref_valid.value_counts(normalize=True)
        curr_counts = curr_valid.value_counts(normalize=True)

        all_categories = sorted(list(set(ref_counts.index).union(set(curr_counts.index))))
        unseen_in_ref = sorted(list(set(curr_counts.index) - set(ref_counts.index)))
        missing_in_curr = sorted(list(set(ref_counts.index) - set(curr_counts.index)))

        # Calculate Population Stability Index (PSI) with epsilon smoothing
        eps = 1e-4
        psi_total = 0.0
        tvd_total = 0.0
        category_breakdown: dict[str, dict[str, float]] = {}

        for cat in all_categories:
            p_ref = float(ref_counts.get(cat, 0.0))
            p_curr = float(curr_counts.get(cat, 0.0))

            tvd_total += abs(p_curr - p_ref)

            # Epsilon smoothing prevents log(0) and division by zero
            smoothed_ref = max(p_ref, eps)
            smoothed_curr = max(p_curr, eps)

            psi_contribution = (smoothed_curr - smoothed_ref) * np.log(smoothed_curr / smoothed_ref)
            psi_total += psi_contribution

            category_breakdown[cat] = {
                "ref_frequency": round(p_ref, 4),
                "curr_frequency": round(p_curr, 4),
                "shift": round(p_curr - p_ref, 4),
            }

        psi_value = round(float(psi_total), 4)
        tvd_value = round(float(tvd_total * 0.5), 4)  # TVD is 0.5 * sum(|p_curr - p_ref|)

        is_drift = psi_value >= self.categorical_psi_threshold

        # Classify severity
        if psi_value < 0.10:
            severity = "NONE"
            message = "Categorical frequencies are consistent with reference."
        elif psi_value < 0.25:
            severity = "MODERATE"
            message = f"Moderate frequency shift observed (PSI: {psi_value:.4f}, TVD: {tvd_value:.4f})."
        else:
            severity = "HIGH"
            message = f"Significant category distribution shift detected (PSI: {psi_value:.4f}, TVD: {tvd_value:.4f})."

        if unseen_in_ref:
            message += f" Contains new unseen categories: {unseen_in_ref}."

        details = {
            "total_variation_distance": tvd_value,
            "unseen_categories": unseen_in_ref,
            "missing_categories": missing_in_curr,
            "frequency_breakdown": category_breakdown,
        }

        return FeatureDriftResult(
            feature_name=col,
            feature_type="categorical",
            is_drift_detected=is_drift,
            severity=severity,
            test_name="population_stability_index",
            statistic=psi_value,
            p_value=None,
            threshold=self.categorical_psi_threshold,
            reference_stats=ref_stats,
            current_stats=curr_stats,
            details=details,
            message=message,
        )

    # -------------------------------------------------------------------------
    # Edge Case Handler
    # -------------------------------------------------------------------------

    def _handle_empty_datasets(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        ref_rows: int,
        curr_rows: int,
    ) -> DriftReport:
        """Handle 0-row datasets gracefully."""
        all_cols = sorted(list(set(reference_df.columns).union(set(current_df.columns))))
        feature_results: dict[str, FeatureDriftResult] = {}

        for col in all_cols:
            feature_results[col] = FeatureDriftResult(
                feature_name=col,
                feature_type="unknown",
                is_drift_detected=False,
                severity="ERROR",
                test_name="none",
                statistic=None,
                p_value=None,
                threshold=0.0,
                message=f"Dataset is empty (reference: {ref_rows} rows, current: {curr_rows} rows).",
            )

        return DriftReport(
            reference_row_count=ref_rows,
            current_row_count=curr_rows,
            total_features_evaluated=len(feature_results),
            drifted_features_count=0,
            drift_share=0.0,
            is_dataset_drifted=False,
            overall_status="ERROR",
            feature_results=feature_results,
            missing_in_current=[],
            new_in_current=[],
            disclaimer="Cannot compute drift because one or both input datasets contain 0 rows.",
        )
