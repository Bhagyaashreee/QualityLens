"""QualityLens - Core Data Quality Engine.

This module provides the DataQualityEngine and QualityReport classes for assessing
dataset hygiene, schema conformance, missing values, duplicates, and value validity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence
import numpy as np
import pandas as pd


@dataclass
class QualityFinding:
    """Individual quality finding with severity classification.

    Severities:
        - "ERROR": Critical issues like missing expected columns or empty dataset.
        - "WARNING": Sub-optimal conditions like duplicate rows or out-of-range values.
        - "INFO": Informational observations like schema detection summary.
    """

    severity: str  # "ERROR", "WARNING", "INFO"
    category: str  # "missing_values", "duplicates", "schema", "value_validation", "general"
    column: str | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    """Structured result returned by DataQualityEngine.

    Contains complete metrics, findings by severity, score breakdown,
    and a human-readable text summary.
    """

    row_count: int
    column_count: int
    missing_summary: dict[str, Any]
    duplicate_summary: dict[str, Any]
    schema_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    score_summary: dict[str, Any]
    findings: list[QualityFinding] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Convenience property for the overall quality score (0.0 to 100.0)."""
        return float(self.score_summary.get("overall_score", 0.0))

    def findings_by_severity(self) -> dict[str, list[QualityFinding]]:
        """Group findings by severity level."""
        grouped: dict[str, list[QualityFinding]] = {"ERROR": [], "WARNING": [], "INFO": []}
        for finding in self.findings:
            sev = finding.severity.upper()
            if sev in grouped:
                grouped[sev].append(finding)
            else:
                grouped.setdefault(sev, []).append(finding)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        """Convert the full report into a serializable dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "overall_score": self.overall_score,
            "missing_summary": self.missing_summary,
            "duplicate_summary": self.duplicate_summary,
            "schema_summary": self.schema_summary,
            "validation_summary": self.validation_summary,
            "score_summary": self.score_summary,
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary_text(self) -> str:
        """Generate a concise, professional human-readable report."""
        grouped = self.findings_by_severity()
        err_count = len(grouped.get("ERROR", []))
        warn_count = len(grouped.get("WARNING", []))
        info_count = len(grouped.get("INFO", []))

        lines = [
            "=" * 70,
            " QUALITYLENS DATA QUALITY AUDIT REPORT",
            "=" * 70,
            f"Dataset Dimensions : {self.row_count:,} rows x {self.column_count} columns",
            f"Overall Quality Score : {self.overall_score:.1f} / 100.0",
            f"Findings Breakdown   : {err_count} Errors | {warn_count} Warnings | {info_count} Info",
            "-" * 70,
            "1. MISSING VALUES SUMMARY:",
            f"   - Total Missing Cells: {self.missing_summary.get('total_missing_cells', 0):,} "
            f"({self.missing_summary.get('overall_missing_percentage', 0.0):.2f}% of total cells)",
        ]

        exceeded = self.missing_summary.get("columns_exceeding_threshold", [])
        if exceeded:
            lines.append(f"   - Columns exceeding threshold ({self.missing_summary.get('threshold_percentage', 0)}%): "
                         f"{', '.join(exceeded)}")
        else:
            lines.append("   - No column exceeded the missing-value threshold.")

        lines.extend([
            "-" * 70,
            "2. DUPLICATES SUMMARY:",
            f"   - Duplicate Rows: {self.duplicate_summary.get('duplicate_rows', 0):,} "
            f"({self.duplicate_summary.get('duplicate_percentage', 0.0):.2f}%)",
            "-" * 70,
            "3. SCHEMA & DATA TYPES:",
        ])

        missing_cols = self.schema_summary.get("missing_columns", [])
        unexpected_cols = self.schema_summary.get("unexpected_columns", [])
        mismatched_types = self.schema_summary.get("type_mismatches", {})
        mixed_cols = self.schema_summary.get("mixed_type_columns", [])

        if missing_cols:
            lines.append(f"   - Missing Expected Columns: {', '.join(missing_cols)}")
        if unexpected_cols:
            lines.append(f"   - Unexpected Extra Columns: {', '.join(unexpected_cols)}")
        if mismatched_types:
            for col, err in mismatched_types.items():
                lines.append(f"   - Type Mismatch [{col}]: Expected {err['expected']}, Got {err['detected']}")
        if mixed_cols:
            lines.append(f"   - Columns with Mixed Python Types: {', '.join(mixed_cols)}")
        if not (missing_cols or unexpected_cols or mismatched_types or mixed_cols):
            lines.append("   - Schema matches expected structure with uniform column types.")

        lines.extend([
            "-" * 70,
            "4. VALUE VALIDATION:",
        ])

        num_val = self.validation_summary.get("numeric_validations", {})
        cat_val = self.validation_summary.get("categorical_validations", {})

        if not num_val and not cat_val:
            lines.append("   - No custom numeric or categorical validation rules configured.")
        else:
            for col, res in num_val.items():
                invalid_cnt = res.get("invalid_count", 0)
                lines.append(
                    f"   - Numeric [{col}]: {invalid_cnt} invalid out-of-bounds records "
                    f"({res.get('invalid_percentage', 0.0):.2f}%) [Range: {res.get('min_allowed')} to {res.get('max_allowed')}]"
                )
            for col, res in cat_val.items():
                invalid_cnt = res.get("invalid_count", 0)
                lines.append(
                    f"   - Categorical [{col}]: {invalid_cnt} disallowed records "
                    f"({res.get('invalid_percentage', 0.0):.2f}%)"
                )

        lines.extend([
            "-" * 70,
            "5. SCORE CALCULATION & EXPLANATION:",
            f"   - Base Score : {self.score_summary.get('base_score', 100.0):.1f}",
        ])
        for comp, deduction in self.score_summary.get("deductions", {}).items():
            if deduction > 0:
                lines.append(f"   - Deduction ({comp}) : -{deduction:.2f} pts")
        lines.append(f"   - Final Calculated Score : {self.overall_score:.1f} / 100.0")
        lines.append(f"   - Note: {self.score_summary.get('disclaimer', '')}")

        if err_count > 0 or warn_count > 0:
            lines.extend([
                "-" * 70,
                "6. DETAILED FINDINGS:",
            ])
            for sev in ["ERROR", "WARNING"]:
                for f in grouped.get(sev, []):
                    col_str = f"[{f.column}] " if f.column else ""
                    lines.append(f"   [{f.severity}] {col_str}{f.message}")

        lines.append("=" * 70)
        return "\n".join(lines)


class DataQualityEngine:
    """Core evaluation engine for tabular data quality.

    Evaluates:
    - Missing values (per column and overall)
    - Duplicate rows
    - Schema conformity and mixed types
    - Numeric ranges and categorical domain constraints
    - Overall deterministic quality score

    Never modifies the input DataFrame in-place.
    """

    def __init__(
        self,
        missing_threshold: float = 0.20,
        expected_schema: dict[str, str] | None = None,
        numeric_ranges: dict[str, dict[str, float | int | None]] | None = None,
        categorical_allowed_values: dict[str, Sequence[Any]] | None = None,
    ) -> None:
        """Initialize the quality engine with configurable rules.

        Args:
            missing_threshold: Maximum acceptable missing value ratio per column
                (e.g., 0.20 for 20%). Columns exceeding this trigger warnings.
            expected_schema: Optional dict mapping column names to expected types
                (e.g., {"id": "int", "name": "string", "age": "float"}).
            numeric_ranges: Optional dict mapping numeric column names to min/max
                bounds (e.g., {"age": {"min": 0, "max": 120}}).
            categorical_allowed_values: Optional dict mapping categorical column
                names to allowed value collections.
        """
        if not (0.0 <= missing_threshold <= 1.0):
            raise ValueError(f"missing_threshold must be between 0.0 and 1.0, got {missing_threshold}")

        self.missing_threshold = missing_threshold
        self.expected_schema = expected_schema or {}
        self.numeric_ranges = numeric_ranges or {}
        self.categorical_allowed_values = {
            col: set(vals) for col, vals in (categorical_allowed_values or {}).items()
        }

    def evaluate(self, df: pd.DataFrame) -> QualityReport:
        """Perform full quality evaluation on the input DataFrame.

        Guarantees that the input DataFrame will not be modified.

        Args:
            df: pandas DataFrame to evaluate.

        Returns:
            QualityReport containing metrics, findings, and score.
        """
        findings: list[QualityFinding] = []

        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, received {type(df).__name__}")

        total_rows = len(df)
        total_cols = len(df.columns)

        # Handle empty DataFrame edge cases
        if total_rows == 0 or total_cols == 0:
            return self._handle_empty_dataframe(df, total_rows, total_cols)

        # Run individual check modules
        missing_summary, missing_findings = self._check_missing_values(df, total_rows)
        duplicate_summary, duplicate_findings = self._check_duplicates(df, total_rows)
        schema_summary, schema_findings = self._check_schema_and_types(df)
        validation_summary, val_findings = self._check_value_validations(df)

        findings.extend(missing_findings)
        findings.extend(duplicate_findings)
        findings.extend(schema_findings)
        findings.extend(val_findings)

        # Calculate deterministic score
        score_summary = self._calculate_quality_score(
            total_rows=total_rows,
            total_cols=total_cols,
            missing_summary=missing_summary,
            duplicate_summary=duplicate_summary,
            schema_summary=schema_summary,
            validation_summary=validation_summary,
        )

        return QualityReport(
            row_count=total_rows,
            column_count=total_cols,
            missing_summary=missing_summary,
            duplicate_summary=duplicate_summary,
            schema_summary=schema_summary,
            validation_summary=validation_summary,
            score_summary=score_summary,
            findings=findings,
        )

    # -------------------------------------------------------------------------
    # Private Check Methods
    # -------------------------------------------------------------------------

    def _handle_empty_dataframe(
        self, df: pd.DataFrame, total_rows: int, total_cols: int
    ) -> QualityReport:
        """Gracefully handle empty DataFrames (0 rows or 0 columns)."""
        findings = [
            QualityFinding(
                severity="ERROR",
                category="general",
                column=None,
                message=f"Dataset is empty ({total_rows} rows, {total_cols} columns). No records to evaluate.",
                details={"row_count": total_rows, "column_count": total_cols},
            )
        ]
        score_summary = {
            "overall_score": 0.0,
            "base_score": 100.0,
            "deductions": {"empty_dataset": 100.0},
            "explanation": "Empty datasets receive an automatic score of 0.0 as no data is present.",
            "disclaimer": self._get_score_disclaimer(),
        }
        return QualityReport(
            row_count=total_rows,
            column_count=total_cols,
            missing_summary={
                "total_cells": 0,
                "total_missing_cells": 0,
                "overall_missing_percentage": 0.0,
                "threshold_percentage": self.missing_threshold * 100.0,
                "columns_exceeding_threshold": [],
                "by_column": {},
            },
            duplicate_summary={
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
            },
            schema_summary={
                "detected_types": {str(c): str(df[c].dtype) for c in df.columns},
                "expected_schema": self.expected_schema,
                "missing_columns": list(self.expected_schema.keys()),
                "unexpected_columns": [str(c) for c in df.columns if c not in self.expected_schema],
                "type_mismatches": {},
                "mixed_type_columns": [],
            },
            validation_summary={
                "numeric_validations": {},
                "categorical_validations": {},
            },
            score_summary=score_summary,
            findings=findings,
        )

    def _check_missing_values(
        self, df: pd.DataFrame, total_rows: int
    ) -> tuple[dict[str, Any], list[QualityFinding]]:
        """Calculate missing value counts and identify threshold exceedances."""
        findings: list[QualityFinding] = []
        total_cells = total_rows * len(df.columns)
        total_missing = int(df.isna().sum().sum())
        overall_missing_pct = round((total_missing / total_cells) * 100.0, 2) if total_cells > 0 else 0.0

        by_column: dict[str, dict[str, Any]] = {}
        exceeded_threshold_cols: list[str] = []

        threshold_pct = round(self.missing_threshold * 100.0, 2)

        for col in df.columns:
            col_str = str(col)
            miss_count = int(df[col].isna().sum())
            miss_pct = round((miss_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
            is_exceeded = miss_pct > threshold_pct

            by_column[col_str] = {
                "missing_count": miss_count,
                "missing_percentage": miss_pct,
                "exceeds_threshold": is_exceeded,
            }

            if is_exceeded:
                exceeded_threshold_cols.append(col_str)
                findings.append(
                    QualityFinding(
                        severity="WARNING" if miss_pct < 50.0 else "ERROR",
                        category="missing_values",
                        column=col_str,
                        message=(
                            f"Column '{col_str}' has {miss_count} missing values ({miss_pct}%), "
                            f"exceeding the threshold of {threshold_pct}%."
                        ),
                        details={"missing_count": miss_count, "missing_percentage": miss_pct},
                    )
                )
            elif miss_count > 0:
                findings.append(
                    QualityFinding(
                        severity="INFO",
                        category="missing_values",
                        column=col_str,
                        message=f"Column '{col_str}' has {miss_count} missing values ({miss_pct}%).",
                        details={"missing_count": miss_count, "missing_percentage": miss_pct},
                    )
                )

        summary = {
            "total_cells": total_cells,
            "total_missing_cells": total_missing,
            "overall_missing_percentage": overall_missing_pct,
            "threshold_percentage": threshold_pct,
            "columns_exceeding_threshold": exceeded_threshold_cols,
            "by_column": by_column,
        }
        return summary, findings

    def _check_duplicates(
        self, df: pd.DataFrame, total_rows: int
    ) -> tuple[dict[str, Any], list[QualityFinding]]:
        """Calculate exact duplicate rows across all columns."""
        findings: list[QualityFinding] = []
        dup_count = int(df.duplicated().sum())
        dup_pct = round((dup_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0

        if dup_count > 0:
            findings.append(
                QualityFinding(
                    severity="WARNING",
                    category="duplicates",
                    column=None,
                    message=f"Found {dup_count} duplicate row(s) ({dup_pct}% of total records).",
                    details={"duplicate_count": dup_count, "duplicate_percentage": dup_pct},
                )
            )
        else:
            findings.append(
                QualityFinding(
                    severity="INFO",
                    category="duplicates",
                    column=None,
                    message="No duplicate rows detected in dataset.",
                    details={"duplicate_count": 0, "duplicate_percentage": 0.0},
                )
            )

        summary = {
            "duplicate_rows": dup_count,
            "duplicate_percentage": dup_pct,
        }
        return summary, findings

    def _check_schema_and_types(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, Any], list[QualityFinding]]:
        """Verify schema against expectations and detect mixed python types."""
        findings: list[QualityFinding] = []
        detected_types: dict[str, str] = {str(col): str(df[col].dtype) for col in df.columns}
        existing_cols = set(detected_types.keys())

        missing_cols: list[str] = []
        unexpected_cols: list[str] = []
        type_mismatches: dict[str, dict[str, str]] = {}
        mixed_type_columns: list[str] = []

        # If expected schema is configured, compare
        if self.expected_schema:
            expected_cols = set(self.expected_schema.keys())
            missing_cols = sorted(list(expected_cols - existing_cols))
            unexpected_cols = sorted(list(existing_cols - expected_cols))

            for col in missing_cols:
                findings.append(
                    QualityFinding(
                        severity="ERROR",
                        category="schema",
                        column=col,
                        message=f"Expected column '{col}' is missing from the dataset.",
                        details={"expected_type": self.expected_schema[col]},
                    )
                )

            for col in unexpected_cols:
                findings.append(
                    QualityFinding(
                        severity="WARNING",
                        category="schema",
                        column=col,
                        message=f"Dataset contains unexpected column '{col}'.",
                        details={"detected_type": detected_types.get(col, "unknown")},
                    )
                )

            # Check matching columns for dtype compatibility
            for col in expected_cols.intersection(existing_cols):
                exp_type = self.expected_schema[col].strip().lower()
                detected_dtype = detected_types[col].lower()

                if not self._is_dtype_compatible(exp_type, detected_dtype):
                    type_mismatches[col] = {
                        "expected": self.expected_schema[col],
                        "detected": detected_types[col],
                    }
                    findings.append(
                        QualityFinding(
                            severity="ERROR",
                            category="schema",
                            column=col,
                            message=(
                                f"Column '{col}' data type mismatch: expected '{self.expected_schema[col]}', "
                                f"detected '{detected_types[col]}'."
                            ),
                            details=type_mismatches[col],
                        )
                    )

        # Detect mixed python types within object columns
        for col in df.columns:
            col_str = str(col)
            if df[col].dtype == "object":
                non_null_series = df[col].dropna()
                if len(non_null_series) > 0:
                    types_present = {type(val).__name__ for val in non_null_series}
                    if len(types_present) > 1:
                        mixed_type_columns.append(col_str)
                        findings.append(
                            QualityFinding(
                                severity="WARNING",
                                category="schema",
                                column=col_str,
                                message=(
                                    f"Column '{col_str}' contains mixed python data types: "
                                    f"{', '.join(sorted(types_present))}."
                                ),
                                details={"types_present": sorted(list(types_present))},
                            )
                        )

        summary = {
            "detected_types": detected_types,
            "expected_schema": self.expected_schema,
            "missing_columns": missing_cols,
            "unexpected_columns": unexpected_cols,
            "type_mismatches": type_mismatches,
            "mixed_type_columns": mixed_type_columns,
        }
        return summary, findings

    def _check_value_validations(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, Any], list[QualityFinding]]:
        """Validate numeric ranges and categorical domain values."""
        findings: list[QualityFinding] = []
        numeric_validations: dict[str, dict[str, Any]] = {}
        categorical_validations: dict[str, dict[str, Any]] = {}

        # 1. Numeric Range Validations
        for col, bounds in self.numeric_ranges.items():
            col_str = str(col)
            if col_str not in df.columns:
                continue

            series = df[col_str]
            non_null = series.dropna()
            total_evaluated = len(non_null)

            if total_evaluated == 0:
                numeric_validations[col_str] = {
                    "total_evaluated": 0,
                    "invalid_count": 0,
                    "invalid_percentage": 0.0,
                    "min_allowed": bounds.get("min"),
                    "max_allowed": bounds.get("max"),
                    "sample_violations": [],
                }
                continue

            # Convert to numeric safely without mutating original DataFrame
            numeric_vals = pd.to_numeric(non_null, errors="coerce")
            coercion_failed = numeric_vals.isna()

            min_val = bounds.get("min")
            max_val = bounds.get("max")

            out_of_bounds = pd.Series(False, index=non_null.index)
            if min_val is not None:
                out_of_bounds = out_of_bounds | (numeric_vals < min_val)
            if max_val is not None:
                out_of_bounds = out_of_bounds | (numeric_vals > max_val)

            invalid_mask = coercion_failed | out_of_bounds
            invalid_count = int(invalid_mask.sum())
            invalid_pct = round((invalid_count / total_evaluated) * 100.0, 2)

            sample_violations = non_null[invalid_mask].head(5).tolist()

            numeric_validations[col_str] = {
                "total_evaluated": total_evaluated,
                "invalid_count": invalid_count,
                "invalid_percentage": invalid_pct,
                "min_allowed": min_val,
                "max_allowed": max_val,
                "sample_violations": sample_violations,
            }

            if invalid_count > 0:
                findings.append(
                    QualityFinding(
                        severity="WARNING",
                        category="value_validation",
                        column=col_str,
                        message=(
                            f"Column '{col_str}' has {invalid_count} numeric value(s) out of valid range "
                            f"[{min_val}, {max_val}] ({invalid_pct}% of valid records)."
                        ),
                        details={
                            "invalid_count": invalid_count,
                            "invalid_percentage": invalid_pct,
                            "samples": sample_violations,
                        },
                    )
                )

        # 2. Categorical Allowed Values Validations
        for col, allowed_set in self.categorical_allowed_values.items():
            col_str = str(col)
            if col_str not in df.columns:
                continue

            series = df[col_str]
            non_null = series.dropna()
            total_evaluated = len(non_null)

            if total_evaluated == 0:
                categorical_validations[col_str] = {
                    "total_evaluated": 0,
                    "invalid_count": 0,
                    "invalid_percentage": 0.0,
                    "allowed_values": sorted([str(v) for v in allowed_set]),
                    "unexpected_values": [],
                }
                continue

            # Compare string-normalized or exact matches
            # We check both exact membership and string membership
            is_valid = non_null.isin(allowed_set)
            invalid_mask = ~is_valid
            invalid_count = int(invalid_mask.sum())
            invalid_pct = round((invalid_count / total_evaluated) * 100.0, 2)
            unexpected_values = list(non_null[invalid_mask].unique())[:5]

            categorical_validations[col_str] = {
                "total_evaluated": total_evaluated,
                "invalid_count": invalid_count,
                "invalid_percentage": invalid_pct,
                "allowed_values": sorted([str(v) for v in allowed_set]),
                "unexpected_values": [str(v) for v in unexpected_values],
            }

            if invalid_count > 0:
                findings.append(
                    QualityFinding(
                        severity="WARNING",
                        category="value_validation",
                        column=col_str,
                        message=(
                            f"Column '{col_str}' has {invalid_count} disallowed categorical record(s) "
                            f"({invalid_pct}%). Found unexpected: {unexpected_values}."
                        ),
                        details={
                            "invalid_count": invalid_count,
                            "invalid_percentage": invalid_pct,
                            "unexpected_values": [str(v) for v in unexpected_values],
                        },
                    )
                )

        summary = {
            "numeric_validations": numeric_validations,
            "categorical_validations": categorical_validations,
        }
        return summary, findings

    # -------------------------------------------------------------------------
    # Scoring Algorithm
    # -------------------------------------------------------------------------

    def _calculate_quality_score(
        self,
        total_rows: int,
        total_cols: int,
        missing_summary: dict[str, Any],
        duplicate_summary: dict[str, Any],
        schema_summary: dict[str, Any],
        validation_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate a transparent 0-100 quality score with deterministic deductions.

        Rule Weights:
        - Completeness (Max 30 pts):
            - Overall missing percentage * 0.20 (up to 20 pts)
            - Threshold exceedance penalty: 5 pts per offending column (up to 10 pts)
        - Uniqueness (Max 20 pts):
            - Duplicate row percentage * 0.20 (up to 20 pts)
        - Schema Conformance (Max 25 pts):
            - Missing required columns: 10 pts each (up to 15 pts)
            - Data type mismatches: 5 pts each (up to 10 pts)
        - Value Validity (Max 25 pts):
            - Numeric range violations percentage * 0.15 (up to 15 pts)
            - Categorical disallowed values percentage * 0.10 (up to 10 pts)
        """
        base_score = 100.0
        deductions: dict[str, float] = {}

        # 1. Completeness Deductions
        overall_missing_pct = missing_summary.get("overall_missing_percentage", 0.0)
        missing_cells_penalty = min(20.0, (overall_missing_pct / 100.0) * 20.0)
        exceeded_cols_count = len(missing_summary.get("columns_exceeding_threshold", []))
        threshold_penalty = min(10.0, exceeded_cols_count * 5.0)
        completeness_deduction = round(missing_cells_penalty + threshold_penalty, 2)
        deductions["completeness"] = completeness_deduction

        # 2. Uniqueness Deductions
        duplicate_pct = duplicate_summary.get("duplicate_percentage", 0.0)
        uniqueness_deduction = round(min(20.0, (duplicate_pct / 100.0) * 20.0), 2)
        deductions["uniqueness"] = uniqueness_deduction

        # 3. Schema Conformance Deductions
        missing_cols_count = len(schema_summary.get("missing_columns", []))
        mismatched_types_count = len(schema_summary.get("type_mismatches", {}))
        schema_missing_penalty = min(15.0, missing_cols_count * 10.0)
        schema_type_penalty = min(10.0, mismatched_types_count * 5.0)
        schema_deduction = round(schema_missing_penalty + schema_type_penalty, 2)
        deductions["schema_conformance"] = schema_deduction

        # 4. Value Validity Deductions
        num_invalid_pct_sum = 0.0
        num_rules_count = 0
        for res in validation_summary.get("numeric_validations", {}).values():
            num_invalid_pct_sum += res.get("invalid_percentage", 0.0)
            num_rules_count += 1
        avg_num_invalid_pct = (num_invalid_pct_sum / num_rules_count) if num_rules_count > 0 else 0.0
        num_penalty = min(15.0, (avg_num_invalid_pct / 100.0) * 15.0)

        cat_invalid_pct_sum = 0.0
        cat_rules_count = 0
        for res in validation_summary.get("categorical_validations", {}).values():
            cat_invalid_pct_sum += res.get("invalid_percentage", 0.0)
            cat_rules_count += 1
        avg_cat_invalid_pct = (cat_invalid_pct_sum / cat_rules_count) if cat_rules_count > 0 else 0.0
        cat_penalty = min(10.0, (avg_cat_invalid_pct / 100.0) * 10.0)

        validity_deduction = round(num_penalty + cat_penalty, 2)
        deductions["value_validity"] = validity_deduction

        # Calculate final score
        total_deduction = round(sum(deductions.values()), 2)
        final_score = max(0.0, round(base_score - total_deduction, 2))

        explanation = (
            f"Base score is 100.0. Deterministic deductions were applied: "
            f"Completeness (-{completeness_deduction:.2f}), "
            f"Uniqueness (-{uniqueness_deduction:.2f}), "
            f"Schema Conformance (-{schema_deduction:.2f}), "
            f"Value Validity (-{validity_deduction:.2f})."
        )

        return {
            "overall_score": final_score,
            "base_score": base_score,
            "total_deduction": total_deduction,
            "deductions": deductions,
            "explanation": explanation,
            "disclaimer": self._get_score_disclaimer(),
        }

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_dtype_compatible(expected_type: str, detected_dtype: str) -> bool:
        """Evaluate flexible, user-friendly data type compatibility."""
        exp = expected_type.lower()
        det = detected_dtype.lower()

        if exp in det:
            return True

        if exp in {"int", "integer", "int64", "int32"}:
            return "int" in det
        if exp in {"float", "float64", "float32", "double"}:
            return "float" in det
        if exp in {"numeric", "number"}:
            return "int" in det or "float" in det
        if exp in {"str", "string", "text", "object"}:
            return "object" in det or "string" in det or "str" in det
        if exp in {"bool", "boolean"}:
            return "bool" in det
        if exp in {"date", "datetime", "timestamp"}:
            return "date" in det or "time" in det

        return False

    @staticmethod
    def _get_score_disclaimer() -> str:
        """Standard disclaimer regarding the meaning of the quality score."""
        return (
            "The QualityLens quality score reflects technical adherence to specified "
            "rules, schema contracts, and hygiene heuristics. It does not certify that "
            "the data is objectively correct, complete with respect to reality, or free "
            "from real-world sampling bias."
        )
