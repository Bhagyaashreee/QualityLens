# QualityLens: Architectural Overview & Technical Design

## 1. Project Mission

**QualityLens** is a lightweight, local-first data quality and drift monitoring framework built with Python. In production data pipelines, poor data quality and silent distribution drift propagate unnoticed, causing downstream machine learning regressions, analytical discrepancies, and broken reporting.

QualityLens provides an automated, deterministic audit mechanism to inspect tabular data before ingestion into analytics or ML pipelines.

---

## 2. System Architecture

```
                       ┌─────────────────────────┐
                       │   Baseline Reference    │
                       │     (Training/Past)     │
                       └───────────┬─────────────┘
                                   │
      ┌─────────────────────────┐  │
      │   Incoming Batch Data   │──┤
      │     (Current/New)       │  │
      └───────────┬─────────────┘  │
                  │                │
                  ▼                ▼
     ┌─────────────────────────┐ ┌─────────────────────────┐
     │   DataQualityEngine     │ │    DataDriftEngine      │
     │      (Milestone 1)      │ │      (Milestone 2)      │
     ├─────────────────────────┤ ├─────────────────────────┤
     │ • Missing Value Checks  │ │ • 2-Sample KS Test (Num)│
     │ • Duplicate Row Counts  │ │ • Quantile Shift (Mean) │
     │ • Schema & Type Mismatch│ │ • PSI & TVD (Cat)       │
     │ • Range & Domain Bounds │ │ • Unseen Category Alert │
     └────────────┬────────────┘ └────────────┬────────────┘
                  │                           │
                  ▼                           ▼
     ┌─────────────────────────┐ ┌─────────────────────────┐
     │      QualityReport      │ │       DriftReport       │
     │ • Score: 0.0 - 100.0    │ │ • Feature Drift Status  │
     │ • Classified Findings   │ │ • Overall Dataset Drift │
     └────────────┬────────────┘ └────────────┬────────────┘
                  │                           │
                  └─────────────┬─────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        CLI Summary Output            JSON Export / API Ready
```

---

## 3. Core Engine Principles

### A. Non-Destructive Evaluation
QualityLens **never mutates the input data**. It acts strictly as an auditor:
- No rows are automatically dropped.
- No nulls are silently imputed or filled.
- No invalid values are overwritten.

This design ensures idempotent, reproducible audits that do not introduce side-effects into your pipeline.

### B. Milestone 1: Deterministic Quality Scoring (0 – 100)
Rather than relying on opaque heuristics or AI approximations, QualityLens calculates a transparent score starting from **100.0 points** using documented penalties across four key dimensions:

1. **Completeness (Max 30 pts deduction)**
   - Ratio of total missing cells across the dataset (up to 20 pts).
   - Penalty for columns exceeding the user-configured missingness threshold (5 pts per offending column, up to 10 pts).
2. **Uniqueness (Max 20 pts deduction)**
   - Ratio of duplicate rows relative to dataset size (up to 20 pts).
3. **Schema Conformance (Max 25 pts deduction)**
   - Missing required columns from `expected_schema` (10 pts per column, up to 15 pts).
   - Column data type mismatches (5 pts per column, up to 10 pts).
4. **Value Validity (Max 25 pts deduction)**
   - Out-of-bounds numeric values across validated columns (up to 15 pts).
   - Disallowed categorical values across validated columns (up to 10 pts).

### C. Milestone 2: Statistical Data Drift Detection
Data drift measures whether the underlying statistical distribution of incoming data has diverged from a reference baseline.

> [!IMPORTANT]
> **Drift is Not Necessarily Corruption**: A feature exhibiting drift does not mean the data is invalid. It indicates that real-world dynamics, seasonal patterns, or user populations have changed. Drift signals that downstream machine learning models or statistical assumptions should be retrained or re-evaluated.

#### 1. Numerical Drift: Two-Sample Kolmogorov-Smirnov Test
- **Method**: The two-sample Kolmogorov-Smirnov test compares the empirical cumulative distribution functions (eCDFs) of the reference and current numeric samples.
- **Statistic ($D$)**: The maximum vertical distance between the two cumulative distributions:
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
- **Hypothesis**: The null hypothesis ($H_0$) states that both samples are drawn from the same continuous distribution. If the computed $p\text{-value} < \alpha$ (default $\alpha = 0.05$), $H_0$ is rejected and drift is flagged.
- **Descriptive Statistics**: In addition to the KS test, QualityLens calculates and compares the mean, standard deviation, median, 25th percentile, 75th percentile, min, and max for both datasets.

#### 2. Categorical Drift: Population Stability Index (PSI) & Total Variation Distance (TVD)
- **Population Stability Index (PSI)**: A standard industry metric for measuring categorical frequency shifts:
  $$\text{PSI} = \sum_{i=1}^k (P_{\text{curr}, i} - P_{\text{ref}, i}) \times \ln\left(\frac{P_{\text{curr}, i}}{P_{\text{ref}, i}}\right)$$
  - $\text{PSI} < 0.10$: Stable; no significant distribution change.
  - $0.10 \le \text{PSI} < 0.20$: Moderate shift; monitor.
  - $\text{PSI} \ge 0.20$: Significant drift detected.
- **Epsilon Smoothing**: QualityLens applies $\epsilon = 10^{-4}$ smoothing to avoid $\ln(0)$ or division by zero when new categories appear or existing categories disappear.
- **Total Variation Distance (TVD)**:
  $$\text{TVD} = \frac{1}{2} \sum_{i=1}^k |P_{\text{curr}, i} - P_{\text{ref}, i}|$$
  Measures the total probability mass that must shift to reconcile the two distributions (bounded in $[0, 1]$).
- **Unseen & Missing Categories**: Explicitly highlights new categories present in the current batch that were never observed in the baseline.

---

## 4. Key Components

| Component | File | Responsibility |
| :--- | :--- | :--- |
| `DataQualityEngine` | `src/quality_engine.py` | Configurable engine for nulls, duplicates, schema, and range checks. |
| `QualityReport` | `src/quality_engine.py` | Dataclass providing quality metrics, severity grouping, and text reports. |
| `DataDriftEngine` | `src/drift_engine.py` | Statistical engine for numerical KS tests, quantile shifts, and categorical PSI/TVD. |
| `DriftReport` | `src/drift_engine.py` | Dataclass providing feature-level drift metrics and overall dataset drift status. |
| `Dashboard Utilities` | `src/dashboard_utils.py` | Safe CSV loading, Plotly charts (gauges, histograms, box plots, bar charts), export formats. |
| `Streamlit App` | `app.py` | Interactive local web dashboard for Quality Audit and Drift Detection. |
| CLI Runner | `src/main.py` | CLI entry point supporting `--mode {quality, drift, all}`. |
| Sample Datasets | `data/sample/` | Synthetic CSV files for quality anomalies and baseline/drift cohorts. |
| Unit Tests | `tests/` | 33 automated unit tests covering quality, drift, and dashboard utilities. |

---

## 5. Technology Stack Decisions

- **Python 3.12**: Modern Python runtime with enhanced performance and strong type annotations.
- **pandas (>=2.2.0)**: Tabular data representation and operations.
- **NumPy (>=1.26.0)**: Numerical vector operations and quantile calculations.
- **SciPy (>=1.12.0)**: Statistical hypothesis tests (`scipy.stats.ks_2samp`, `scipy.stats.wasserstein_distance`).
- **Streamlit (>=1.35.0)**: Local-first, free interactive web dashboard framework.
- **Plotly (>=5.20.0)**: Interactive chart visualizations (gauges, histograms, box plots, bar charts).
- **pytest (>=8.0.0)**: Fast, deterministic unit test runner.
- **Zero Paid Dependencies**: 100% free, local execution with no external API calls, cloud dependencies, or tracking.

---

## 6. Milestone Roadmap

- [x] **Milestone 1**: Core Data Quality Engine, deterministic scoring, CLI runner, unit tests.
- [x] **Milestone 2**: Statistical Data Drift Detection (KS test, quantile comparison, PSI, TVD, unseen categories).
- [x] **Milestone 3**: Streamlit Interactive Dashboard & Plotly Visualizations (scorecard, distribution plots, exports).
- [ ] **Milestone 4 (Planned)**: Automated Reporting & Pipeline Integration (CI/CD quality gates).
