# QualityLens 🔍
> **A Lightweight Data Quality and Drift Monitoring Framework**

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen.svg)]()
[![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-red.svg)]()

QualityLens is a modular, local-first Python framework designed to help data engineers, analysts, and ML practitioners verify data reliability. It operates across three core pillars:

1. **Data Quality Audit (Milestone 1)**: Evaluates single datasets for hygiene flaws (missing values, duplicate rows, schema violations, invalid numeric ranges, unexpected categories) and calculates an objective 0–100 quality score.
2. **Data Drift Detection (Milestone 2)**: Compares incoming batch data against a historical baseline reference to detect statistical distribution divergence across numerical and categorical features.
3. **Interactive Dashboard (Milestone 3)**: A responsive local Streamlit & Plotly web application featuring dynamic visual scorecards, distribution charts, feature drill-downs, and instant JSON/CSV exports.

---

## ⚖️ Quality Checks vs. Drift Detection: Key Distinction

| Dimension | Data Quality Engine (Milestone 1) | Data Drift Engine (Milestone 2) |
| :--- | :--- | :--- |
| **Scope** | Single dataset in isolation. | Two datasets: Reference Baseline vs. Current Batch. |
| **Core Question** | *"Is this dataset cleanly formatted and compliant with rules?"* | *"Has the statistical distribution shifted compared to the baseline?"* |
| **Typical Issues Detected** | Null cells, duplicate rows, negative ages, missing columns. | Increased average student study hours, new remote enrollment programs. |
| **Evaluation Method** | Deterministic rule checking & threshold deductions. | Two-sample Kolmogorov-Smirnov test, PSI, and quantile shifts. |
| **Outcome** | Quality Score (0–100) & Classified Findings (ERROR / WARN). | Drift Status (`STABLE`, `WARNING`, `DRIFT_DETECTED`) & p-values. |
| **Meaning of Alert** | Data contains technical defects or breaches validation rules. | Real-world patterns or population demographics changed. |

> [!NOTE]
> **Drift is Not Necessarily Bad Data**: A detected drift does not mean the data is corrupt or low-quality. It means real-world patterns have changed (e.g., student study habits changing or new programs being launched).

---

## 🚀 Features Implemented

### Milestone 1: Core Data Quality Engine (`DataQualityEngine`)
- **Missing Value Auditing**: Global and per-column counts and percentages; threshold exceedance flags.
- **Duplicate Detection**: Identifies exact duplicate rows across all fields.
- **Schema & Type Verification**: Validates actual vs. expected schemas, flags missing or extra columns, and detects mixed Python types.
- **Value Constraints**: Configurable min/max numeric boundaries and allowed categorical sets without data mutation.
- **Deterministic 0–100 Scoring**: Mathematical penalty model explaining every point deduction.

### Milestone 2: Statistical Data Drift Engine (`DataDriftEngine`)
- **Numerical Drift Detection**:
  - **Two-Sample Kolmogorov-Smirnov (KS) Test**: Compares empirical cumulative distribution functions without assuming normality.
  - **Summary Statistics Comparison**: Compares mean, standard deviation, median, 25th/75th percentiles, min, and max.
  - **Configurable Significance**: Configurable $\alpha$ significance threshold (default $p < 0.05$).
- **Categorical Drift Detection**:
  - **Population Stability Index (PSI)**: Standard industry metric for frequency divergence with $\epsilon = 10^{-4}$ smoothing.
  - **Total Variation Distance (TVD)**: Calculates probability mass divergence bounded between $0.0$ and $1.0$.
  - **Unseen Category Tracking**: Automatically flags new categorical values present in current data that never existed in the baseline.
- **Resilient Edge-Case Handling**:
  - Gracefully handles empty DataFrames (0 rows) and missing columns.
  - Flags incompatible data types between reference and current datasets.
  - Detects small sample sizes ($N < 5$) and warns when statistical power is insufficient.
- **Strict Non-Destructive Guarantee**: Neither the reference nor current DataFrame is ever mutated in-place.

### Milestone 3: Interactive Visual Dashboard (`app.py` & `src/dashboard_utils.py`)
- **Streamlit Web Application**: Local-first responsive UI with navigation between Quality Audit, Drift Detection, and Documentation.
- **Dynamic Visualizations with Plotly**:
  - **Radial Quality Gauge**: Visualizes 0–100 quality score with emerald, amber, and red health bands.
  - **Missing Values Chart**: Horizontal bar chart indicating column missingness against configured thresholds.
  - **Overlaid Distribution Density Plots**: Histograms and box plots comparing Reference vs. Current numerical features.
  - **Categorical Distribution Bar Charts**: Grouped bar charts comparing category frequency percentages.
- **Interactive Feature Inspector**: Select and drill down into any feature to review exact statistics, KS $p$-values, PSI, and unseen categories.
- **Report Exports**: One-click download buttons for structured JSON reports and tabular CSV finding summaries.
- **Built-in Quick Test Toggle**: Checkbox to instantly load synthetic student datasets without manual file selection.

---

## 📸 Dashboard Screenshots & Capture Guidance

Below is the layout preview for the interactive dashboard views:

### 1. Data Quality Audit View
```
+-------------------------------------------------------------------------+
| QualityLens: Data Quality Audit                                          |
|-------------------------------------------------------------------------|
| [Quality Score: 95.3/100] [Rows: 10] [Cols: 7] [Missing: 3] [Dups: 1]   |
|-------------------------------------------------------------------------|
| [ Radial Quality Gauge ]     | [ Missing Values Bar Chart by Column ]    |
|-------------------------------------------------------------------------|
| Tabs: [Errors (0)] | [Warnings (3)] | [Info (3)] | [Data Preview]        |
| [Download JSON Report]  [Download Findings CSV]                          |
+-------------------------------------------------------------------------+
```

### 2. Data Drift Detection View
```
+-------------------------------------------------------------------------+
| QualityLens: Data Drift Detection                                       |
|-------------------------------------------------------------------------|
| [Status: DRIFT DETECTED] [Evaluated: 5] [Drifted: 2/5 (40%)]            |
|-------------------------------------------------------------------------|
| Feature Drilldown: [study_hours] (numerical) -> HIGH SEVERITY           |
| KS-Stat: 1.0000 | p-value: 1.69e-17 | Ref Mean: 14.90 | Curr Mean: 28.83|
| [ Plotly Overlaid Density Histogram (Reference vs. Current) ]           |
|-------------------------------------------------------------------------|
| Full Feature Summary Table (age, gpa, study_hours, major, status)       |
| [Download Drift JSON]  [Download Drift Summary CSV]                     |
+-------------------------------------------------------------------------+
```

#### How to Capture and Save Real Screenshots:
1. Launch the app: `python -m streamlit run app.py`
2. Open `http://localhost:8501` in your browser.
3. Take snapshots of the Quality and Drift tabs using your OS screenshot utility (e.g. `Win + Shift + S`).
4. Save the images to `docs/images/quality_audit.png` and `docs/images/drift_detection.png` and embed them into markdown.

---

## 📐 Statistical Metric Explanations

### 1. Kolmogorov-Smirnov (KS) Two-Sample Test
- **What it measures**: Maximum vertical difference between the empirical cumulative distribution functions of the reference ($F_{\text{ref}}$) and current ($F_{\text{curr}}$) samples:
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
- **Interpretation**: Under the null hypothesis $H_0$, both samples originate from the same distribution. If the computed $p\text{-value} < \alpha$ (default $0.05$), $H_0$ is rejected and drift is detected.

### 2. Population Stability Index (PSI)
- **What it measures**: Categorical frequency distribution shift across all categories:
  $$\text{PSI} = \sum_{i=1}^k (P_{\text{curr}, i} - P_{\text{ref}, i}) \times \ln\left(\frac{P_{\text{curr}, i}}{P_{\text{ref}, i}}\right)$$
- **Standard Thresholds**:
  - $\text{PSI} < 0.10$: No significant change (**Stable**).
  - $0.10 \le \text{PSI} < 0.20$: Moderate distribution shift (**Monitor**).
  - $\text{PSI} \ge 0.20$: Significant distribution shift (**Drift Detected**).

### 3. Total Variation Distance (TVD)
- **What it measures**: Half the $L_1$ norm of categorical frequency differences:
  $$\text{TVD} = \frac{1}{2} \sum_{i=1}^k |P_{\text{curr}, i} - P_{\text{ref}, i}|$$
- **Interpretation**: Represents the percentage of total observations that would have to change categories for the current distribution to match the reference baseline.

---

## ⚠️ Known Limitations
1. **Small Sample Sizes**: When sample sizes are small ($N < 30$), statistical tests lose statistical power. The engine provides an explicit warning when $N < 5$.
2. **High-Cardinality Features**: Categorical columns with thousands of distinct categories (e.g. UUIDs or unique names) can dilute PSI buckets.
3. **Correlation Shifts**: Milestone 2 evaluates univariate drift feature-by-feature; multi-feature covariate interaction shifts will be expanded in future releases.
4. **Local Dashboard**: Streamlit runs on your local machine (`localhost:8501`); it does not require external cloud servers or paid hosting.

---

## 🛠️ Technology Stack

| Layer | Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Core execution environment. |
| **Data Engine** | pandas (>=2.2.0) | Tabular data structures and frequency analysis. |
| **Numerics** | NumPy (>=1.26.0) | Array math, quantiles, and logarithmic smoothing. |
| **Statistics** | SciPy (>=1.12.0) | Two-sample KS test and distribution distance metrics. |
| **Web Dashboard**| Streamlit (>=1.35.0)| Interactive local web user interface. |
| **Charts** | Plotly (>=5.20.0) | Interactive radial gauges, density plots, and bar charts. |
| **Testing** | pytest (>=8.0.0) | 33 automated unit tests. |
| **Cost** | 100% Free | Fully local; no external APIs, cloud services, or paid keys. |

---

## 💻 Setup Instructions for Windows PowerShell

### Step 1: Open PowerShell in the project directory
```powershell
cd c:\Users\shree\OneDrive\Desktop\QualityLens
```

### Step 2: Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 🚀 Launching the Interactive Dashboard

Launch the Streamlit web dashboard locally:

```powershell
python -m streamlit run app.py
```

The application will open automatically in your browser at:
`http://localhost:8501`

---

## 🧪 Running Automated Tests

Run the full 33-test suite with verbose test breakdown:

```powershell
python -m pytest -v
```

All 33 test cases run in under 4 seconds without network access:
- **12 Quality Engine Tests** (`tests/test_quality_engine.py`): Clean data, missing values, duplicates, out-of-bounds numbers, unexpected categories, missing schema columns, empty frames, mixed Python types, score formulas, and DataFrame non-mutation.
- **12 Drift Engine Tests** (`tests/test_drift_engine.py`): Identical baseline comparison, KS numerical drift, PSI categorical shift, unseen categories, missing columns in current batch, incompatible dtypes, empty DataFrames, small samples, non-mutation guarantee, and end-to-end sample verification.
- **9 Dashboard Utility Tests** (`tests/test_dashboard_utils.py`): Safe CSV parser, radial quality score gauge, missingness bar chart, numerical overlaid distribution plots, categorical frequency bar charts, and JSON/CSV export formatting.

---

## 📊 Running the Command-Line Demonstration

### Run both Quality Audit and Drift Detection:
```powershell
python -m src.main
```

### Run only Milestone 1 (Data Quality Audit):
```powershell
python -m src.main --mode quality
```

### Run only Milestone 2 (Data Drift Detection):
```powershell
python -m src.main --mode drift
```

### Run Drift Detection on custom CSV files:
```powershell
python -m src.main --mode drift --reference path\to\baseline.csv --current path\to\batch.csv
```

---

## 🗺️ Project Milestones & Roadmap

- [x] **Milestone 1: Core Data Quality Engine** *(Completed)*
- [x] **Milestone 2: Data Drift Detection** *(Completed)*
- [x] **Milestone 3: Interactive Visual Dashboard** *(Completed)*
- [ ] **Milestone 4: Automated Reporting & Pipeline Integration** *(Planned)*
  - HTML & JSON exportable audit summaries.
  - Automated CI/CD quality gate enforcement.
