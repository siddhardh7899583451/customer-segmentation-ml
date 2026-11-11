# Customer Segmentation & Behavioral Analytics Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-unit%20suite-blue.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deployment: Static Space](https://img.shields.io/badge/Hugging%20Face-Static%20SDK-orange.svg)](deployment/huggingface-space/)

An enterprise-grade unsupervised machine learning platform and client-side behavioral intelligence dashboard. The platform ingests transactional retail purchase logs, performs rigorous data hygiene and customer-level RFM (Recency, Frequency, Monetary) feature extraction, compares clustering algorithms (K-Means++, DBSCAN, Agglomerative Hierarchical), validates mathematical cluster properties, and profiles empirical customer segments into targeted marketing action plans.

---

## Key Highlights

- **Dual-Engine Architecture:**
  - **Enterprise Python ML Pipeline:** Production CLI and modular Python 3.12 package executing log-scaled standardisation, multi-model evaluation, PCA dimensionality reduction, and automated report generation.
  - **Standalone Web Dashboard:** Zero-backend, pure HTML5/CSS3/Vanilla JavaScript application. Ingests user CSVs, parses transactions, computes RFM metrics client-side, renders responsive canvas charts, and exports results offline with zero server requirements.
- **Strict Mathematical Validation:** Enforces mathematical pre-conditions before calculating Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics ($2 \le k < N$), properly isolating noise in density algorithms like DBSCAN without reporting fabricated or misleading metrics.
- **Empirical Persona Derivation:** Segments are labeled dynamically based on statistical rankings across monetary spend, purchase frequency, and recency, directly mapping managerial strategies to real cluster behavior.
- **Enterprise Security:** Mitigates CSV Formula Injection (DDE) by sanitizing text exports, enforces XSS defense through strict text-node DOM creation, and processes data strictly within local workspace memory.
- **Hugging Face Static Package:** Dedicated static deployment bundle in `deployment/huggingface-space/` (prepared, **deployed** ).

---

## Directory Structure

```
customer-segmentation-ml/
├── .gitignore
├── LICENSE                          # MIT License
├── requirements.txt                 # Pinned Python 3.12 dependencies
├── README.md                        # Master documentation
├── PROJECT_PLAN.md                  # Comprehensive architectural plan
├── PROJECT_STATUS.md                # Real-time milestone tracker
├── TASK_CHECKLIST.md                # Phase-by-phase completion checklist
├── DECISIONS.md                     # Engineering decision records (ADRs)
├── ERROR_LOG.md                     # Incident tracker and bug resolutions
├── TEST_REPORT.md                   # Automated pytest execution report
├── DEPLOYMENT_STATUS.md             # Hugging Face deployment specification
├── NEXT_STEPS.md                    # Production roadmap
├── main.py                          # CLI runner for ML pipeline
├── src/
│   ├── __init__.py
│   ├── config.py                    # Hyperparameters, schemas, and path configuration
│   ├── data_loader.py               # UCI Online Retail ingestion with synthetic demo fallback
│   ├── synthetic_data.py            # Reproducible synthetic transaction & RFM generator
│   ├── preprocessor.py              # Deduplication, cancellation segregation, validation
│   ├── feature_engineering.py       # Customer RFM derivation, log transforms, scaling
│   ├── models.py                    # K-Means++, DBSCAN, Agglomerative, PCA implementations
│   ├── evaluation.py                # Mathematical metrics validation guardrails
│   ├── personas.py                  # Statistical profiling & marketing recommendation engine
│   ├── visualization.py             # Matplotlib/Seaborn publication-grade plot generation
│   ├── pipeline.py                  # End-to-end orchestration pipeline
│   └── utils.py                     # Structured logging, CSV formula sanitization, JSON export
├── data/
│   └── sample_customers_demo.csv    # Clearly labeled synthetic demo dataset
├── output/
│   ├── segmented_customers.csv      # Customer-level segmentation output
│   ├── reports/
│   │   ├── data_quality_report.json # Preprocessing audit and row counts
│   │   ├── elbow_analysis.json      # Inertia & Silhouette values across k=2..8
│   │   ├── model_comparison.json    # K-Means vs DBSCAN vs Agglomerative metrics
│   │   └── customer_personas.json   # Profiles, traits, and marketing recommendations
│   └── figures/
│       ├── elbow_silhouette_curve.png
│       ├── pca_cluster_projection.png
│       ├── rfm_distributions.png
│       └── persona_revenue_comparison.png
├── dashboard/
│   ├── index.html                   # Zero-backend standalone interactive dashboard
│   ├── styles.css                   # Responsive SaaS styling
│   ├── app.js                       # Vanilla JS RFM, Canvas charts, and table pagination
│   └── sample_demo_data.js          # Offline demo customer dataset
├── deployment/
│   └── huggingface-space/           # Standalone deployment bundle for HF Static Spaces
│       ├── README.md                # HF static space configuration header
│       ├── index.html
│       ├── styles.css
│       ├── app.js
│       └── sample_demo_data.js
├── notebooks/
│   └── customer_segmentation_walkthrough.ipynb  # Pedagogical analysis notebook
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_preprocessor.py
    ├── test_feature_engineering.py
    ├── test_models.py
    ├── test_evaluation.py
    ├── test_personas.py
    ├── test_pipeline.py
    ├── test_edge_cases.py
    └── test_js_parity.py            # Python vs JavaScript calculation parity
```

---

## Dataset Provenance

The platform supports two distinct data modalities:

1. **Real-World Target Dataset:**
   - **Source:** [UCI Machine Learning Repository: Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
   - **Scope:** 541,909 transnational transactions between 01/12/2010 and 09/12/2011 for a UK-based registered non-store online retailer.
   - **Schema:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

2. **Synthetic Demonstration Dataset:**
   - When running offline or testing without raw dataset access, the platform generates a synthetic transaction dataset (`data/sample_customers_demo.csv`) reflecting 5 distinct customer personas plus realistic transactional anomalies (cancellations, guest checkouts, duplicate lines).
   - **Labeling Notice:** All synthetic data is explicitly tagged with `DataClassification = "SYNTHETIC_DEMO"` and never presented as genuine customer findings.

---

## Installation & Quick Start

### 1. Environment Setup (Python 3.12)

```bash
# Clone repository and enter directory
cd C:\Projects\customer-segmentation-ml

# Create virtual environment with Python 3.12
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate virtual environment (macOS / Linux)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Execute the Machine Learning Pipeline

```bash
# Run with automatic dataset acquisition / synthetic demo fallback
python main.py

# Force synthetic demo generation
python main.py --demo --k 4

# Run on a custom Excel or CSV dataset
python main.py --data "data/Online Retail.xlsx" --k 4 --output output
```

### 3. Run Automated Pytest Suite

```bash
pytest -v
```

Expected output: most unit tests pass (see TEST_REPORT.md). Pipeline integration may require ≥2 GB free RAM.

### 4. Code Quality & Linting

```bash
ruff check .
```

Expected output: `All checks passed!`.

### 5. Launch the Standalone Web Dashboard

The web dashboard requires **no web server** or Python backend. Simply open `dashboard/index.html` in any web browser:

```bash
# Windows
start dashboard\index.html

# macOS
open dashboard/index.html

# Linux
xdg-open dashboard/index.html
```

---

## Machine Learning Methodology

### 1. Preprocessing & Quality Assurance
- **Customer ID Validation:** Rows missing `CustomerID` represent guest checkouts and are excluded from customer-level behavioral tracking.
- **Cancellation Segregation:** Invoices prefixed with `C` or containing negative quantities are segregated. They are used to calculate customer return rates without distorting gross positive spend.
- **Price & Quantity Checks:** Zero or negative unit prices and non-positive purchase quantities are filtered out.
- **Deduplication:** Identical duplicate transaction logs are stripped.

### 2. RFM Feature Extraction & Log Transformation
- **Recency ($R$):** Elapsed calendar days between reference date ($T_{max} + 1\text{ day}$) and customer's latest transaction date.
- **Frequency ($F$):** Distinct positive invoice count.
- **Monetary ($M$):** Cumulative spend $\sum (\text{Quantity} \times \text{UnitPrice})$.
- **Transformation:** Retail monetary and frequency features follow heavy right-skewed power-law distributions. We apply $x \mapsto \log(1 + x)$ followed by `StandardScaler` to normalize distributions before Euclidean distance calculations.

### 3. Clustering Algorithms & Evaluation Guardrails
The platform implements and compares three unsupervised clustering approaches:
- **K-Means (k-means++):** Uses intelligent centroid seeding to partition standardized feature space into balanced, spherical clusters. Evaluated across $k \in [2, 8]$ via Inertia and Silhouette curves.
- **DBSCAN:** Explores density-connected regions $(\varepsilon=0.45, \text{min\_samples}=5)$ by default (tunable). Uncovers irregular density structures and explicitly flags high-spending enterprise outliers as noise points ($-1$).
- **Agglomerative Hierarchical:** Uses Ward linkage to minimize within-cluster variance bottom-up, providing deterministic hierarchical grouping.

### 4. Model Selection Rationale
In retail customer segmentation, **K-Means++** is recommended as the primary production model. Standardized log-RFM features form coherent convex clusters that map cleanly to operational business tiers. DBSCAN (default $\varepsilon=0.45$, `min_samples=5`) is retained for density-based exploration; on the synthetic demo it produced multiple clusters with low noise, but results remain data-dependent and secondary to K-Means++ for managerial personas. Earlier defaults ($\varepsilon=0.65$) could collapse to a single dense cluster — see `output/reports/dbscan_sensitivity_analysis.json`.

---

## Discovered Customer Personas & Business Action Plans

> **Example results from the synthetic demo dataset** (`python main.py --demo --k 4`).  
> These figures are **not** UCI Online Retail production results. Real-data personas will differ.

| Cluster ID | Persona Name | Customer Share | Revenue Share | Avg Spend | Avg Recency | Strategic Recommendation |
|---|---|---|---|---|---|---|
| **0** | **Loyal Repeat Regulars** | 29.2% | 39.8% | $811.31 | 20.3 days | Incentivize cross-category purchases ($75 threshold offers) to elevate basket size. Automated replenishment alerts. |
| **1** | **Recent Starters & Promising** | 33.8% | 3.7% | $65.51 | 38.9 days | Multi-stage onboarding email sequence with product education. Time-limited second-purchase discount (valid 14 days). |
| **2** | **Top-Tier VIP Champions** | 9.8% | 47.8% | $2,903.25 | 2.4 days | Dedicated VIP concierge, executive previews of new product releases, early access. Avoid margin-eroding discounts. |
| **3** | **At-Risk / Lapsed Customers** | 27.2% | 8.7% | $189.47 | 188.5 days | Multi-channel win-back reactivation campaigns with personalized incentives. Customer satisfaction exit surveys. |

---

## Standalone Web Dashboard Features

### Dashboard CSV upload requirements
- **Raw transaction CSV (strict minimum):** `InvoiceNo`, `Quantity`, `UnitPrice`, `CustomerID` (all required). Incomplete schemas are rejected with a clear error. Empty or header-only files are rejected. `InvoiceDate` is recommended when present.
- **RFM CSV:** `recency`, `frequency`, `monetary` (all required). Optional: `customer_id` / `CustomerID`, `avg_order_value`.
- **Demo file:** `data/sample_customers_demo.csv` (synthetic; tagged `DataClassification = SYNTHETIC_DEMO`).
- **Note:** Dashboard segment labels use client-side heuristics and are **not** the Python-trained cluster models.

- **Responsive SaaS UI:** Modern styling with clean typography, status indicators, and mobile-friendly layouts.
- **Client-Side CSV Ingestion:** Ingests raw transaction CSV logs or pre-calculated RFM tables directly into client memory via drag-and-drop or file selection.
- **Native Canvas Visualizations:** Zero external charting CDNs required. Renders high-DPI retina charts for volume vs revenue comparisons and behavioral scatter plots.
- **Customer Explorer Table:** Real-time search filtering by Customer ID or Segment name, accompanied by smooth pagination.
- **Export Sanitization:** Exports segmented analysis to CSV while prepending single quotes to fields beginning with `=`, `+`, `-`, or `@` to prevent CSV formula injection.

---

## Security & Compliance Audit

- **Zero Unauthorized External Telemetry:** No user data or transaction logs are sent to external web endpoints or commercial LLM APIs.
- **Safe DOM Rendering:** Avoids `innerHTML` on user-supplied CSV fields; uses strict `textContent` and `createElement` DOM manipulation to eliminate Cross-Site Scripting (XSS).
- **Formula Injection Defense:** CSV exports are filtered against Dynamic Data Exchange (DDE) formulas.
- **Path Traversal Protection:** All file reads strictly adhere to project-local paths.

---

## Known Limitations & Future Roadmap

1. **Static RFM Boundaries:** The client-side dashboard currently utilizes empirical quantile heuristics for instant offline categorization. Integrating WebAssembly-compiled scikit-learn models (via Pyodide) would enable client-side K-Means execution.
2. **Seasonality:** The baseline pipeline calculates RFM over the available observation window. Implementing rolling 90-day RFM windows would help detect micro-seasonal shifts.
3. **Customer Lifetime Value (CLV):** Future iterations can integrate BG/NBD and Gamma-Gamma probabilistic models to predict expected future transactions and customer financial value.

---

## License

This project is licensed under the [MIT License](LICENSE).
