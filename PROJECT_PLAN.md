# Project Plan: Customer Segmentation & Behavioral Analytics Platform

## 1. Executive Summary
The Customer Segmentation & Behavioral Analytics Platform is an end-to-end unsupervised machine learning and business intelligence system. It transforms transactional customer records into actionable customer segments (personas), computes RFM (Recency, Frequency, Monetary) metrics, compares clustering models (K-Means++, DBSCAN, Agglomerative Hierarchical), visualizes high-dimensional spaces via PCA, and delivers an interactive zero-backend vanilla HTML/CSS/JS dashboard ready for Hugging Face Spaces static deployment.

---

## 2. Architecture & Directory Structure

```
customer-segmentation-ml/
├── .gitignore
├── requirements.txt
├── README.md
├── PROJECT_PLAN.md
├── PROJECT_STATUS.md
├── TASK_CHECKLIST.md
├── DECISIONS.md
├── ERROR_LOG.md
├── TEST_REPORT.md
├── DEPLOYMENT_STATUS.md
├── NEXT_STEPS.md
├── main.py                          # CLI runner for the ML pipeline
├── src/
│   ├── __init__.py
│   ├── config.py                    # Project paths, hyperparameters, constants
│   ├── data_loader.py               # Dataset ingestion & download helper
│   ├── preprocessor.py              # Cleaning, filtering, validation, outlier handling
│   ├── feature_engineering.py       # RFM computation, transaction aggregation, behavioral metrics
│   ├── models.py                    # K-Means++, DBSCAN, Agglomerative clustering implementations
│   ├── evaluation.py                # Silhouette, Davies-Bouldin, Calinski-Harabasz metrics
│   ├── personas.py                  # Profiling, automated segment labeling, marketing recommendations
│   ├── visualization.py             # Matplotlib/Seaborn PCA, cluster scatter, distributions
│   ├── pipeline.py                  # End-to-end reproducible ML execution pipeline
│   └── utils.py                     # Logging, security sanitization, CSV export
├── data/
│   └── sample_customers_demo.csv    # Clearly labeled synthetic demo dataset
├── output/
│   ├── reports/                     # Data quality report, model comparison metrics (JSON/CSV)
│   ├── figures/                     # PCA 2D/3D plots, cluster distributions, elbow curve
│   └── segmented_customers.csv      # Customer-level segmentation output
├── dashboard/
│   ├── index.html                   # Zero-backend standalone responsive web dashboard
│   ├── styles.css                   # Modern clean CSS
│   ├── app.js                       # Vanilla JavaScript for client-side RFM, clustering & charts
│   └── sample_demo_data.js          # Built-in synthetic demo data for instant client evaluation
├── deployment/
│   └── huggingface-space/           # Standalone static deployment package with HF metadata
│       ├── README.md                # Hugging Face metadata frontmatter
│       ├── index.html
│       ├── styles.css
│       ├── app.js
│       └── sample_demo_data.js
├── notebooks/
│   └── customer_segmentation_walkthrough.ipynb  # Comprehensive pedagogical notebook
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
    └── test_js_parity.py            # Verifies JS client calculations match Python results
```

---

## 3. Implementation Phases

1. **Phase 1: Project Clean & Reset** (Completed)
   - Verified exact working directory `C:\Projects\customer-segmentation-ml`.
   - Cleaned all residual files, verified completely empty directory.

2. **Phase 2: Environment Setup** (In Progress)
   - Locate Python 3.12 specifically (`C:\Program Files\Python312\python.exe`).
   - Create `.venv` using Python 3.12.
   - Install dependencies (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pytest`, `ruff`, `openpyxl`).
   - Pin `requirements.txt`.
   - Exclude web frameworks (no Gradio, Streamlit, Flask, FastAPI).

3. **Phase 3: Dataset Ingestion & Synthetic Fallback**
   - Implement data loader capable of downloading or reading the UCI Online Retail dataset (`Online Retail.xlsx` / `.csv`).
   - Provide a realistic synthetic dataset generator (`src/synthetic_data.py` / `data/sample_customers_demo.csv`) clearly labeled as demo data.

4. **Phase 4: Data Preprocessing & Feature Engineering**
   - Clean transaction data: remove null `CustomerID`, filter negative quantities (cancellations recorded with 'C'), validate prices ($>0$), deduplicate rows.
   - Compute RFM metrics per customer:
     - Recency: Days between reference date and latest customer transaction.
     - Frequency: Count of distinct purchase invoices.
     - Monetary: Sum of $(Quantity \times UnitPrice)$.
   - Compute behavioral metrics: Average Order Value (AOV), cancellation rate, product diversity, return counts.
   - Robust outlier detection (IQR / Log transform).

5. **Phase 5: Machine Learning Models & Evaluation**
   - Feature standardisation (`StandardScaler`).
   - Models:
     - K-Means with K-Means++ initialization (Elbow search $k=2..8$).
     - DBSCAN with $\varepsilon$ and `min_samples` exploration (noise point detection).
     - Agglomerative Hierarchical Clustering (Ward linkage).
   - Evaluation metrics:
     - Silhouette Score (strictly valid when $2 \le k < n$, not all noise).
     - Davies-Bouldin Score.
     - Calinski-Harabasz Score.
     - Inertia / Elbow Method for K-Means.
   - Dimensionality reduction: PCA 2D projection.
   - Rigorous model comparison and selection rationale.

6. **Phase 6: Persona Profiling & Business Action Plans**
   - Derive cluster persona labels from statistical medians/means (e.g. "High-Value Champions", "Loyal Regulars", "At-Risk Churners", "New Starters", "Bargain Hunters").
   - Compute per-cluster statistics: Count, Percentage, Mean Monetary, Mean Frequency, Mean Recency.
   - Produce tailored business and marketing recommendations for each persona.

7. **Phase 7: Modular Python Package & CLI**
   - Clean, modular code in `src/`.
   - Logging, config dataclass, defensive error handling.
   - CLI script (`main.py`) with arguments for data path, model choice, output folder.
   - Step-by-step Jupyter walkthrough notebook.

8. **Phase 8: Standalone Browser Web Dashboard**
   - Pure HTML, CSS, Vanilla JavaScript (zero Python backend).
   - Responsive layout with dark/light visual appeal.
   - Features: CSV file upload, dataset schema validation, summary cards, RFM distribution histograms, PCA/cluster scatter plot using pure SVG/Canvas (zero external dependencies required), persona cards, customer-level data table with search/pagination, CSV export.
   - Built-in demo data option.
   - Strict XSS protection and formula injection prevention.

9. **Phase 9: Hugging Face Static Deployment Package**
   - Self-contained directory in `deployment/huggingface-space/`.
   - Valid Hugging Face `README.md` metadata header (`sdk: static`, `app_file: index.html`).
   - Verified relative asset links.

10. **Phase 10: Automated Testing Suite**
    - Pytest tests across all Python modules.
    - Python vs JavaScript parity test for RFM calculations.
    - Edge cases: empty dataset, single customer, NaN values, uniform data, outliers.

11. **Phase 11: Security, Code Quality & Ruff Audit**
    - Ruff linting and formatting.
    - Security audit: XSS prevention, CSV formula injection neutralization, file path validation.

12. **Phase 12: Recovery & Traceability Documentation**
    - Real-time updates to all project tracking documents.

13. **Phase 13: GitHub Readiness**
    - Git initialization, comprehensive `.gitignore`, MIT License, clear documentation.

14. **Phase 14: Three-Tier Final Verification**
    - Round 1: Functional verification.
    - Round 2: Technical & mathematical verification.
    - Round 3: Clean-state verification.

15. **Phase 15: Final Artifacts & ZIP Generation**
    - Package clean source ZIP: `customer-segmentation-ml-final.zip`.
