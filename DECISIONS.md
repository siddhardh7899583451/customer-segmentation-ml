# Architecture and Technical Decisions Log

## Decision 001: Python 3.12 Virtual Environment
- **Context:** System has Python 3.12, 3.13, 3.14 installed. Project specification explicitly requires Python 3.12.
- **Decision:** Explicitly bound the virtual environment creation to `C:\Program Files\Python312\python.exe`.
- **Outcome:** Clean `.venv` created and verified running Python 3.12.0.

## Decision 002: Zero-Backend Browser Dashboard
- **Context:** Project requires a standalone web dashboard without any Python web frameworks (no Gradio, Streamlit, Flask, FastAPI).
- **Decision:** Built a pure HTML5, modern CSS3, and vanilla JavaScript client-side application.
- **Outcome:** The dashboard can be served statically, opened locally via `file:///`, or hosted on Hugging Face static spaces. All visualizations (PCA scatter, RFM distributions, segment comparisons) use pure inline Canvas/SVG or offline-compatible JS without requiring an active Python server.

## Decision 003: Real Dataset vs Synthetic Fallback
- **Context:** Primary dataset target is UCI Online Retail. Network environments or archive access may be restricted or slow.
- **Decision:** Build a flexible data loader that attempts to download/load the official UCI dataset, while providing a mathematically sound, clearly labeled synthetic demo dataset generator (`data/sample_customers_demo.csv`). The UI and documentation explicitly label synthetic results as demo data.

## Decision 004: Clustering Algorithms & Evaluation Guardrails
- **Context:** Different clustering algorithms (K-Means, DBSCAN, Agglomerative) have different assumptions and geometric properties.
- **Decision:**
  - Standardize features using `StandardScaler` to ensure scale invariance across Recency, Frequency, and Monetary values.
  - Compute Silhouette score, Davies-Bouldin, and Calinski-Harabasz metrics only when mathematically valid (number of clusters $2 \le k < N$ and clusters with non-noise points).
  - Explicitly handle DBSCAN noise points (-1) during evaluation and report noise counts transparently.

## Decision 005: Parity Between Python ML Pipeline and JavaScript Dashboard
- **Context:** The web dashboard performs client-side RFM calculations for uploaded CSVs.
- **Decision:** Maintain identical business logic in `src/feature_engineering.py` and `dashboard/app.js` (e.g. invoice date parsing, cancellation filtering with 'C' invoice prefixes, total spend formula $Quantity \times UnitPrice$). Implement automated parity tests in pytest to verify identical results.

## Decision 006: Security & Formula Injection Defense
- **Context:** Handling user-supplied CSV files and displaying customer attributes in DOM.
- **Decision:**
  - Sanitize all CSV exports by prepending a single quote `'` to any field starting with `=`, `+`, `-`, or `@`.
  - Avoid `innerHTML` with user inputs; use `textContent` and `createElement` for DOM rendering to prevent XSS.
