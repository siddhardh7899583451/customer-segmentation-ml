# PROJECT AUDIT — Customer Segmentation & Behavioral Analytics Platform

**Audit Date:** 2026-11-11  
**Last documentation correction:** 2026-11-11 (final documented release)  
**Auditor Role:** Senior ML Engineer / QA / Security / Documentation  
**Source of Truth:** `customer-segmentation-ml-final.zip` and subsequent documentation-only correction  

**Environment notes:** Limited RAM (~1.2 GB) in the audit environment. Full UCI Online Retail (541k rows) plus multi-model evaluation and high-DPI matplotlib figures can OOM. Synthetic demo and sampled real-data validation were performed successfully.

---

## Executive Summary

The project is substantially complete (~92% by independent audit criteria) and well-structured. Core ML pipeline, RFM engineering, multi-algorithm clustering with evaluation guardrails, empirical personas, standalone dashboard, tests, security hygiene, and Hugging Face static package are present and largely functional.

**DBSCAN defaults (current, verified on synthetic demo):**
- `eps = 0.45`
- `min_samples = 5`
- Exact default run on standardized synthetic RFM (500 customers): **5 clusters**, **5 noise points (1.0%)**, Silhouette **0.3959**, Davies-Bouldin **1.0901**, Calinski-Harabasz **411.87**
- Metrics are valid (2 ≤ k < N after excluding noise). Results are **not** from the full UCI dataset.

Remediation already applied in prior passes:
- `load_dataset` missing-path + synthetic fallback preference fixed
- Dashboard raw-transaction CSV validation strengthened
- Documentation contradictions reconciled (no remaining 100% / 31/31 claims without evidence)
- Figure DPI default reduced for memory efficiency (mathematical results unchanged)
- DBSCAN sensitivity analysis performed; default moved from 0.65 → 0.45

K-Means++ remains the primary recommended model for managerial segmentation.

---

## Requirement Verification Table

| Requirement              | Existing Implementation | Verification Evidence | Status | Required Action |
| ------------------------ | ----------------------- | --------------------- | ------ | --------------- |
| Data loading             | `src/data_loader.py` supports UCI download, local xlsx/csv, synthetic fallback | Download succeeded; loader missing-path fix applied | Complete and verified | None |
| Data preprocessing       | `DataCleaner` removes missing CustomerID, duplicates, invalid price/qty, segregates cancellations | Quality report: 541909 → 392692 rows, 4338 customers (UCI observed) | Complete and verified | None |
| Data quality reporting   | `DataQualityReport` + JSON export | `output/reports/data_quality_report.json` (synthetic) and `data_quality_report_uci_observed.json` | Complete and verified | None |
| RFM feature engineering  | `RFMFeatureExtractor` with log1p + StandardScaler | Extracted profiles on synthetic and sampled real data; parity test passed | Complete and verified | None |
| K-Means++                | `KMeansSegmenter` k-means++ | Synthetic full run + real sample | Complete and verified | None |
| DBSCAN                   | `DBSCANSegmenter`; defaults `eps=0.45`, `min_samples=5` | **Exact default verified on synthetic demo**: 5 clusters, 1.0% noise, Silhouette 0.3959, DBI 1.0901, CHI 411.87. Not verified on full UCI. | Complete and verified (synthetic default) | None for default path; UCI DBSCAN remains optional |
| Agglomerative clustering | Ward linkage | Synthetic + real sample | Complete and verified | None |
| Clustering evaluation    | Silhouette / DBI / CHI only when 2 ≤ k < N and non-noise | Guardrails correct; metrics reported only when valid | Complete and verified | None |
| PCA visualization        | 2D/3D PCA + matplotlib plots | Figures generated on synthetic | Complete and verified | None |
| Customer personas        | `PersonaProfiler.profile_clusters` empirical ranking | Synthetic personas + real-sample profiles | Complete and verified | None |
| Recommendations          | Embedded in persona objects / JSON | Present in reports | Complete and verified | None |
| Standalone dashboard     | HTML/CSS/vanilla JS, no backend | Files present; JS syntax checked | Implemented; **browser automation not executed** | Manual Chrome verification |
| CSV upload               | File input + drag-drop in `app.js` | Code present | Implemented; browser not automated | Manual Chrome verification |
| CSV validation           | Strict required-column sets | Raw transaction requires **InvoiceNo, Quantity, UnitPrice, CustomerID**; empty/header-only rejected; RFM schema (recency, frequency, monetary) preserved | Complete and verified (code + logic) | Manual browser confirmation recommended |
| Customer explorer        | Paginated table, search, filter, sort | `renderCustomerTable` uses createElement/textContent | Implemented; browser not automated | Manual Chrome verification |
| Search and filtering     | Client-side | Present | Implemented; browser not automated | Manual Chrome verification |
| CSV export               | Sanitized formula-injection export | `sanitizeForCsv` present | Complete and verified (code review) | None |
| Browser testing          | None automated in repo | Playwright/Selenium not available in audit environment | Not implemented | Manual Chrome procedure in NEXT_STEPS.md |
| Automated testing        | 31 pytest items | Prior remediation: **30 passed, 1 deselected** (pipeline memory-sensitive) | Partially complete | Treat pipeline test as environment-dependent |
| Security review          | CSV formula sanitize, textContent usage, no eval | Static review positive | Complete and verified (static) | See SECURITY_AUDIT.md |
| Hugging Face package     | `deployment/huggingface-space/` static SDK | Structure consistent with dashboard; JS validation mirrored | Complete and verified (structure) | **NOT DEPLOYED** |
| Documentation            | README + status docs + this audit | Contradictions reconciled in remediation and this documentation correction | Complete and verified for consistency | Keep aligned on future changes |
| Release package contents | Clean ZIP without runtime envs | Virtual environments, caches, credentials, nested ZIPs excluded | Complete and verified | None |

---

## Real Dataset Validation (UCI Online Retail)

- Source: https://archive.ics.uci.edu/dataset/352/online+retail
- Observed full cleaning metrics (prior audit):
  - 541909 → 392692 valid rows
  - Missing CustomerID: 135080; duplicates: 5225; cancelled: 8872; invalid price: 40
  - Unique customers: 4338; date range 2010-12-01 → 2011-12-09; 37 countries
- Sampled (50–80k row) validation: RFM, K-Means, Agglomerative, PCA, personas succeeded
- **DBSCAN exact default was verified on synthetic demo only**, not on full UCI
- Full pipeline + multi-figure generation on complete UCI can OOM under ~1.2 GB RAM

---

## DBSCAN Defaults and Limitations

| Item | Value |
|------|--------|
| Default eps | **0.45** |
| Default min_samples | **5** |
| Synthetic demo result | 5 clusters, 5 noise (1.0%), Silhouette 0.3959, DBI 1.0901, CHI 411.87 |

**Honest comparison under the same min_samples=5:**
- eps=0.40 → 7 clusters, 2.8% noise, Silhouette **0.4395** (higher than 0.45)
- eps=0.45 → 5 clusters, 1.0% noise, Silhouette 0.3959
- eps=0.50 → 4 clusters, 0.4% noise, Silhouette 0.4258

eps=0.45 was retained as a middle value in the useful band (~0.40–0.50) with fewer micro-clusters and very low noise; it was **not** chosen to maximize Silhouette. The earlier grid peak at eps=0.40 / min_samples=8 (Silhouette 0.5054) required changing min_samples and was not adopted as the default pair.

**K-Means++ remains the primary recommended model.** DBSCAN is retained for density-based exploration and is data-dependent. DBSCAN is not claimed to be universally optimal.

Details: `output/reports/dbscan_sensitivity_analysis.json` and `output/reports/model_comparison.json`.

---

## CSV Validation (Dashboard)

- **Raw transaction schema (strict):** InvoiceNo, Quantity, UnitPrice, CustomerID (all required). Incomplete sets produce clear error messages.
- **Empty / header-only CSVs:** Rejected.
- **RFM schema:** recency, frequency, monetary — preserved and accepted.
- Parser does not accept a raw transaction file merely because it contains a single related column.

---

## Known Limitations (Honest)

1. Automated browser testing (Playwright/Selenium) was **not** executed.
2. Full unsampled UCI pipeline with all figures remains memory-intensive on low-RAM hosts.
3. Dashboard JS segmentation is **rule-based heuristics**, not the Python-trained clustering models.
4. Full pipeline integration test remains memory-sensitive under ~1.2 GB free RAM.
5. No GitHub push performed; release ZIP excludes `.git`.
6. Hugging Face static package is prepared but **NOT DEPLOYED**.
7. Under min_samples=5, eps=0.40 scores a higher Silhouette than the chosen default eps=0.45.

---

## Release Package Contents

The release archive:
- **Includes:** source, dashboard, tests, reports, documentation, Hugging Face static package, synthetic demo CSV
- **Excludes:** `.git`, `.venv` / virtual environments, `__pycache__`, `.pytest_cache`, nested ZIP files, credentials, `.env` files

Windows users should create a local venv with the PowerShell commands in the README. The release package does **not** ship a virtual environment.

---

## Final Remediation & Documentation Correction Log

1. `data_loader` missing-path fallback fix
2. Dashboard CSV required-column validation strengthened (InvoiceNo, Quantity, UnitPrice, CustomerID; empty/header-only rejected)
3. DBSCAN sensitivity analysis; default eps 0.65 → 0.45; exact default metrics recorded on synthetic demo
4. Figure DPI default reduced for memory (math unchanged)
5. Documentation contradictions reconciled (README, STATUS, CHECKLIST, TEST_REPORT, NEXT_STEPS, this audit)
6. This documentation-only correction: PROJECT_AUDIT aligned with final defaults, CSV rules, package contents, and verification statuses

**No claim of 100% completion.**
