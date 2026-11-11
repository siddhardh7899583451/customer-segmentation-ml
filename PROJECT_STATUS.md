# Project Status: Customer Segmentation & Behavioral Analytics Platform

**Current State:** Substantially complete after final remediation  
**Last remediation:** 2026-11-11  
**Overall progress:** ~92% (see PROJECT_AUDIT.md)

## Status summary

| Area | Status |
|------|--------|
| Python ML pipeline (synthetic) | Verified |
| Python ML pipeline (full UCI) | Cleaning + RFM verified; full multi-figure run may need >=2-4 GB free RAM |
| Unit / integration tests | Majority pass; pipeline integration is memory-sensitive |
| Dashboard (static) | Code complete + strengthened CSV validation; browser testing is manual |
| Security | Static review passed (formula injection + XSS mitigations) |
| Hugging Face static package | Internally consistent; NOT DEPLOYED |
| Documentation | Contradictions reconciled; no 100% / 31/31 claims without evidence |

## Verified artifacts

- data/sample_customers_demo.csv — labeled synthetic
- output/reports/data_quality_report.json — synthetic run metrics
- output/reports/dbscan_sensitivity_analysis.json — DBSCAN grid on synthetic RFM
- output/figures/* — synthetic-run figures
- deployment/huggingface-space/ — static SDK package

## Honest limitations

1. Automated browser tests were not executed (Playwright unavailable).
2. Full UCI + multi-plot pipeline is memory-intensive on low-RAM hosts.
3. Dashboard JS segmentation is rule-based heuristics, not the Python-trained models.
4. Release ZIP excludes .git and virtual environments; remote Git history is not claimable from the package alone.
5. DBSCAN remains secondary to K-Means++; defaults adjusted after sensitivity analysis but stay data-dependent.

See PROJECT_AUDIT.md, SECURITY_AUDIT.md, TEST_REPORT.md, and NEXT_STEPS.md.
