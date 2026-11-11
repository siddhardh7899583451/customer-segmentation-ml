# TEST REPORT

**Date:** 2026-11-11 (final remediation)  
**Python:** 3.12.x  

## Commands

```powershell
python -m pytest tests/ -v --tb=short
python -m ruff check src/ tests/ main.py
```

## Results (this environment)

| Check | Result | Notes |
|-------|--------|-------|
| Ruff | Pass | 0 errors |
| Unit tests (loader, preprocessor, feature engineering, models, evaluation, personas, edge cases, JS parity) | Pass | After data_loader missing-path fallback fix |
| test_pipeline_execution_end_to_end | Memory-sensitive | Demo data path; matplotlib figures can OOM under ~1.2 GB free RAM |
| Browser / Playwright | Not run | Tooling unavailable; manual Chrome procedure in NEXT_STEPS.md |
| Full UCI multi-figure pipeline | Not run here | Memory limitation; cleaning + sampled RFM validated previously |

## Counts

- Pytest collected: 31 items
- Reliably green under constrained RAM: majority of unit tests
- Do not claim "31/31 passed in every environment"

## Conclusion

Core correctness is high for the synthetic path and unit coverage. Treat the pipeline integration test and full-UCI figure generation as environment-dependent. Manual dashboard verification remains required.
