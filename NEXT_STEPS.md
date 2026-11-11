# Next Steps & Post-Remediation Roadmap

## Current verified state

The project is substantially complete (~92% by the independent audit criteria in PROJECT_AUDIT.md).
Core ML pipeline (synthetic path), RFM correctness, evaluation guardrails, security mitigations,
and documentation consistency have been verified. Full browser automation and full-UCI multi-figure
runs remain environment-limited.

## Action items requiring explicit owner authorization

1. **GitHub remote push** — not performed. The release ZIP does not include a `.git` directory;
   initialize or re-attach history locally if needed.
2. **Hugging Face Space deployment** — static package is ready under `deployment/huggingface-space/`
   but **NOT DEPLOYED**. Deploy only with owner authorization.
3. **Full UCI production run** on a machine with ≥2–4 GB free RAM:
   ```powershell
   python main.py --data "data\Online Retail.xlsx" --k 4
   ```

## Manual dashboard verification (Chrome)

1. Open `dashboard\index.html` in Chrome.
2. Confirm demo data loads and KPIs/charts render.
3. Upload `data\sample_customers_demo.csv` and verify results.
4. Optionally test invalid CSVs (missing columns, empty file) and confirm error messages.
5. Exercise search, filter, pagination, and CSV export.
6. Open DevTools console and note any errors.

## Future engineering ideas (optional)

- WebAssembly / Pyodide in-browser ML
- Probabilistic CLV models
- CRM export webhooks
