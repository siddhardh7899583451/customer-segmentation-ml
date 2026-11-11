# Error Log & Issue Resolution Tracker

| Incident ID | Timestamp | Phase / Component | Description | Root Cause | Resolution | Status |
|---|---|---|---|---|---|---|
| ERR-001 | 2026-11-10T13:08 | Initial Setup | Write cancellation on `generate_data.py` during background install | Tool timeout during concurrent pip execution | Directory cleaned in Phase 1; Python 3.12 `.venv` initialized | Resolved |
| ERR-002 | 2026-11-10T13:36 | Feature Engineering | Pandas 3/4 deprecation warnings on `inplace=True` | Future pandas deprecation of inplace modifications | Updated all DataFrame operations to standard functional assignments (`df = df.rename(...)`) | Resolved |
| ERR-003 | 2026-11-10T13:36 | Pipeline | Custom output directory override test failed in temp directory | Hardcoded global constant `OUTPUT_DIR` instead of instance `self.config.output_dir` | Updated pipeline to route all report and figure exports through `self.config.output_dir` | Resolved |
| ERR-004 | 2026-11-10T13:42 | Client Parity | 1-day discrepancy between Python `.dt.days` and JS elapsed days | JavaScript test simulated `Math.round` instead of integer day truncation | Aligned JS client engine in `app.js` to `Math.floor` for exact calendar day parity | Resolved |
| ERR-005 | 2026-11-10T13:45 | Code Quality | Ruff warnings for import ordering and unused unpacked loop variables | Unused unpacked tuple variables in test fixtures | Auto-formatted imports and prefixed unused fixture variables with `_` | Resolved |
