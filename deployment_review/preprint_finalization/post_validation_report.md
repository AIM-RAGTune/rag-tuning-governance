# Preprint Finalization Validation Report

- Publication validator: passed.
- `pytest -q tests/publication`: 265 passed, 6 skipped.
- `make validate-publication`: passed.
- `make test`: 265 passed, 6 skipped.
- `python3 -m compileall src scripts`: passed.
- `git diff --check`: passed.
- arXiv abstract check: 1278/1920 characters, 164 words.
- Clean temporary source-manifest LaTeX build: passed with TeX layout warnings.
- Page count: 20.
- Undefined references: 0.
- Missing citations: 0.
- Missing figures: 0.
- Raster figures: 0.
- Protected claim/status paths: unchanged.
- Immutable tag: `v0.1.0-rc1` resolves to `ff529005bf7d00a0c3f79ba991563f3923d63205`.

## Hygiene Scans

- Large-file scan: no large tracked public artifact; ignored `.local_data` cache observed.
- Raw text / prompt / generated-answer scan: expected schema, config, test, hash, and false-flag references only.
- Secret scan: expected scanner, test, and environment-variable-name references only.
- Private-path scan: expected test assertions only.
- Overclaim scan: expected unsupported-claim boundary text and scanner/test patterns only.

## Not Yet Done

Standing workflow/package actions are not recorded as complete in this report. They should be verified after this content PR is opened and, where required, merged.
