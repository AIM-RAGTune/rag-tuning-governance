# Post-Sanitization Validation Report

- Overall result: `PASS_WITH_DOCUMENTED_TARGETED_ENV_LIMITATION`
- Raw text scan: `PASS`
- Secret scan: `PASS`
- Private path scan: `PASS`
- Overclaim scan: `PASS`
- Publication validator: `PASS`
- Publication tests: `4 passed`
- Compileall: `PASS`
- `make validate-publication`: `PASS`
- `make test`: `PASS`
- Targeted CRAG hardening test: dependency-limited in the active local export environment because `PyYAML` was unavailable to the pytest interpreter. `PyYAML` remains declared in repository dependency files.

Deployment is allowed because publication gates passed and the dependency-limited targeted test does not indicate a sanitization failure.
