# Public release validation

- Date: 2026-07-18
- Scope: README, citation metadata, requirements, configs, source code, scripts, tests, docs, docs/assets, and public Markdown reports.
- Repository name recommendation: `chest-xray-pneumonia-classification`

## Excluded from public release

- Raw chest X-ray images
- Generated manifests and full split CSV files
- `results/` experiment directories
- Model weights (`*.keras`, `*.h5`)
- Calibration/joblib artifacts
- Full prediction CSV files
- Bootstrap detail CSV files
- Console logs
- Failed audit directories and one-time final-test lock audit artifacts

## Local path and sensitive information scan

`scripts/validate_public_release.sh` scanned public Markdown and docs for local paths and local identifiers including Windows drive paths, WSL mount paths, local usernames, machine names, and WSL disk references.

Result: passed.

The script also scanned the public tree for common credential-pattern strings and private-key markers.

Result: passed.

## Tests

Command:

```bash
PYTHON=~/.venvs/chest-xray/bin/python MPLCONFIGDIR=/tmp/matplotlib-chest-xray ./scripts/validate_public_release.sh
```

Result:

```text
87 passed
```

The validation script does not run training, does not require GPU, does not load real test data, and does not access `results/` for validation.

## Markdown and image checks

- README image references: passed
- Markdown relative links: passed
- Public tree artifact exclusion check: passed
- Large CSV exclusion check: passed

## Public results manifest

- File: `docs/PUBLIC_RESULTS_MANIFEST.json`
- Records: 113
- SHA-256: validated by `scripts/validate_public_release.sh`
- Hash validation: passed

## Training, test, and frozen protocol safety

- Training executed during release validation: no
- Final test re-run: no
- Test predictions regenerated: no
- Frozen protocol modified: no
- `PROTOCOL_FROZEN` modified: no
- `TEST_EVALUATED` modified: no
- Original images or manifests modified: no
- Model weights modified: no

## Known unresolved issue

No repository license has been selected yet. A `LICENSE` file was not created because license selection requires explicit user confirmation.

Suggested common options for a code-only research repository include MIT, Apache-2.0, or BSD-3-Clause. Dataset licensing remains governed by the original dataset provider and is not changed by this repository license.
