# GitHub Release Report

## Release target

- Repository URL: https://github.com/Evelyn-R7/chest-xray-pneumonia-classification
- License: MIT License
- Branch: main
- Commit hash: 669b9ac61f3751038ea9bfa414354b43561c85fa
- Commit message: Release reproducible chest X-ray pneumonia study

## Public release validation

- PUBLIC_RESULTS_MANIFEST SHA-256: 5b75e5913fd27572717fe7914ae21a0f8f8323c9df21d2b553cfb626f62d198f
- Pytest result during public release validation: 87 passed
- Public release validation result: passed
- Public manifest validation: passed, 114 files
- README image reference check: passed
- Markdown relative link check: passed
- Sensitive/local path validation: passed by the public release validation script

## Git release status

- Git identity: Evelyn-R7 <Evelyn.ZXR@outlook.com>
- Files committed: 115
- Committed file payload size: 1,519,231 bytes
- Large committed files above 5 MiB: none detected
- Push status: pushed successfully to `origin/main`
- GitHub CLI authentication after device verification: token available
- Remote repository creation: completed with GitHub CLI
- Remote empty check before push: passed

## Explicit exclusions

The release commit does not include:

- raw or processed image data
- `data/`
- `results/`
- model weights
- prediction detail CSV files
- local final-test lock files
- local audit artifacts
- virtual environments
- Python cache directories

## Safety confirmations

- Training executed during release: no
- Final test re-evaluated during release: no
- Predictions regenerated during release: no
- Frozen protocol modified during release: no
- Original images modified during release: no
- Manifest files modified during release: no
- Model weights modified during release: no

## Notes

The GitHub repository was created with GitHub CLI after device verification, confirmed empty, and the local `main` branch was pushed without force.
