#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON:-python3}"

echo "[public release] Python syntax checks"
"$PYTHON_BIN" -m py_compile $(find src scripts tests -name '*.py' -not -path '*/__pycache__/*')

echo "[public release] Synthetic/no-real-data tests"
"$PYTHON_BIN" -m pytest -q \
  tests/test_final_protocol.py \
  tests/test_final_test_evaluation.py \
  tests/test_posthoc_analysis.py \
  tests/test_transfer_multiseed_aggregation.py \
  tests/test_imbalance_strategies.py

echo "[public release] Check forbidden local paths in public Markdown/docs"
if grep -RInE 'D:\\|D:/|/mnt/d/|/mnt/c/|/home/evelyn|茶小栀|LAPTOP-|ext4\.vhdx|\.codex' README.md docs reports/*.md 2>/dev/null; then
  echo "Forbidden local path or local identifier found" >&2
  exit 1
fi

echo "[public release] Check sensitive tokens"
if grep -RInE 'api[_-]?key|password|KAGGLE_KEY|KAGGLE_USERNAME|BEGIN PRIVATE KEY|client_secret|access_secret' README.md docs reports configs src scripts tests \
  --exclude='validate_public_release.sh' 2>/dev/null; then
  echo "Potential secret found" >&2
  exit 1
fi

echo "[public release] Check public tree excludes private artifacts"
if find README.md CITATION.cff configs src scripts tests docs reports -type f \( -name '*.keras' -o -name '*.h5' -o -name '*.joblib' -o -name '*.log' -o -name 'console.log' -o -name '*.npy' -o -name '*.npz' \) | grep .; then
  echo "Private artifact found in public tree" >&2
  exit 1
fi

echo "[public release] Check for large CSV files in public tree"
if find README.md CITATION.cff configs src scripts tests docs reports -type f -name '*.csv' -size +20k | grep .; then
  echo "Large CSV found in public tree" >&2
  exit 1
fi

echo "[public release] Validate PUBLIC_RESULTS_MANIFEST hashes"
"$PYTHON_BIN" - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('.')
records = json.loads(Path('docs/PUBLIC_RESULTS_MANIFEST.json').read_text())
for record in records:
    path = root / record['relative_path']
    if not path.is_file():
        raise SystemExit(f"Missing manifest file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record['sha256']:
        raise SystemExit(f"Hash mismatch: {path}")
print(f"manifest ok: {len(records)} files")
PY

echo "[public release] Check README image references"
"$PYTHON_BIN" - <<'PY'
import re
from pathlib import Path
text = Path('README.md').read_text()
for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text):
    if target.startswith('http'):
        continue
    if not Path(target).is_file():
        raise SystemExit(f"Missing README image: {target}")
print("README images ok")
PY

echo "[public release] Check Markdown relative links"
"$PYTHON_BIN" - <<'PY'
import re
from pathlib import Path
for md in [Path('README.md'), *Path('docs').glob('*.md'), *Path('reports').glob('*.md')]:
    text = md.read_text(encoding='utf-8')
    for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)#][^)]*)\)', text):
        if target.startswith(('http://', 'https://', 'mailto:')):
            continue
        path = (md.parent / target).resolve()
        if not path.exists():
            raise SystemExit(f"Broken link in {md}: {target}")
print("Markdown links ok")
PY

echo "[public release] SUCCESS: public release validation completed without training, GPU, results access, or test evaluation."
