# Reproducibility

The original project was developed on Windows with WSL2 and Python 3.11. The local WSL distribution name is not required for other users.

Core environment:

- Python 3.11
- TensorFlow 2.21
- pandas, numpy, Pillow, PyYAML, matplotlib, pytest, scikit-learn
- NVIDIA GPU optional for training; CPU is sufficient for audit and unit tests but training will be slow.

Typical setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "tensorflow[and-cuda]==2.21.*"
python -m pip install -r requirements.txt
```

Example workflow:

```bash
python src/create_splits_v3_clean.py --help
python src/check_data_pipeline.py --help
python -m pytest -q
python src/train_cnn.py --help
python src/train_transfer.py --help
```

The one-time final-test lock files are part of the original study audit trail and are not required for ordinary public-code reuse. Do not bypass one-time final-test protections when reproducing the protocol.
