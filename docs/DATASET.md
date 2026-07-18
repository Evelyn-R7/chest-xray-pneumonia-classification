# Dataset

This project uses the Kaggle dataset `paultimothymooney/chest-xray-pneumonia` (Chest X-Ray Images, Pneumonia). Raw images are not distributed in this repository.

The original code and documentation in this repository are released under the MIT License. The chest X-ray data remain governed by the original data provider's terms. ImageNet pretrained weights and third-party dependencies follow their own licenses.

Download example:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

Expected local layout after extraction:

```text
data/raw/chest_xray/
  train/
  val/
  test/
```

The original public dataset has a very small official validation split (16 images), which is not suitable for robust model selection. This project therefore performs a leakage-aware re-splitting process. Patient IDs are inferred from filename patterns because external patient metadata are not provided.

The v3_clean protocol excludes 457 PNEUMONIA images from model development because their inferred patient IDs overlap the official test patients. The final v3_clean counts are:

| split | images |
| --- | ---: |
| train | 3821 |
| validation | 954 |
| test | 624 |

The final test set is the official test split and is isolated until model selection and threshold freezing are complete.

Data licensing and citation should follow the Kaggle dataset page and original data providers. This repository does not re-license the images.
