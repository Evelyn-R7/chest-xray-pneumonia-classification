#!/usr/bin/env bash
set -euo pipefail

model="${1:-}"
cache_dir="$HOME/.keras/models"
mkdir -p "$cache_dir"

case "$model" in
  vgg16)
    filename="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    url="https://storage.googleapis.com/tensorflow/keras-applications/vgg16/$filename"
    expected_md5="6d6bbae143d832006294945121d1f1fc"
    ;;
  efficientnetb0)
    filename="efficientnetb0_notop.h5"
    url="https://storage.googleapis.com/keras-applications/$filename"
    expected_md5="50bc09e76180e00e4465e1a485ddc09d"
    ;;
  *)
    echo "Usage: $0 {vgg16|efficientnetb0}" >&2
    exit 2
    ;;
esac

target="$cache_dir/$filename"
partial="$target.part"
if [[ -f "$target" ]] && echo "$expected_md5  $target" | md5sum --check --status; then
  echo "Weight file already present and valid: $target"
  exit 0
fi

echo "Downloading official ImageNet weights with resume support:"
echo "  $url"
echo "  -> $partial"
curl --location --fail --show-error \
  --retry 50 --retry-all-errors --retry-delay 5 --connect-timeout 30 \
  --continue-at - --output "$partial" "$url"

echo "$expected_md5  $partial" | md5sum --check
mv -f "$partial" "$target"
echo "Weight file downloaded and verified: $target"
