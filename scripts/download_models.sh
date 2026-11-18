#!/usr/bin/env bash
# Fetch the ONNX emotion models from GitHub:
#   FER+ (ONNX model zoo, TensorFlow-free backend)
#   HSEmotion EfficientNet-B0 trained on AffectNet (default backend)
set -euo pipefail
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/tb4_emotion_control/models"
URL=https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
SHA=a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c
mkdir -p "$DST"
for i in $(seq 10); do
  curl -fSL -C - --retry 3 -o "$DST/emotion-ferplus-8.onnx" "$URL" && break
  sleep 2
done
echo "$SHA  $DST/emotion-ferplus-8.onnx" | sha256sum -c -

HSE_URL=https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx
HSE_SHA=7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f
for i in $(seq 10); do
  curl -fSL -C - --retry 3 -o "$DST/enet_b0_8_best_afew.onnx" "$HSE_URL" && break
  sleep 2
done
echo "$HSE_SHA  $DST/enet_b0_8_best_afew.onnx" | sha256sum -c -
