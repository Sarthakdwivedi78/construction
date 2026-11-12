#!/usr/bin/env bash
set -euo pipefail

: "${PIXTRAL_SERVER_API_KEY:?Set PIXTRAL_SERVER_API_KEY in the GPU host's secret manager}"

docker run --rm \
  --gpus all \
  --shm-size 32g \
  --publish 8200:8200 \
  --volume pixtral-hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model a1273352/pixtral-12b-construction-safety \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8200 \
  --api-key "$PIXTRAL_SERVER_API_KEY"
