#!/bin/bash
# Example: serve Qwen3.5-4B with a Domino draft model on NVIDIA GPU.
#
# The Domino checkpoint must be a DFlash-style draft model whose config.json
# contains a dflash_config block with projector_type="domino" and the required
# Domino hyper-parameters (emb_dim, gru_hidden_dim, pure_draft_prefix_len,
# shift_label).

set -euo pipefail

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/path/to/Qwen3.5-4B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/path/to/Qwen3.5-4B-Domino}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-16}

python -m vllm.entrypoints.openai.api_server \
    --model "${TARGET_MODEL_PATH}" \
    --speculative-model "${DRAFT_MODEL_PATH}" \
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
    --trust-remote-code
