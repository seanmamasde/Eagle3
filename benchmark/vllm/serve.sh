#!/usr/bin/env bash
# -e/--eagle to launch the eagle3 model
# -h for help
set -euo pipefail

PORT=8000
MODEL_NAME='gemma3-27b'
EXTRA_ARGS=()

while getopts ":e-:" opt; do
  case "$opt" in
    e)  PORT=8001
        MODEL_NAME='gemma3-27b-eagle3'
        EXTRA_ARGS+=(--speculative_config \
'{"method":"eagle3","model":"/models/gemma3-27b-eagle3",
"draft_tensor_parallel_size":1,"num_speculative_tokens":2}') ;;
    \?|h) echo "Usage: $0 [-e|--eagle]"; exit 1 ;;
    -)  case "${OPTARG}" in
          eagle)  PORT=8001
                  MODEL_NAME='gemma3-27b-eagle3'
                  EXTRA_ARGS=(--speculative_config \
'{"method":"eagle3","model":"/models/gemma3-27b-eagle3",
"draft_tensor_parallel_size":1,"num_speculative_tokens":2}')
                  ;;
          help)  echo "Usage: $0 [-e|--eagle]" ; exit 0 ;;
          *)  echo "Unknown option --${OPTARG}" >&2; exit 1 ;;
        esac ;;
  esac
done
shift $((OPTIND-1))

singularity exec --nv \
  --bind /home/seanma0627/src/eagle3/weights:/models \
  vllm.sif \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port "${PORT}" \
    --model /models/gemma3-27b \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --served-model-name "${MODEL_NAME}" \
    "${EXTRA_ARGS[@]}"
