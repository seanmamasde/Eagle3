#!/usr/bin/env bash
# -e/--eagle to launch the eagle3 model
# -h for help
set -euo pipefail

PORT=8000
MODEL='/models/taide'

while getopts ":e-:" opt; do
  case "${opt}" in
    e) PORT=8001 ;;
    -)
      case "${OPTARG}" in
        eagle) PORT=8001; ;;
        *) echo "Unknown option --${OPTARG}" >&2; exit 1 ;;
      esac ;;
    ?|h) echo "Usage: $0 [-e|--eagle]" ; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

REPO=/home/seanma0627/src/eagle3
TOK="$REPO/weights/taide"
DATA="$REPO/data_regen/regenerated/sharegpt_gpt4_cleaned_test.json"

cd "$REPO/vllm/benchmarks"

python3 benchmark_serving.py \
  --base_url "http://localhost:${PORT}" \
  --backend openai-chat \
  --model "${MODEL}" \
  --tokenizer "${TOK}" \
  --dataset-name sharegpt \
  --dataset-path "${DATA}" \
  --endpoint /v1/chat/completions \
  --num-prompts 1

cd -
