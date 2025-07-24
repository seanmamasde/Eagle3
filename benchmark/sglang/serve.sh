#!/usr/bin/env bash
# -e/--eagle to launch the eagle3 model
# -h for help
set -euo pipefail

PORT=8000
MODEL_NAME=taide-baseline
EXTRA_ARGS=()

while getopts ":eh-:" opt; do
  case "$opt" in
    e)  PORT=8001
        MODEL_NAME=taide-eagle3
        EXTRA_ARGS+=( \
          --speculative-algorithm EAGLE3 \
          --speculative-draft-model-path '/models/taide-eagle3-sglang' \
          --speculative-num-steps 3 \
          --speculative-eagle-topk 24 \
          --speculative-num-draft-tokens 128 \
        ) ;;
    h|\?) echo "Usage: $0 [-e|--eagle]" ; exit 0 ;;
    -)  case "$OPTARG" in
          eagle)  PORT=8001
                  MODEL_NAME=taide-eagle3
                  EXTRA_ARGS=( \
                    --speculative-algorithm EAGLE3 \
                    --speculative-draft-model-path /models/taide-eagle3-sglang \
                    --speculative-num-steps 3 \
                    --speculative-eagle-topk 24 \
                    --speculative-num-draft-tokens 128 \
                  ) ;;
          help)  echo "Usage: $0 [-e|--eagle]" ; exit 0 ;;
          *)     echo "Unknown option --$OPTARG" >&2; exit 1 ;;
        esac ;;
  esac
done
shift $((OPTIND-1))

singularity exec --nv \
  --bind /home/seanma0627/src/eagle3/weights:/models \
  sglang.sif \
  python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model /models/taide \
    --dtype float16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --max-total-tokens 131072 \
    --cuda-graph-max-bs 32 \
    "${EXTRA_ARGS[@]}"
