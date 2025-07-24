from __future__ import annotations
import time
import statistics
import argparse
from pathlib import Path
from typing import List

import torch
import os
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "1"

BASE_MODEL_DIR = Path("/home/seanma0627/src/eagle3/weights/taide")
PROMPT = "怎麼學習一個新的語言？"
ROUNDS = 10
TRIM_RATIO = 0.2
DTYPE = torch.float16
MAX_MODEL_LEN = 2048
TP_SIZE = 1
SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=256,
)
TOTAL_TOKENS = len(PROMPT.split()) + SAMPLING.max_tokens


def _build_llm(enforce_eager: bool,
               speculative: bool = False) -> LLM:
    return LLM(
        model=str(BASE_MODEL_DIR),
        dtype=DTYPE,
        tensor_parallel_size=TP_SIZE,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_MODEL_LEN,
        # kv_cache_dtype="fp8_e4m3",
        # disable_async_output_proc=False,
        # compilation_config={c
        #     "level": 3,
        #     "full_cuda_graph": True,
        #     "splitting_ops": [],
        #     "use_cudagraph": True,
        #     "cudagraph_num_of_warmups": 2,
        #     "inductor_compile_config": {
        #         "enable_auto_functionalized_v2": True
        #     }
        # }
    )


def _run_rounds(llm):
    """Run ROUNDS iterations and collect per-round latency + hit-rate."""
    latencies = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        outputs = llm.generate(PROMPT, SAMPLING)
        latencies.append(time.perf_counter() - t0)
    return latencies


def _trimmed_mean(values: List[float]) -> float:
    k = int(len(values) * TRIM_RATIO)
    return statistics.mean(sorted(values)[k:-k] or values)


def bench(enforce_eager: bool):
    plain = _build_llm(enforce_eager, speculative=False)
    base_lat = _run_rounds(plain)
    base_avg = _trimmed_mean(base_lat)
    base_tps = TOTAL_TOKENS / base_avg
    print(f"Baseline latency  : {base_avg:.3f}s  (trimmed mean over {ROUNDS})")
    print(f"Tok/s             : {base_tps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true",
                        help="Disable CUDA Graphs for both runs")
    args = parser.parse_args()

    bench(enforce_eager=args.eager)
