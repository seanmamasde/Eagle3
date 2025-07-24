from __future__ import annotations
import time
import statistics
import argparse
import os
import torch
from pathlib import Path
from typing import List
import sglang as sgl

BASE_MODEL_DIR = Path("/home/seanma0627/src/eagle3/weights/taide")
EAGLE_MODEL_DIR = Path("/home/seanma0627/src/eagle3/weights/taide-eagle3-sglang")
# PROMPT = "How to learn a new language?"
PROMPT = "怎麼學習一個新的語言？"
ROUNDS = 10
BATCH_SIZE = 1
MAX_MODEL_LEN = 2048
SPEC_TOKEN = 3


def build_engine(cuda_graph: bool) -> sgl.Engine:
    return sgl.Engine(
        model_path=str(BASE_MODEL_DIR),
        dtype=torch.float16,
        # kv_cache_dtype="fp8_e4m3",
        disable_cuda_graph=not cuda_graph,
        attention_backend="flashinfer",
        mem_fraction_static=0.8,
        max_running_requests=32,
        max_total_tokens=131072,

        ### SD ###
        # enable_torch_compile="draft", # optimizes forever
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=str(EAGLE_MODEL_DIR),
        speculative_num_steps=SPEC_TOKEN,
        speculative_eagle_topk=24,
        speculative_num_draft_tokens=128,
    )


def run(engine: sgl.Engine) -> tuple[float, float]:
    latencies = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        outs = engine.generate(
            [PROMPT]*BATCH_SIZE, dict(temperature=0.0, top_p=1.0,
                                      #   max_tokens=256
                                      ))
        latencies.append(time.perf_counter() - t0)

        # acc = draft = 0
        # for o in outs:
        # acc += o.metrics.spec_token_acceptance_counts[1:].sum()
        # draft += SPEC_TOKEN

    return _trim(latencies)


def _trim(values: List[float]) -> float:
    k = int(len(values) * 0.2)
    return statistics.mean(sorted(values)[k:-k] or values)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eager", action="store_true", help="disable CUDA graphs")
    args = p.parse_args()

    eng = build_engine(cuda_graph=not args.eager)
    latency = run(eng)
    tps = (len(PROMPT.split()) + 256) * BATCH_SIZE / latency

    print(f"EAGLE-3 latency : {latency:.3f}s  (trimmed mean over {ROUNDS})")
    print(f"Throughput      : {tps:.1f} tok/s")
