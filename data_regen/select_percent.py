import random
import pathlib

SRC = pathlib.Path("regenerated/sharegpt_gpt4_cleaned_train.jsonl")
DST = pathlib.Path("regenerated/sharegpt_train_1pct.jsonl")
P = 0.01
SEED = 42

random.seed(SEED)

with SRC.open() as fin, DST.open("w") as fout:
    for line in fin:
        if random.random() < P:
            fout.write(line)
