import json
import random
import argparse
from pathlib import Path

INPUT_FILE = 'regenerated/sharegpt_gpt4_full_original.jsonl'
TRAIN_FILE = 'regenerated/sharegpt_gpt4_cleaned_train.jsonl'
TEST_FILE = 'regenerated/sharegpt_gpt4_cleaned_test.jsonl'
OUTPUT_FILE = 'regenerated/sharegpt_gpt4_cleaned_full.jsonl'

TRAIN_RATIO = 0.95
RANDOM_SEED = 42


def load_and_clean(input_path):
    """Read JSONL, drop last human turn if present, and drop trailing empty GPT pairs."""
    cleaned = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get('conversations', [])

            if conv and conv[-1].get('from') == 'human':
                conv = conv[:-1]

            if (
                len(conv) >= 2
                and conv[-1].get('from') == 'gpt'
                and not conv[-1].get('value')
                and conv[-2].get('from') == 'human'
            ):
                conv = conv[:-2]

            obj['conversations'] = conv
            cleaned.append(obj)
    return cleaned


def split_data(data, train_ratio, seed=None):
    """Shuffle data and split into train/test lists."""
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def write_jsonl(data, output_path):
    """Write list of objects to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for obj in data:
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Clean ShareGPT and optionally split or emit full dataset")
    parser.add_argument('-f', '--full', action='store_true',
                        help="Write a single cleaned JSONL file instead of train/test split.")
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_FILE,
                        help=f"Filename for full cleaned output (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    # 1. Load & clean
    cleaned = load_and_clean(INPUT_FILE)
    print(f"Total records after cleaning: {len(cleaned)}")

    if args.full:
        # 2a. Write the entire cleaned dataset
        write_jsonl(cleaned, args.output)
        print(f"Wrote full cleaned dataset to {args.output}")
    else:
        # 2b. Split
        train_set, test_set = split_data(
            cleaned, TRAIN_RATIO, seed=RANDOM_SEED)
        print(f"Train set: {len(train_set)} records")
        print(f"Test set : {len(test_set)} records")

        # 3. Write out
        write_jsonl(train_set, TRAIN_FILE)
        write_jsonl(test_set, TEST_FILE)
        print("Wrote train and test JSONL files.")


if __name__ == '__main__':
    main()
