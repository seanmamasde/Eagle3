from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import nn, optim
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datasets import load_dataset
from configs import EConfig
from cnets import Model
from accelerate.utils import set_seed
from cnets import padding
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_llama_kv import LlamaConfig
import re
import json
import argparse
import deepspeed
# from deepspeed import zero

##### HPC SETTINGS #####
# deepspeed.init_distributed()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
# torch.set_num_threads(4)
# torch.set_num_interop_threads(4)
# os.environ["DATASETS_DISABLE_MP"] = "1"
# os.environ["HF_DATASETS_NUM_PROCS"] = "1"
# os.environ["KEEP_IN_MEMORY"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["DS_SKIP_TRITON_AUTOTUNE"] = "1"
########################

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str,
                    default='/home/seanma0627/src/eagle3/weights/taide')
parser.add_argument('--trainpath', type=str,
                    default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_gpt4_cleaned_train.jsonl")
                    # default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_train_1pct.jsonl")
parser.add_argument('--testpath', type=str,
                    default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_gpt4_cleaned_test.jsonl")
                    # default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_test_1pct.jsonl")
parser.add_argument('--savedir', type=str, default='saved_models')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 40,
    "num_workers": 2,
    # "max_len": 2048,
    "config_path": "config.json",
}

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.backends.cuda.matmul.allow_tf32 = True


def build_dataset_rank(
        tokenizer, datapath
):
    ds = load_dataset(
        'json',
        data_files=datapath,
        # streaming=True
        keep_in_memory=True,
    )
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 2

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]
            if not source:
                continue
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                # if sentence["from"]=="gpt":
                #     sentence["value"]=" "+sentence["value"]
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            total_len = len(input_ids)

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3
                # cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            # new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    # import datasets
    # datasets.config.NUM_PROC = 1
    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        # num_proc=1,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    # ds1.with_format(type="torch")
    return ds1


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(
            item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, path=args.basepath, load_emb=True, load_head=True)
# with zero.Init(config_dict_or_path=ds_config, dtype=torch.float16):
#     model = Model(config, path=args.basepath, load_emb=True, load_head=True)
model.scandata(args.trainpath, args.basepath)


criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    import wandb

    wandb.login(key="")
    wandb.init(project="taide-eagle3", entity="seanmamasde",
               config=ds_config, mode="offline")

os.makedirs(args.savedir, exist_ok=True)

print(
    f"Global rank: {global_rank}, Local rank: {rank}, World size: {world_size}")
print("========= LOADING DATASET =========")

sampler = DistributedSampler(
    testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4,
                         pin_memory=True,
                         collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(
    traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)


for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(train_loader)):

        model.zero_grad()

        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(
                                                   rank),
                                               loss_mask=data["loss_mask"],
                                               )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i]
                    for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)

        model_engine.step()

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb.log(logdict)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()]
                         for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochacc_{i}": acc_i})
            print(
                f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochploss_{i}": loss_i})
            print(
                f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                   attention_mask=data["attention_mask"].to(
                                                       rank),
                                                   loss_mask=data["loss_mask"],
                                                   )
            epoch_acces = [epoch_acces[i] + [acces[i]]
                           for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()]
                             for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochacc_{i}": acc_i})
            print(
                f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochploss_{i}": loss_i})
            print(
                f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    model_engine.save_16bit_model(
        f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(
            model_engine, save_dir=f"{args.savedir}/state_{epoch}")
