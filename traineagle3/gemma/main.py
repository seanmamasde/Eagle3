from configs import EConfig          # your subclass of Gemma3TextConfig
from transformers import Gemma3Config
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
from transformers import AutoTokenizer, Gemma3TextConfig, Gemma3ForCausalLM
from modeling_gemma3_kv import Gemma3TextConfig as LocalGemmaConfig
import re
import json
import argparse
import deepspeed
from pprint import pprint
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
                    default='/home/seanma0627/src/eagle3/weights/gemma3')
parser.add_argument('--trainpath', type=str,
                    # default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_gpt4_cleaned_train.jsonl")
                    # default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_train_1pct_gemma3.jsonl")
                    default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_train_1pct.jsonl")
parser.add_argument('--testpath', type=str,
                    # default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_gpt4_cleaned_test.jsonl")
                    # default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_test_1pct_gemma3.jsonl")
                    default="/home/seanma0627/src/eagle3/data_regen/regenerated/sharegpt_test_1pct.jsonl")
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
    "num_epochs": 1,
    "num_workers": 2,
    # "max_len": 2048,
    "config_path": "config.json",

}

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.backends.cuda.matmul.allow_tf32 = True


def to_blocks(txt: str):
    return [{"type": "text", "text": txt}]


def build_dataset_rank(
    tokenizer, datapath
):
    # tokenizer.legacy_chat_template = True
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

    # def preprocess_function(examples):
    #     new_examples = {
    #         "attention_mask": [],
    #         "input_ids": [],
    #         "loss_mask": []
    #     }
    #     for i in range(len(examples['id'])):
    #         # messages = [
    #         #     {"role": "system",
    #         #      "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
    #         # ]
    #         # system_prompt = (
    #         #     "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    #         #     "as possible, while being safe. Your answers should not include any harmful, "
    #         #     "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    #         #     "that your responses are socially unbiased and positive in nature.\n\n"
    #         #     "If a question does not make any sense, or is not factually coherent, explain why "
    #         #     "instead of answering something not correct. If you don't know the answer to a "
    #         #     "question, please don't share false information."
    #         # )
    #         messages = []
    #         # convroles = ["user", "assistant"]
    #         source = examples['conversations'][i]
    #         roles = {"human": "user", "gpt": "model"}
    #         # print("-"*80)
    #         # pprint(source)
    #         # print("-"*80)
    #         # print(source[0]['from']) # user/model
    #         if not source:
    #             continue
    #         for j, sentence in enumerate(source):
    #             raw_role = sentence["from"]
    #             role = roles.get(raw_role)
    #             assert role is not None, f"unknown role {raw_role}"

    #             # if j == 0 and role == "user":
    #             #     merged = f"{system_prompt}\n\n{sentence['value']}"
    #             # else:
    #             #     merged = sentence["value"]
    #             merged = sentence["value"]
    #             messages.append({"role": role, "content": to_blocks(merged)})
    #         conversation = tokenizer.apply_chat_template(
    #             messages,
    #             tokenize=False,
    #             add_generation_prompt=False,
    #         )

    #         if not tokenizer.pad_token_id:
    #             tokenizer.pad_token_id = tokenizer.unk_token_id

    #         input_ids = tokenizer(
    #             conversation,
    #             return_tensors="pt",
    #             max_length=2048,
    #             add_special_tokens=False,
    #         ).input_ids[0]
    #         loss_mask = torch.ones_like(input_ids)
    #         # print(i)

    #         EOT = "<|end_of_turn|>\n"
    #         SEP_USER = f"{EOT}<|start_of_turn|>user\n"
    #         SEP_MODEL = f"{EOT}<|start_of_turn|>model\n"

    #         # after tokenizer.apply_chat_template(...)
    #         turns = conversation.split(SEP_USER)
    #         if len(turns) < 2:
    #             continue

    #         turns[1] = turns[0] + SEP_USER + turns[1]
    #         turns = turns[1:]

    #         for sent in source:
    #             messages.append(
    #                 {"role": role, "content": to_blocks(sent["value"])})
    #         # tpl = tokenizer.apply_chat_template(
    #         #     messages,
    #         #     tokenize=True,
    #         #     add_generation_prompt=False,
    #         #     return_dict=True,
    #         #     # return_assistant_tokens_mask=True,
    #         #     return_attention_mask=True,
    #         #     return_tensors="pt",
    #         # )
    #         # # ['input_ids', 'attention_mask', 'assistant_masks']
    #         # # print(tpl.keys())

    #         # input_ids = tpl["input_ids"][0]
    #         # loss_mask = tpl["assistant_tokens_mask"][0].to(torch.long)
    #         # # loss_mask = tpl["assistant_masks"][0].to(torch.long)
    #         # attention_mask = tpl["attention_mask"][0]

    #         # new_examples["input_ids"].append(input_ids[None, :])
    #         # new_examples["loss_mask"].append(loss_mask[None, :])
    #         # new_examples["attention_mask"].append(attention_mask[None, :])
    #         # continue

    #         cur_len = 1
    #         loss_mask[:cur_len] = 0
    #         for i, turn in enumerate(turns):
    #             if turn == "":
    #                 break
    #             turn_len = len(tokenizer(turn).input_ids)

    #             parts = turn.split(SEP_MODEL)
    #             if len(parts) != 2:
    #                 break
    #             parts[0] += SEP_MODEL
    #             # "-2" is hardcoded for the Gemma3 tokenizer to make the offset correct.
    #             instruction_len = len(tokenizer(parts[0]).input_ids) - 1

    #             # Ignore the user instructions
    #             if i == 0:
    #                 loss_mask[cur_len: cur_len + instruction_len - 2] = 0
    #             else:
    #                 loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
    #             cur_len += turn_len
    #             if i != 0:
    #                 cur_len += 3
    #             # cur_len+=2

    #             # if i != 0 and not tokenizer.legacy:
    #             #     # The legacy and non-legacy modes handle special tokens differently
    #             #     cur_len -= 1

    #         loss_mask[cur_len:] = 0
    #         attention_mask = torch.ones_like(loss_mask)

    #         # new_examples["conversation"].append(conversation)
    #         new_examples["input_ids"].append(input_ids[None, :])
    #         new_examples["loss_mask"].append(loss_mask[None, :])
    #         new_examples["attention_mask"].append(attention_mask[None, :])

    #     return new_examples

    def _normalize(value):
        # Acceptable types for chat_template
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            # Already block format?
            if all(isinstance(el, dict) and "type" in el and "text" in el for el in value):
                return value
            # List of strings → join them
            if all(isinstance(el, str) for el in value):
                return "\n".join(value)
        # Fallback: coerce to string
        return str(value)

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            messages = [
                # no system prompt in gemma3
                # {"role": "system",
                #  "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "model"]
            roles = {"human": "user", "gpt": "model"}
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
                    {"role": role, "content": _normalize(sentence["value"])}
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

            # between a user turn and the assistant's reply
            sep = "<end_of_turn>\n<start_of_turn>model\n"

            # between an assistant turn and the next user turn
            sep2 = "<end_of_turn>\n<start_of_turn>user\n"

            total_len = len(input_ids)

            # print(conversation)
            # print(type(conversation))

            turns = conversation.split(sep2)
            # print(turns)
            # print("=" * 80)
            # print(len(turns))

            if len(turns) >= 2:
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


tokenizer = AutoTokenizer.from_pretrained(args.basepath, use_fast=True)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)

# config = EConfig.from_pretrained(train_config["config_path"])
wrapper_cfg = Gemma3Config.from_pretrained(train_config["config_path"])
txt_cfg = EConfig(**wrapper_cfg.text_config.to_dict())
# print("hidden_size           :", config.hidden_size)
# print("num_attention_heads   :", config.num_attention_heads)
# print("head_dim              :", getattr(config, "head_dim", None))
"""
EConfig {
  "_sliding_window_pattern": 6,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": null,
  "bos_token_id": 2,
  "draft_vocab_size": null,
  "eos_token_id": 1,
  "final_logit_softcapping": null,
  "head_dim": 256,
  "hidden_act": "silu",
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 2304,
  "initializer_range": 0.02,
  "intermediate_size": 9216,
  "layer_types": [
    "sliding_attention"
  ],
  "max_position_embeddings": 131072,
  "model_type": "gemma3_text",
  "num_attention_heads": 8,
  "num_hidden_layers": 1,
  "num_key_value_heads": 4,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_local_base_freq": 10000.0,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": 1024,
  "tie_word_embeddings": true,
  "transformers_version": "4.53.2",
  "use_cache": true,
  "vocab_size": 262208
}
"""
txt_cfg.torch_dtype = torch.bfloat16
txt_cfg.hidden_size = 2560
txt_cfg.head_dim = 256
txt_cfg.num_attention_heads = 8
txt_cfg.num_key_value_heads = 4
txt_cfg.intermediate_size = 10240
txt_cfg.num_hidden_layers = 26
txt_cfg.layer_types = [
    "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention",
]
txt_cfg.vocab_size = 262_208
txt_cfg.sliding_window = 1024
txt_cfg._sliding_window_pattern = 6
txt_cfg.bos_token_id, txt_cfg.eos_token_id, txt_cfg.pad_token_id = 1, 2, 0
txt_cfg.attention_bias = False
txt_cfg.attention_dropout = 0.0
txt_cfg.hidden_act = "silu"
txt_cfg.rms_norm_eps = 1e-6
txt_cfg.initializer_range = 0.02
txt_cfg.tie_word_embeddings = True
txt_cfg.use_cache = True
txt_cfg.rope_scaling = None
txt_cfg.max_position_embeddings = 131_072
txt_cfg.draft_vocab_size = 12340
txt_cfg.attn_implementation = "eager" 
# print(txt_cfg)

model = Model(txt_cfg, path=args.basepath, load_emb=True, load_head=True)
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

    # model_engine.save_16bit_model(
    #     f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    # if epoch % 10 == 0:
    #     deepspeed.DeepSpeedEngine.save_checkpoint(
    #         model_engine, save_dir=f"{args.savedir}/state_{epoch}")
    # ---- save after every epoch (only one epoch in this run) ----
    ckpt_dir = os.path.join(args.savedir, f"epoch_{epoch}")
    model_engine.save_checkpoint(ckpt_dir)
    if global_rank == 0:
        from deepspeed.utils.zero_to_fp32 import \
            get_fp32_state_dict_from_zero_checkpoint
        fp32_state = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
        torch.save(fp32_state, os.path.join(ckpt_dir, "pytorch_model.bin"))
