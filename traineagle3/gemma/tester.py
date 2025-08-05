# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained(
#     '/home/seanma0627/src/eagle3/weights/gemma3', use_fast=True)
# msgs = [{"role": "user", "content": "Why is the sky blue?"}]
# conv = tok.apply_chat_template(
#     msgs, tokenize=False, add_generation_prompt=False)
# print(repr(conv))
"""
'<bos><start_of_turn>user\nWhy is the sky blue?<end_of_turn>\n'
"""

import torch
import os
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

torch.set_float32_matmul_precision('high')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
model_dir = "/home/seanma0627/src/eagle3/weights/gemma3"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_dir,
    device_map=device,
    torch_dtype=torch.bfloat16
).eval()


# def chat_round(msgs, user_msg, max_new=256):
#     """Append user_msg, generate assistant reply, return reply & updated msgs list."""
#     msgs.append({"role": "user", "content": user_msg})
#     input_ids = tokenizer.apply_chat_template(
#         msgs, add_generation_prompt=True, tokenize=True,
#         return_tensors="pt"
#     ).to(device)
#     cut = input_ids.shape[-1]          # tokens before generation prompt
#     with torch.inference_mode():
#         out = model.generate(
#             input_ids, max_new_tokens=max_new, do_sample=False)
#     reply = tokenizer.decode(out[0][cut:], skip_special_tokens=True).strip()
#     msgs.append({"role": "assistant", "content": reply})
#     return reply, msgs


# dialog = [{"role": "system", "content": "You are a concise, helpful assistant."}]

# _, dialog = chat_round(dialog, "Why is the sky blue?")

# follow_up = "Does Rayleigh scattering explain sunrise and sunset colours too?"
# _, dialog = chat_round(dialog, follow_up)

# for turn in dialog[1:]:  # skip system prompt
#     print(f"from: {turn['role']}, content: {turn['content']}\n")

def chat_verbose(dialogue, user_msg, max_new=256):
    # append the user message
    dialogue.append({"role": "user", "content": user_msg})

    # show the raw prompt string produced by gemma3's chat template
    prompt_str = tokenizer.apply_chat_template(
        dialogue,
        tokenize=False,              # we want the plain string first
        add_generation_prompt=False  # just format existing turns
    )
    print("\n=== RAW CHAT TEMPLATE STRING ===")
    print(repr(prompt_str))  # shows <bos>, <start_of_turn>, etc.

    # re-tokenize (so we can see ids)
    input_ids = tokenizer.apply_chat_template(
        dialogue,
        tokenize=True,
        add_generation_prompt=True,  # add the special “start model reply” tokens
        return_tensors="pt"
    ).to(device)
    print("\n=== TOKEN IDS ===")
    print(input_ids.tolist()[0])  # flatten tensor to Python list

    # decode back to text
    print("\n=== DECODED BACK FROM IDS ===")
    print(tokenizer.decode(input_ids[0]))

    # generate assistant reply
    cut = input_ids.shape[-1]  # index where generation starts
    with torch.inference_mode():
        out = model.generate(
            input_ids, max_new_tokens=max_new, do_sample=False, top_p=None, top_k=None)
    assistant_reply = tokenizer.decode(
        out[0][cut:], skip_special_tokens=True).strip()

    # Append reply to dialogue and show it
    dialogue.append({"role": "assistant", "content": assistant_reply})
    print("\n=== ASSISTANT REPLY ===")
    print(assistant_reply)

    return dialogue


dialog = []

# round 1
dialog = chat_verbose(dialog, "Why is the sky blue?")

print("=" * 80)

# follow up
follow_up = "Does Rayleigh scattering explain sunrise colours too?"
dialog = chat_verbose(dialog, follow_up)
