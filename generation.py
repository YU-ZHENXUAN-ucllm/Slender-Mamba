import argparse
import time
import json

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PretrainedConfig, PreTrainedModel
import math
from functools import partial
import json
import os
import copy

from collections import namedtuple
from slender_mamba.mixer_model import MambaLMHeadModel
from slender_mamba.ops.Bitembedding import replace_embeddings_in_pytorch_model
from slender_mamba.ops.Bitembedding import replace_linears_in_pytorch_model
from typing import Tuple

# Code from https://github.com/state-spaces/mamba/tree/main

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="slender-mamba")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16


mamba2_config_dict ={
    "d_model": 768,
    "d_intermediate": 0,
    "n_layer":24,
    "vocab_size": 50288,
    "ssm_cfg": {
        "layer": "Mamba2"
    },
    "attn_layer_idx": [],
    "attn_cfg": {},
    "rms_norm": True,
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": True
}

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("slender-mamba")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    config = PretrainedConfig(**mamba2_config_dict)
    model = MambaLMHeadModel(config)

    print(model)
    replace_linears_in_pytorch_model(model)
    print("After replacement:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')


    replace_embeddings_in_pytorch_model(model)

    print("After embedding replacement:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')

    checkpoint_path = "/work/ucllm/mamba/evals/lambda/m130m_bitnet158_embedding158/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(state_dict)
    # Initialize dictionary to store parameters of specific parts
    specific_parts_params = {
        'mixer.in_proj': 0,
        'mixer.conv1d': 0,
        'mixer.out_proj': 0,
        'mixer.norm': 0,
        'lm_head': 0,
        'embedding': 0
    }

    # Loop over all modules and sum parameters for specified parts
    for name, module in model.named_modules():
        for part in specific_parts_params:
            if part in name:
                specific_parts_params[part] += sum(p.numel() for p in module.parameters())

    # Correcting the total_params to include only trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')

    # Prevent double counting if lm_head and embedding share parameters
    if 'lm_head' in specific_parts_params and 'embedding' in specific_parts_params:
        # This correction assumes lm_head and embedding are the only possible shared parameters
        # You need to adjust based on your model's specific architecture
        combined_params = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
        specific_parts_params['lm_head'] = combined_params
        specific_parts_params['embedding'] = combined_params

    # Calculate the sum of parameters for all specified parts
    specific_parts_sum = sum(specific_parts_params.values())

    # Calculate the percentage of total parameters
    specific_parts_percentage = specific_parts_sum / total_params * 100

    print("Specific parts parameters and their percentages of total:")
    for part, params in specific_parts_params.items():
        print(f"{part}: {params} parameters, {params / total_params:.2%} of total")
    print(f"Total of specified parts: {specific_parts_sum} parameters, {specific_parts_percentage:.2%} of total")
    model.to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen
torch.cuda.reset_peak_memory_stats()
if is_mamba:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
out = fn()
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")
max_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 将字节转换为兆字节
print(f"Max memory used: {max_memory_used} MB")