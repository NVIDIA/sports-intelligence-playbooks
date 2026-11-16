# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA fine-tune of Cosmos3 on a video conversation dataset, feeding clips as frames.

Reads the playbook's conversation JSONL (user turn: one `video` item and one `text`
question; assistant turn: the answer), loads the JPEG frames extract_frames.py wrote
for each clip, and passes them to the chat template as that many `image` items followed
by the question. Training therefore sees exactly the frames a frame-based client sends
at inference, with no video decoder in the loop.

LoRA (rank r, alpha 2r, no dropout) is applied to the language model's attention and
MLP projections only; the vision tower and projector stay frozen. Optimizer: AdamW with
weight decay 0.05, linear warmup then cosine decay to --min-lr, gradient clipping at
1.0. The global batch runs as gradient-accumulation steps on one GPU. The loss covers
the assistant answer only.

Outputs: <out>/adapter (PEFT adapter, loadable with peft and, after
avlm/deployment/vLLM/utils/flatten_cosmos3_lora_adapter.py, with vLLM) and
<out>/train_log.jsonl (one JSON record per step: loss, grad_norm, lr, memory, elapsed,
and val_loss every --val-every steps).

usage:
  python avlm/training/hf_peft/cosmos3/train_lora.py \
    --train /path/to/train.jsonl --val /path/to/validation.jsonl \
    --frames /path/to/frames --out /path/to/output \
    [--model nvidia/Cosmos3-Nano --steps 150 --lr 2e-4 --rank 64 --warmup 10]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Attention projections plus the MLP, which is gated (gate/up/down) in Cosmos3-Nano and a plain fc1/fc2 pair in Cosmos3-Edge.
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|fc1|fc2)$"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano", help="Hub id or local checkpoint of the Cosmos3 base")
    parser.add_argument("--train", required=True, help="training conversation JSONL")
    parser.add_argument("--val", required=True, help="validation conversation JSONL (scored whole every --val-every steps)")
    parser.add_argument("--frames", required=True, help="frames root written by extract_frames.py")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--steps", type=int, default=150, help="optimizer steps")
    parser.add_argument("--global-batch", type=int, default=8, help="samples per optimizer step (gradient accumulation)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=10, help="linear warmup steps")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default 2 * rank)")
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="build one sample, run one forward pass, exit")
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def clip_frames(frames_root: str, video_path: str) -> list[Image.Image]:
    files = sorted(glob.glob(os.path.join(frames_root, os.path.splitext(video_path)[0], "f*.jpg")))
    if not files:
        raise FileNotFoundError(f"no frames for {video_path} under {frames_root}")
    return [Image.open(f).convert("RGB") for f in files]


def build_example(processor, frames_root: str, row: dict) -> dict:
    user, assistant = row["conversation"][0], row["conversation"][1]
    video = next(i["path"] for i in user["content"] if i.get("type") == "video")
    question = next(i["text"] for i in user["content"] if i.get("type") == "text")
    answer = assistant["content"][0]["text"]
    images = clip_frames(frames_root, video)
    messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prompt + answer + (processor.tokenizer.eos_token or "")
    prompt_len = processor(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1]
    encoded = processor(text=[full], images=images, return_tensors="pt")
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_len] = IGNORE_INDEX
    encoded["labels"] = labels
    return encoded


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device)
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model)

    train_rows, val_rows = load_rows(args.train), load_rows(args.val)
    probe = build_example(processor, args.frames, train_rows[0])
    print(f"sample: {probe['input_ids'].shape[1]} tokens, {int((probe['labels'] != IGNORE_INDEX).sum())} answer tokens", flush=True)

    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha or 2 * args.rank, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    adapted = sum(1 for name, _ in model.named_modules() if name.endswith("lora_A"))
    if adapted == 0:
        raise SystemExit("no module matched the LoRA target pattern")
    print(f"{adapted} adapted modules", flush=True)

    def forward(encoded: dict) -> torch.Tensor:
        return model(**{k: v.to(args.device) for k, v in encoded.items()}).loss

    if args.smoke:
        with torch.no_grad():
            print(f"smoke forward loss {float(forward(probe)):.4f}", flush=True)
        return

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    except ValueError as err:
        print(f"gradient checkpointing unavailable: {err}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.999))

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            total = sum(forward(build_example(processor, args.frames, row)).item() for row in val_rows)
        model.train()
        return total / len(val_rows)

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    generator = torch.Generator().manual_seed(args.seed)
    order: list[int] = []
    started = time.time()
    model.train()
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        losses = []
        for _ in range(args.global_batch):
            if not order:
                order = torch.randperm(len(train_rows), generator=generator).tolist()
            loss = forward(build_example(processor, args.frames, train_rows[order.pop()])) / args.global_batch
            loss.backward()
            losses.append(loss.item() * args.global_batch)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "loss": sum(losses) / len(losses),
            "grad_norm": float(grad_norm),
            "lr": lr_at(step, args),
            "mem_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else 0.0,
            "elapsed_s": time.time() - started,
        }
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            record["val_loss"] = validate()
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    model.save_pretrained(os.path.join(args.out, "adapter"))
    print(f"adapter written to {os.path.join(args.out, 'adapter')}", flush=True)


if __name__ == "__main__":
    main()
