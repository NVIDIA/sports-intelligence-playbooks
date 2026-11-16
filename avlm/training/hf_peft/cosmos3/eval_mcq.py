# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a multiple-choice conversation JSONL with Cosmos3, with or without a LoRA adapter.

For every row, each `(X) option` line of the question is scored as the full assistant
answer given the clip's frames and the question (teacher-forced log-probability of the
answer tokens). The highest-scoring letter is the prediction; it is compared with the
row's `label`. No server is needed, so this also evaluates adapters that vLLM cannot
serve, such as Cosmos3-Edge ones.

Writes one JSON record per row to --out (label, prediction, per-letter probabilities)
and prints the accuracy at the end.

usage:
  python avlm/training/hf_peft/cosmos3/eval_mcq.py \
    --jsonl /path/to/test.jsonl --frames /path/to/frames --out /path/to/scores.jsonl \
    [--model nvidia/Cosmos3-Nano] [--adapter /path/to/output/adapter]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

OPTION_LINE = re.compile(r"^\((\w)\) (.+)$", flags=re.M)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", required=True, help="conversation JSONL with a `label` letter per row")
    parser.add_argument("--frames", required=True, help="frames root written by extract_frames.py")
    parser.add_argument("--out", required=True, help="output JSONL of per-row scores")
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano", help="Hub id or local checkpoint of the base")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory; omitted scores the base model")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def clip_frames(frames_root: str, video_path: str) -> list[Image.Image]:
    files = sorted(glob.glob(os.path.join(frames_root, os.path.splitext(video_path)[0], "f*.jpg")))
    if not files:
        raise FileNotFoundError(f"no frames for {video_path} under {frames_root}")
    return [Image.open(f).convert("RGB") for f in files]


@torch.no_grad()
def answer_logprob(model, processor, device: str, images: list[Image.Image], prompt: str, answer: str) -> float:
    prompt_len = processor(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1]
    full = prompt + answer + (processor.tokenizer.eos_token or "")
    encoded = {k: v.to(device) for k, v in processor(text=[full], images=images, return_tensors="pt").items()}
    logits = model(**encoded).logits[0, prompt_len - 1:-1].float()
    answer_ids = encoded["input_ids"][0, prompt_len:]
    return float(torch.log_softmax(logits, -1).gather(1, answer_ids[:, None]).sum())


def main() -> None:
    args = parse_args()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    with open(args.jsonl) as f:
        rows = [json.loads(line) for line in f]
    correct = 0
    with open(args.out, "w") as out:
        for index, row in enumerate(rows, 1):
            user = row["conversation"][0]
            video = next(i["path"] for i in user["content"] if i.get("type") == "video")
            question = next(i["text"] for i in user["content"] if i.get("type") == "text")
            options = dict(OPTION_LINE.findall(question))
            if not options:
                raise SystemExit(f"{row['id']}: no `(X) option` lines in the question")
            images = clip_frames(args.frames, video)
            messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": question}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            scores = {letter: answer_logprob(model, processor, args.device, images, prompt, f"({letter}) {text}") for letter, text in options.items()}
            normalized = torch.log_softmax(torch.tensor(list(scores.values())), 0).exp()
            probs = {letter: float(p) for letter, p in zip(scores, normalized)}
            prediction = max(probs, key=probs.get)
            hit = prediction == row["label"]
            correct += hit
            out.write(json.dumps({"id": row["id"], "label": row["label"], "prediction": prediction, "correct": hit, "probs": probs}) + "\n")
            out.flush()
            print(f"{index}/{len(rows)} {row['id']}: label {row['label']} prediction {prediction} p(label)={probs[row['label']]:.3f}", flush=True)
    print(f"accuracy {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
