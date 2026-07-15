#!/usr/bin/env python3
# -*- coding: ascii -*-
"""C5-RPB Downstream Task Evaluation v1
Compare standard vs C5-RPB attention on concrete downstream tasks.

Tasks:
  1. Needle-in-Haystack (position bias test)
  2. Long-range dependency (connect distant facts)
  3. Multi-hop reasoning (chain 2 facts)
  4. Code completion (structured output)
  5. Math word problem (symbolic reasoning)

Usage:
  python c5_downstream_eval.py --model_path PATH --device cpu
  python c5_downstream_eval.py --model_path PATH --device cpu --amp 0.5
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import argparse
import math
import json
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# C5-RPB
# ============================================================================

def make_c5_rpb_tensor(n_heads, seq_len, amplitude=1.0, phi_shift=0.0, device="cpu", dtype=torch.float32):
    B = torch.zeros(n_heads, seq_len, seq_len, device=device, dtype=dtype)
    for h in range(n_heads):
        phase_h = 2 * math.pi * (h % 5) / 5 + phi_shift
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos = (i - j) / max(seq_len, 1)
                phase = phase_h + math.pi * rel_pos
                B[h, i, j] = amplitude * math.cos(phase)
    return B

# ============================================================================
# Manual forward with C5-RPB (generates text token by token)
# ============================================================================

def generate_with_rpb(model, input_ids, max_new_tokens, rpb_amp, device,
                      n_layers, n_heads, n_kv_heads, hidden_size, head_dim,
                      phi_shift=0.0, do_sample=False, temperature=1.0):
    """Generate tokens with C5-RPB injected into every layer."""
    model.eval()
    generated = input_ids.clone()
    bsz = input_ids.shape[0]

    with torch.no_grad():
        for step in range(max_new_tokens):
            seq_len = generated.shape[1]
            # Build RPB for current sequence length
            rpb = make_c5_rpb_tensor(n_heads, seq_len + 2, amplitude=rpb_amp,
                                     phi_shift=phi_shift, device=device, dtype=torch.float32)

            # Forward pass manually
            hidden = model.model.embed_tokens(generated)
            rotary_emb = None
            if hasattr(model.model, "rotary_emb"):
                rotary_emb = model.model.rotary_emb
            elif hasattr(model.model.layers[0].self_attn, "rotary_emb"):
                rotary_emb = model.model.layers[0].self_attn.rotary_emb

            for l_idx in range(n_layers):
                layer = model.model.layers[l_idx]
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                attn = layer.self_attn

                q = attn.q_proj(hidden).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
                k = attn.k_proj(hidden).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                v = attn.v_proj(hidden).view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)

                if n_kv_heads < n_heads:
                    n_rep = n_heads // n_kv_heads
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len, head_dim)
                    v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len, head_dim)

                if rotary_emb is not None:
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    try:
                        cos_r, sin_r = rotary_emb(v, position_ids, seq_len=seq_len)
                    except TypeError:
                        try:
                            cos_r, sin_r = rotary_emb(v, position_ids)
                        except Exception:
                            cos_r, sin_r = rotary_emb(position_ids, seq_len=seq_len)

                    def rotate_half(x):
                        x1 = x[..., :x.shape[-1]//2]
                        x2 = x[..., x.shape[-1]//2:]
                        return torch.cat((-x2, x1), dim=-1)

                    q = q * cos_r + rotate_half(q) * sin_r
                    k = k * cos_r + rotate_half(k) * sin_r

                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                scores = scores + rpb[:, :seq_len, :seq_len].unsqueeze(0)

                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden.dtype),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)
                scores = scores + causal_mask

                attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, hidden_size)
                attn_output = attn.o_proj(attn_output)

                hidden = residual + attn_output
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden

            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)

            next_logits = logits[:, -1, :]
            if do_sample:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

    return generated

# ============================================================================
# Task definitions
# ============================================================================

# Task 1: Needle-in-Haystack
# Insert a specific fact at different positions in a long passage, then ask about it.
def build_needle_haystack(context_fill, needle, question, needle_pos_ratio, tokenizer, max_context=800):
    """Build a needle-in-haystack prompt."""
    # Split context into two parts
    total_tokens = max_context
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    fill_tokens = tokenizer.encode(context_fill, add_special_tokens=False)

    # Repeat fill to reach target length
    while len(fill_tokens) < total_tokens:
        fill_tokens = fill_tokens + fill_tokens
    fill_tokens = fill_tokens[:total_tokens]

    # Insert needle at position
    insert_pos = int(len(fill_tokens) * needle_pos_ratio)
    combined = fill_tokens[:insert_pos] + needle_tokens + fill_tokens[insert_pos:]
    combined = combined[:max_context]

    # Add question at end
    full_input = combined + question_tokens

    return torch.tensor([full_input], dtype=torch.long)

NEEDLE_HAYSTACK_TASKS = [
    {
        "fill": "The history of computing spans from ancient abacuses to modern quantum computers. Early mechanical calculators like the Pascaline and Difference Engine laid the groundwork for automatic computation. The invention of the transistor at Bell Labs in nineteen forty seven revolutionized electronics. Programming languages evolved from assembly to high level languages like Fortran and COBOL. The internet emerged from ARPANET in the late nineteen sixties. Personal computers became mainstream in the nineteen eighties with the IBM PC and Apple Macintosh. Mobile computing transformed society in the two thousands with smartphones. Cloud computing shifted infrastructure from local to distributed systems. Machine learning and neural networks experienced a renaissance with deep learning breakthroughs. Natural language processing advanced rapidly with transformer architectures and large language models. ",
        "needle": "The secret access code for the vault is DELTA-7749. ",
        "question": "What is the secret access code for the vault? Answer briefly:",
        "positions": [0.1, 0.5, 0.9],  # start, middle, end
        "answer_keyword": "DELTA-7749",
    },
    {
        "fill": "Geography encompasses the study of Earths physical features climate patterns and human settlements. The Amazon rainforest produces approximately twenty percent of the worlds oxygen. Mount Everest reaches an elevation of eight thousand eight hundred forty eight meters above sea level. The Sahara Desert covers about nine point two million square kilometers across North Africa. The Pacific Ocean contains more than half of the worlds free water. The Mariana Trench reaches depths of nearly eleven thousand meters. The Nile River stretches approximately six thousand six hundred fifty kilometers. Lake Baikal in Siberia holds roughly twenty percent of the worlds surface freshwater. The Antarctic ice sheet contains about sixty one percent of all fresh water on Earth. ",
        "needle": "Professor Zhangs office is located in Room 308 on the third floor of Building C. ",
        "question": "Where is Professor Zhangs office? Answer briefly:",
        "positions": [0.1, 0.5, 0.9],
        "answer_keyword": "Room 308",
    },
    {
        "fill": "Biology reveals the complexity of life from molecular interactions to ecosystem dynamics. DNA was identified as the carrier of genetic information by Avery and colleagues in nineteen forty four. The double helix structure was elucidated by Watson and Crick in nineteen fifty three. The genetic code was cracked in the nineteen sixties. Recombinant DNA technology emerged in the nineteen seventies enabling genetic engineering. The Human Genome Project completed sequencing in two thousand three. CRISPR gene editing was adapted from bacterial immune systems in two thousand twelve. Synthetic biology aims to design and construct new biological parts and systems. Proteomics studies the full set of proteins expressed by a genome. Metagenomics analyzes genetic material recovered directly from environmental samples. ",
        "needle": "The password for the laboratory computer system is NEUTRON-42X. ",
        "question": "What is the password for the laboratory computer system? Answer briefly:",
        "positions": [0.1, 0.5, 0.9],
        "answer_keyword": "NEUTRON-42X",
    },
]

# Task 2: Long-range dependency
LONG_RANGE_TASKS = [
    {
        "context": "Alice has a red car. Bob has a blue bicycle. Carol has a green motorcycle. Dave has a yellow scooter. Eve has a purple truck. Frank has a orange van. Grace has a pink kayak. Henry has a brown canoe. Irene has a white sailboat. Jack has a black helicopter. After everyone traveled to the city center using their respective vehicles, they met at the restaurant. Alice parked her vehicle in the garage. Bob chained his vehicle to a post. Carol revved her vehicle loudly. Dave folded his vehicle and carried it inside. Eve backed her vehicle into a loading dock. Frank squeezed his vehicle into a tight spot. Grace carried her vehicle to the river. Henry lifted his vehicle onto the roof. Irene secured her vehicle at the marina. Jack landed his vehicle on the rooftop pad. At the restaurant, the waiter asked each person how they arrived.",
        "question": "What color is the vehicle that Carol rode to the restaurant?",
        "answer": "green",
    },
    {
        "context": "The first number in the sequence is seventeen. The second number is forty two. The third number is eighty nine. The fourth number is one hundred fifty six. The fifth number is two hundred thirty one. The sixth number is three hundred fourteen. The seventh number is four hundred five. Between the first and second numbers, the difference is twenty five. Between the second and third, the difference is forty seven. Between the third and fourth, the difference is sixty seven. Between the fourth and fifth, the difference is seventy five. Between the fifth and sixth, the difference is eighty three. Between the sixth and seventh, the difference is ninety one. The pattern of differences is increasing by varying amounts.",
        "question": "What was the fourth number in the sequence?",
        "answer": "156",
    },
]

# Task 3: Multi-hop reasoning
MULTIHOP_TASKS = [
    {
        "prompt": "Fact 1: The capital of France is Paris. Fact 2: The Eiffel Tower is located in the capital of France. Question: In which city is the Eiffel Tower located? Answer:",
        "answer": "Paris",
    },
    {
        "prompt": "Fact 1: Water boils at 100 degrees Celsius at sea level. Fact 2: The boiling point decreases by approximately 1 degree for every 285 meters of elevation gain. Fact 3: Denver is at an elevation of approximately 1600 meters above sea level. Question: Approximately what temperature does water boil in Denver? Answer:",
        "answer": "94",
    },
    {
        "prompt": "Fact 1: X is the father of Y. Fact 2: Y is the mother of Z. Question: What is the relationship of X to Z? Answer:",
        "answer": "grandfather",
    },
]

# Task 4: Code completion
CODE_TASKS = [
    {
        "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
        "answer": "fibonacci(n-2)",
    },
    {
        "prompt": "def is_palindrome(s):\n    return s == s[",
        "answer": "::-1]",
    },
    {
        "prompt": "def find_max(lst):\n    max_val = lst[0]\n    for val in lst:\n        if val > max_val:\n            max_val =",
        "answer": "val",
    },
]

# Task 5: Math word problems
MATH_TASKS = [
    {
        "prompt": "A store sells apples for $2 each and oranges for $3 each. If John buys 4 apples and 3 oranges, how much does he spend in total? Answer:",
        "answer": "17",
    },
    {
        "prompt": "A train travels at 60 km/h for 2 hours, then at 80 km/h for 1.5 hours. What is the total distance traveled in km? Answer:",
        "answer": "240",
    },
    {
        "prompt": "If a rectangle has a length of 12 meters and a width of 8 meters, what is its area in square meters? Answer:",
        "answer": "96",
    },
]

# ============================================================================
# Evaluation helpers
# ============================================================================

def check_answer(generated_text, answer, task_type="exact"):
    """Check if the generated text contains the correct answer."""
    gen_lower = generated_text.lower().strip()
    ans_lower = answer.lower().strip()

    if task_type == "keyword":
        return ans_lower in gen_lower
    elif task_type == "contains":
        # For numeric answers, check if the number appears
        return ans_lower in gen_lower
    else:
        # Exact match at the end
        return ans_lower in gen_lower[-100:]

def extract_answer_text(tokenizer, generated_ids, prompt_ids_len, max_decode=150):
    """Extract the generated answer part (after the prompt)."""
    answer_ids = generated_ids[0, prompt_ids_len:]
    text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return text[:max_decode]

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default=r"C:\Users\WANGJUN\d10\ms_cache\models\Qwen--Qwen2.5-1.5B\snapshots\master")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--amp", type=float, default=0.5, help="C5-RPB amplitude (default: 0.5 free-lunch)")
    parser.add_argument("--max_context", type=int, default=600, help="Max context for needle-haystack")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Max tokens to generate per answer")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_path = os.path.normpath(os.path.abspath(args.model_path))
    model_name = os.path.basename(os.path.dirname(model_path)).replace("--", "/")

    print("=" * 70, flush=True)
    print("C5-RPB Downstream Evaluation v1", flush=True)
    print(f"Model: {model_name} | amp: {args.amp}", flush=True)
    print("=" * 70, flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager", local_files_only=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    print(f"  {model_name}: {n_layers}L, {n_heads}H, {n_kv_heads}KV, {hidden_size}d, {n_params:.1f}B", flush=True)

    results = {"standard": {}, "c5rpb": {}}
    all_results = []

    # ========================================================================
    # Task 1: Needle-in-Haystack
    # ========================================================================
    print("\n" + "=" * 70, flush=True)
    print("TASK 1: Needle-in-Haystack (Position Bias)", flush=True)
    print("=" * 70, flush=True)

    nih_std_correct = 0
    nih_c5_correct = 0
    nih_total = 0

    for ti, task in enumerate(NEEDLE_HAYSTACK_TASKS):
        for pos in task["positions"]:
            nih_total += 1
            input_ids = build_needle_haystack(
                task["fill"], task["needle"], task["question"],
                pos, tokenizer, max_context=args.max_context
            ).to(args.device)

            pos_label = f"p={pos:.1f}"
            print(f"\n  [{ti+1}/{len(NEEDLE_HAYSTACK_TASKS)}] pos={pos}...", flush=True)

            # Standard
            with torch.no_grad():
                std_gen = model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            std_text = extract_answer_text(tokenizer, std_gen, input_ids.shape[1])
            std_ok = check_answer(std_text, task["answer_keyword"], "keyword")
            if std_ok:
                nih_std_correct += 1

            # C5-RPB
            c5_gen = generate_with_rpb(
                model, input_ids, args.max_new_tokens, args.amp, args.device,
                n_layers, n_heads, n_kv_heads, hidden_size, head_dim
            )
            c5_text = extract_answer_text(tokenizer, c5_gen, input_ids.shape[1])
            c5_ok = check_answer(c5_text, task["answer_keyword"], "keyword")
            if c5_ok:
                nih_c5_correct += 1

            marker_s = "Y" if std_ok else "N"
            marker_c = "Y" if c5_ok else "N"
            print(f"    Std[{marker_s}]: {std_text[:80]}", flush=True)
            print(f"    C5 [{marker_c}]: {c5_text[:80]}", flush=True)

            all_results.append({
                "task": "needle_haystack", "subtask": ti, "position": pos,
                "std_correct": std_ok, "c5_correct": c5_ok,
                "std_text": std_text[:100], "c5_text": c5_text[:100],
            })

    nih_std_rate = nih_std_correct / max(nih_total, 1)
    nih_c5_rate = nih_c5_correct / max(nih_total, 1)
    print(f"\n  Needle-in-Haystack: Std={nih_std_correct}/{nih_total} ({nih_std_rate:.1%}), C5={nih_c5_correct}/{nih_total} ({nih_c5_rate:.1%})", flush=True)

    # ========================================================================
    # Task 2: Long-range dependency
    # ========================================================================
    print("\n" + "=" * 70, flush=True)
    print("TASK 2: Long-range Dependency", flush=True)
    print("=" * 70, flush=True)

    lr_std_correct = 0
    lr_c5_correct = 0
    lr_total = len(LONG_RANGE_TASKS)

    for ti, task in enumerate(LONG_RANGE_TASKS):
        prompt = task["context"] + "\n" + task["question"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)

        print(f"\n  [{ti+1}/{lr_total}] Question: {task['question'][:50]}...", flush=True)

        # Standard
        with torch.no_grad():
            std_gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        std_text = extract_answer_text(tokenizer, std_gen, input_ids.shape[1])
        std_ok = check_answer(std_text, task["answer"], "contains")
        if std_ok:
            lr_std_correct += 1

        # C5-RPB
        c5_gen = generate_with_rpb(
            model, input_ids, args.max_new_tokens, args.amp, args.device,
            n_layers, n_heads, n_kv_heads, hidden_size, head_dim
        )
        c5_text = extract_answer_text(tokenizer, c5_gen, input_ids.shape[1])
        c5_ok = check_answer(c5_text, task["answer"], "contains")
        if c5_ok:
            lr_c5_correct += 1

        marker_s = "Y" if std_ok else "N"
        marker_c = "Y" if c5_ok else "N"
        print(f"    Std[{marker_s}]: {std_text[:80]}", flush=True)
        print(f"    C5 [{marker_c}]: {c5_text[:80]}", flush=True)

        all_results.append({
            "task": "long_range", "subtask": ti,
            "std_correct": std_ok, "c5_correct": c5_ok,
            "std_text": std_text[:100], "c5_text": c5_text[:100],
        })

    lr_std_rate = lr_std_correct / max(lr_total, 1)
    lr_c5_rate = lr_c5_correct / max(lr_total, 1)
    print(f"\n  Long-range: Std={lr_std_correct}/{lr_total} ({lr_std_rate:.1%}), C5={lr_c5_correct}/{lr_total} ({lr_c5_rate:.1%})", flush=True)

    # ========================================================================
    # Task 3: Multi-hop reasoning
    # ========================================================================
    print("\n" + "=" * 70, flush=True)
    print("TASK 3: Multi-hop Reasoning", flush=True)
    print("=" * 70, flush=True)

    mh_std_correct = 0
    mh_c5_correct = 0
    mh_total = len(MULTIHOP_TASKS)

    for ti, task in enumerate(MULTIHOP_TASKS):
        input_ids = tokenizer.encode(task["prompt"], return_tensors="pt").to(args.device)

        print(f"\n  [{ti+1}/{mh_total}]...", flush=True)

        # Standard
        with torch.no_grad():
            std_gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        std_text = extract_answer_text(tokenizer, std_gen, input_ids.shape[1])
        std_ok = check_answer(std_text, task["answer"], "contains")
        if std_ok:
            mh_std_correct += 1

        # C5-RPB
        c5_gen = generate_with_rpb(
            model, input_ids, args.max_new_tokens, args.amp, args.device,
            n_layers, n_heads, n_kv_heads, hidden_size, head_dim
        )
        c5_text = extract_answer_text(tokenizer, c5_gen, input_ids.shape[1])
        c5_ok = check_answer(c5_text, task["answer"], "contains")
        if c5_ok:
            mh_c5_correct += 1

        marker_s = "Y" if std_ok else "N"
        marker_c = "Y" if c5_ok else "N"
        print(f"    Std[{marker_s}]: {std_text[:80]}", flush=True)
        print(f"    C5 [{marker_c}]: {c5_text[:80]}", flush=True)

        all_results.append({
            "task": "multihop", "subtask": ti,
            "std_correct": std_ok, "c5_correct": c5_ok,
            "std_text": std_text[:100], "c5_text": c5_text[:100],
        })

    mh_std_rate = mh_std_correct / max(mh_total, 1)
    mh_c5_rate = mh_c5_correct / max(mh_total, 1)
    print(f"\n  Multi-hop: Std={mh_std_correct}/{mh_total} ({mh_std_rate:.1%}), C5={mh_c5_correct}/{mh_total} ({mh_c5_rate:.1%})", flush=True)

    # ========================================================================
    # Task 4: Code completion
    # ========================================================================
    print("\n" + "=" * 70, flush=True)
    print("TASK 4: Code Completion", flush=True)
    print("=" * 70, flush=True)

    cc_std_correct = 0
    cc_c5_correct = 0
    cc_total = len(CODE_TASKS)

    for ti, task in enumerate(CODE_TASKS):
        input_ids = tokenizer.encode(task["prompt"], return_tensors="pt").to(args.device)

        print(f"\n  [{ti+1}/{cc_total}]...", flush=True)

        # Standard
        with torch.no_grad():
            std_gen = model.generate(input_ids, max_new_tokens=30, do_sample=False)
        std_text = extract_answer_text(tokenizer, std_gen, input_ids.shape[1], max_decode=80)
        std_ok = check_answer(std_text, task["answer"], "contains")
        if std_ok:
            cc_std_correct += 1

        # C5-RPB
        c5_gen = generate_with_rpb(
            model, input_ids, 30, args.amp, args.device,
            n_layers, n_heads, n_kv_heads, hidden_size, head_dim
        )
        c5_text = extract_answer_text(tokenizer, c5_gen, input_ids.shape[1], max_decode=80)
        c5_ok = check_answer(c5_text, task["answer"], "contains")
        if c5_ok:
            cc_c5_correct += 1

        marker_s = "Y" if std_ok else "N"
        marker_c = "Y" if c5_ok else "N"
        print(f"    Std[{marker_s}]: {std_text[:60]}", flush=True)
        print(f"    C5 [{marker_c}]: {c5_text[:60]}", flush=True)

        all_results.append({
            "task": "code", "subtask": ti,
            "std_correct": std_ok, "c5_correct": c5_ok,
            "std_text": std_text[:100], "c5_text": c5_text[:100],
        })

    cc_std_rate = cc_std_correct / max(cc_total, 1)
    cc_c5_rate = cc_c5_correct / max(cc_total, 1)
    print(f"\n  Code: Std={cc_std_correct}/{cc_total} ({cc_std_rate:.1%}), C5={cc_c5_correct}/{cc_total} ({cc_c5_rate:.1%})", flush=True)

    # ========================================================================
    # Task 5: Math word problems
    # ========================================================================
    print("\n" + "=" * 70, flush=True)
    print("TASK 5: Math Word Problems", flush=True)
    print("=" * 70, flush=True)

    ma_std_correct = 0
    ma_c5_correct = 0
    ma_total = len(MATH_TASKS)

    for ti, task in enumerate(MATH_TASKS):
        input_ids = tokenizer.encode(task["prompt"], return_tensors="pt").to(args.device)

        print(f"\n  [{ti+1}/{ma_total}]...", flush=True)

        # Standard
        with torch.no_grad():
            std_gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        std_text = extract_answer_text(tokenizer, std_gen, input_ids.shape[1])
        std_ok = check_answer(std_text, task["answer"], "contains")
        if std_ok:
            ma_std_correct += 1

        # C5-RPB
        c5_gen = generate_with_rpb(
            model, input_ids, args.max_new_tokens, args.amp, args.device,
            n_layers, n_heads, n_kv_heads, hidden_size, head_dim
        )
        c5_text = extract_answer_text(tokenizer, c5_gen, input_ids.shape[1])
        c5_ok = check_answer(c5_text, task["answer"], "contains")
        if c5_ok:
            ma_c5_correct += 1

        marker_s = "Y" if std_ok else "N"
        marker_c = "Y" if c5_ok else "N"
        print(f"    Std[{marker_s}]: {std_text[:80]}", flush=True)
        print(f"    C5 [{marker_c}]: {c5_text[:80]}", flush=True)

        all_results.append({
            "task": "math", "subtask": ti,
            "std_correct": std_ok, "c5_correct": c5_ok,
            "std_text": std_text[:100], "c5_text": c5_text[:100],
        })

    ma_std_rate = ma_std_correct / max(ma_total, 1)
    ma_c5_rate = ma_c5_correct / max(ma_total, 1)
    print(f"\n  Math: Std={ma_std_correct}/{ma_total} ({ma_std_rate:.1%}), C5={ma_c5_correct}/{ma_total} ({ma_c5_rate:.1%})", flush=True)

    # ========================================================================
    # Summary
    # ========================================================================
    total_std = nih_std_correct + lr_std_correct + mh_std_correct + cc_std_correct + ma_std_correct
    total_c5 = nih_c5_correct + lr_c5_correct + mh_c5_correct + cc_c5_correct + ma_c5_correct
    total_q = nih_total + lr_total + mh_total + cc_total + ma_total

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    summary_table = f"""
| Task                  | Std Correct | C5-RPB Correct | Delta |
|-----------------------|-------------|----------------|-------|
| 1. Needle-in-Haystack | {nih_std_correct}/{nih_total}         | {nih_c5_correct}/{nih_total}              | {(nih_c5_correct-nih_std_correct):+d}    |
| 2. Long-range Dep.    | {lr_std_correct}/{lr_total}         | {lr_c5_correct}/{lr_total}              | {(lr_c5_correct-lr_std_correct):+d}    |
| 3. Multi-hop Reason.  | {mh_std_correct}/{mh_total}         | {mh_c5_correct}/{mh_total}              | {(mh_c5_correct-mh_std_correct):+d}    |
| 4. Code Completion    | {cc_std_correct}/{cc_total}         | {cc_c5_correct}/{cc_total}              | {(cc_c5_correct-cc_std_correct):+d}    |
| 5. Math Word Problems | {ma_std_correct}/{ma_total}         | {ma_c5_correct}/{ma_total}              | {(ma_c5_correct-ma_std_correct):+d}    |
| **TOTAL**             | **{total_std}/{total_q}**        | **{total_c5}/{total_q}**             | **{(total_c5-total_std):+d}**   |

Model: {model_name} | amp: {args.amp} | Max context: {args.max_context}
"""
    print(summary_table, flush=True)

    verdict = ""
    if total_c5 > total_std:
        verdict = "C5-RPB WINS! Structure injection improves downstream performance."
    elif total_c5 == total_std:
        verdict = "TIE. C5-RPB matches standard with no degradation (free lunch confirmed on tasks)."
    else:
        verdict = "Standard leads. C5-RPB at this amp may need tuning for task-level gains."

    print(f"\nVerdict: {verdict}", flush=True)

    # Save results
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = args.output or f"c5_downstream_eval_{model_name.replace('/', '_')}_{now}.json"

    output_data = {
        "model": model_name,
        "amp": args.amp,
        "date": now,
        "summary": {
            "total_std_correct": total_std,
            "total_c5_correct": total_c5,
            "total_questions": total_q,
            "needle_haystack": {"std": nih_std_correct, "c5": nih_c5_correct, "total": nih_total},
            "long_range": {"std": lr_std_correct, "c5": lr_c5_correct, "total": lr_total},
            "multihop": {"std": mh_std_correct, "c5": mh_c5_correct, "total": mh_total},
            "code": {"std": cc_std_correct, "c5": cc_c5_correct, "total": cc_total},
            "math": {"std": ma_std_correct, "c5": ma_c5_correct, "total": ma_total},
            "verdict": verdict,
        },
        "details": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
