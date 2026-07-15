#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5-RPB Attention - Qwen2.5 Real Model Verification
Verify C5-RPB on trained models where heads are already differentiated.
Compare: standard vs C5-RPB vs C5-RPB+Z2
Measure: attention weight C5 cyclic structure + Z2 collapse shift

Usage:
  python c5_rpb_qwen_verify.py [--model_path PATH] [--rpb_amp FLOAT] [--device cpu|cuda]

Compatible: transformers 5.13.1, Qwen2.5-1.5B/3B
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import argparse
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# C5-RPB Generation
# ============================================================================

def make_c5_rpb_tensor(n_heads, seq_len, amplitude=1.0, phi_shift=0.0, device='cpu', dtype=torch.float32):
    """Generate C5-structured relative position bias (PyTorch tensor)
    B[h, i, j] = A * cos(2pi(h%5)/5 + phi_shift + pi*(i-j)/seq_len)
    """
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
# 5 motion test prompts
# ============================================================================

MOTION_NAMES = ["Ren", "Yu", "Luo", "Lie", "Yu2"]
MOTION_ORDER = MOTION_NAMES

PROMPTS = {
    "Ren":  "Analyze the mathematical structure of prime numbers and explain why they form the foundation of number theory.",
    "Yu":   "What if consciousness emerges not from complexity but from a specific geometric pattern in neural activity?",
    "Luo":  "When a civilization collapses, what are the last things that disappear and why do they persist?",
    "Lie":  "The contradiction between determinism and free will: can both be true simultaneously in a quantum framework?",
    "Yu2":  "After all the known forces are accounted for, what remains unexplained about the structure of reality?",
}

# ============================================================================
# C5 Structure Measurement
# ============================================================================

C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ  = [(0,2),(0,3),(1,3),(1,4),(2,4)]

def measure_attn_c5(attn_weights_dict):
    """Measure C5 cyclic structure in attention weights
    attn_weights_dict: {layer_idx: ndarray [5_motions, n_heads, seq_len, seq_len]}
    Returns: {layer: {k1_ratio, adj_sim, nonadj_sim, nearest_c5, sim_matrix}}
    """
    results = {}

    for layer_idx, weights in attn_weights_dict.items():
        n_heads = weights.shape[1]

        # 5 motions x n_heads activation matrix
        motion_head_act = np.zeros((5, n_heads))
        for mi in range(5):
            w = weights[mi]  # [n_heads, seq, seq]
            for h in range(n_heads):
                motion_head_act[mi, h] = w[h, -1, :].mean()

        # C5 structure (similarity between 5 motions)
        norms = np.linalg.norm(motion_head_act, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = motion_head_act / norms
        sim = normalized @ normalized.T

        adj_sim = np.mean([sim[i,j] for i,j in C5_ADJACENT])
        nonadj_sim = np.mean([sim[i,j] for i,j in C5_NONADJ])
        circular_ratio = adj_sim / max(abs(nonadj_sim), 1e-10)

        nearest_c5 = 0
        for i in range(5):
            sims = sim[i].copy()
            sims[i] = -999
            nearest = np.argmax(sims)
            if nearest in [(i+1)%5, (i-1)%5]:
                nearest_c5 += 1

        # DFT k=1
        n = 5
        W_dft = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
        dft = W_dft @ motion_head_act
        freq_energy = np.array([np.mean(np.abs(dft[k])**2) for k in range(n)])
        total = freq_energy.sum()
        k1_ratio = (freq_energy[1] + freq_energy[4]) / max(total, 1e-10)

        results[layer_idx] = {
            'k1_ratio': float(k1_ratio),
            'adj_sim': float(adj_sim),
            'nonadj_sim': float(nonadj_sim),
            'circular_ratio': float(circular_ratio),
            'nearest_c5': nearest_c5,
            'sim_matrix': sim,
        }

    return results

def measure_head_c5_structure(attn_weights):
    """Measure C5 structure between heads"""
    n_heads = attn_weights.shape[1]
    if n_heads >= 5:
        selected_heads = [i * n_heads // 5 for i in range(5)]
    else:
        selected_heads = list(range(n_heads))

    act = np.zeros((5, len(selected_heads)))
    for mi in range(5):
        w = attn_weights[mi]
        for hi, h in enumerate(selected_heads):
            act[mi, hi] = w[h, -1, :].mean()

    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = act / norms
    sim = normalized @ normalized.T

    adj_sim = np.mean([sim[i,j] for i,j in C5_ADJACENT])
    nonadj_sim = np.mean([sim[i,j] for i,j in C5_NONADJ])

    return {'adj_sim': float(adj_sim), 'nonadj_sim': float(nonadj_sim), 'sim_matrix': sim}

# ============================================================================
# Manual layer-by-layer forward with C5-RPB injection
# ============================================================================

def forward_with_rpb(model, input_ids, rpb_per_layer, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, capture_attn=False):
    """Run forward pass manually, layer by layer, injecting C5-RPB into attention scores.
    Returns: dict {layer_idx: ndarray [n_heads, seq, seq]} if capture_attn else {}
    """
    captured = {} if capture_attn else None

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        bsz, seq_len_q, _ = hidden.shape

        for l_idx, layer in enumerate(model.model.layers):
            rpb = rpb_per_layer.get(l_idx) if rpb_per_layer else None

            # Self-attention
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            attn = layer.self_attn
            query = attn.q_proj(hidden)
            key = attn.k_proj(hidden)
            value = attn.v_proj(hidden)

            q = query.view(bsz, seq_len_q, n_heads, head_dim).transpose(1, 2)
            k = key.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)
            v = value.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)

            # GQA
            if n_kv_heads < n_heads:
                n_rep = n_heads // n_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)
                v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)

            # RoPE
            position_ids = torch.arange(seq_len_q, device=hidden.device).unsqueeze(0)
            cos_r, sin_r = attn.rotary_emb(v, position_ids, seq_len=seq_len_q)

            def rotate_half(x):
                x1 = x[..., :x.shape[-1]//2]
                x2 = x[..., x.shape[-1]//2:]
                return torch.cat((-x2, x1), dim=-1)

            q = q * cos_r + rotate_half(q) * sin_r
            k = k * cos_r + rotate_half(k) * sin_r

            # Attention scores
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

            # C5-RPB injection
            if rpb is not None:
                rpb_slice = rpb[:, :seq_len_q, :seq_len_q]
                scores = scores + rpb_slice.unsqueeze(0)

            # Causal mask
            mask = torch.triu(torch.full((seq_len_q, seq_len_q), float('-inf'), device=hidden.device, dtype=hidden.dtype), diagonal=1)
            scores = scores + mask

            # Softmax
            attn_w = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

            # Capture
            if capture_attn:
                captured[l_idx] = attn_w[0].cpu().float().numpy()

            # Continue forward
            attn_out = torch.matmul(attn_w, v)
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, seq_len_q, hidden_size)
            attn_out = attn.o_proj(attn_out)

            hidden = residual + attn_out

            # MLP
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        # Final norm
        hidden = model.model.norm(hidden)

    return captured

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r'C:\Users\WANGJUN\d10\ms_cache\models\Qwen-Qwen2.5-1.5B\snapshots\master',
                        help='Local model path')
    parser.add_argument('--rpb_amp', type=float, default=2.0,
                        help='C5-RPB amplitude')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Number of tokens to generate')
    args = parser.parse_args()

    # Normalize local path
    model_path = os.path.normpath(os.path.abspath(args.model_path))
    if not os.path.isdir(model_path):
        print(f"ERROR: Model path not found: {model_path}")
        return

    print("=" * 70, flush=True)
    print("C5-RPB Attention - Qwen2.5 Real Model Verification", flush=True)
    print("=" * 70, flush=True)
    print(f"\nModel: {model_path}", flush=True)
    print(f"RPB amplitude: {args.rpb_amp}", flush=True)
    print(f"Device: {args.device}", flush=True)

    # ===== Load model =====
    print("\n[1] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
    print("  Tokenizer loaded.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    print("  Model loaded.", flush=True)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads

    print(f"  Layers: {n_layers}, Heads: {n_heads}, KV Heads: {n_kv_heads}", flush=True)
    print(f"  Hidden: {hidden_size}, Head dim: {head_dim}", flush=True)

    # ===== Prepare inputs (pad to same length) =====
    print("\n[2] Preparing 5 motion inputs...", flush=True)
    encoded = {}
    for motion, prompt in PROMPTS.items():
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        encoded[motion] = inputs
        print(f"  {motion}: {prompt[:50]}... (seq_len={inputs['input_ids'].shape[1]})", flush=True)

    # Pad all to same length
    max_seq = max(v['input_ids'].shape[1] for v in encoded.values())
    print(f"  Max seq length: {max_seq}", flush=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for motion in encoded:
        cur_len = encoded[motion]['input_ids'].shape[1]
        if cur_len < max_seq:
            pad_len = max_seq - cur_len
            encoded[motion]['input_ids'] = torch.cat(
                [encoded[motion]['input_ids'], torch.full((1, pad_len), pad_id, device=args.device)], dim=1)
            encoded[motion]['attention_mask'] = torch.cat(
                [encoded[motion]['attention_mask'], torch.zeros((1, pad_len), device=args.device, dtype=torch.long)], dim=1)
    print(f"  All inputs padded to seq_len={max_seq}", flush=True)

    # ===== Experiment 1: Standard Attention =====
    print("\n[3] Standard Attention (no RPB)...", flush=True)
    std_attn_stacked = {}
    for l in range(n_layers):
        std_attn_stacked[l] = []

    with torch.no_grad():
        for motion in MOTION_ORDER:
            inputs = encoded[motion]
            output = model(**inputs, output_attentions=True)
            for l, attn in enumerate(output.attentions):
                std_attn_stacked[l].append(attn[0].cpu().float().numpy())

    for l in std_attn_stacked:
        std_attn_stacked[l] = np.stack(std_attn_stacked[l])  # [5, n_heads, seq, seq]
    print("  Done.", flush=True)

    # ===== Experiment 2: C5-RPB Attention =====
    print(f"\n[4] C5-RPB Attention (amp={args.rpb_amp})...", flush=True)

    rpb_normal = make_c5_rpb_tensor(n_heads, max_seq + 10, amplitude=args.rpb_amp, device=args.device, dtype=torch.float32)
    rpb_per_layer = {l: rpb_normal for l in range(n_layers)}

    c5_attn_stacked = {}
    for l in range(n_layers):
        c5_attn_stacked[l] = []

    for motion in MOTION_ORDER:
        inputs = encoded[motion]
        captured = forward_with_rpb(model, inputs['input_ids'], rpb_per_layer, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, capture_attn=True)
        for l in captured:
            c5_attn_stacked[l].append(captured[l])

    for l in c5_attn_stacked:
        c5_attn_stacked[l] = np.stack(c5_attn_stacked[l])
    print("  Done.", flush=True)

    # ===== Measure C5 structure =====
    print("\n[5] C5 structure measurement...", flush=True)

    std_c5 = measure_attn_c5(std_attn_stacked)
    c5_c5 = measure_attn_c5(c5_attn_stacked)

    print(f"\n  {'Layer':>6} | {'Std k1':>8} {'C5-RPB k1':>10} | {'Std adj':>8} {'C5-RPB adj':>10} | {'Dk1':>8}", flush=True)
    print(f"  {'-'*65}", flush=True)

    sample_layers = [0, 1, 5, 10, 14, 20, n_layers-1]
    sample_layers = [l for l in sample_layers if l in std_c5 and l in c5_c5]
    for l in sample_layers:
        s = std_c5[l]
        c = c5_c5[l]
        dk1 = c['k1_ratio'] - s['k1_ratio']
        print(f"  {l:6d} | {s['k1_ratio']:8.4f} {c['k1_ratio']:10.4f} | "
              f"{s['adj_sim']:8.4f} {c['adj_sim']:10.4f} | {dk1:8.4f}", flush=True)

    # ===== Head-dimension C5 =====
    print(f"\n[6] Head-dimension C5 structure (last layer)...", flush=True)

    last_layer = n_layers - 1
    for tag, attn_data in [("Standard", std_attn_stacked), ("C5-RPB", c5_attn_stacked)]:
        if last_layer not in attn_data:
            continue
        hc = measure_head_c5_structure(attn_data[last_layer])
        print(f"  {tag}: adj_sim={hc['adj_sim']:.4f}, nonadj={hc['nonadj_sim']:.4f}", flush=True)
        sim = hc['sim_matrix']
        for i in range(5):
            row = ' '.join(f'{sim[i,j]:7.3f}' for j in range(5))
            print(f"    {MOTION_ORDER[i]:>4}: {row}", flush=True)

    # ===== Z2 collapse experiment =====
    collapse_layer = last_layer // 2
    print(f"\n[7] Z2 collapse (layer {collapse_layer}: C5-RPB -> Z2-RPB)...", flush=True)

    rpb_z2 = make_c5_rpb_tensor(n_heads, max_seq + 10, amplitude=args.rpb_amp,
                                  phi_shift=math.pi, device=args.device, dtype=torch.float32)

    rpb_collapse = {}
    for l in range(n_layers):
        if l <= collapse_layer:
            rpb_collapse[l] = rpb_normal
        else:
            rpb_collapse[l] = rpb_z2

    z2_attn_stacked = {}
    for l in range(n_layers):
        z2_attn_stacked[l] = []

    for motion in MOTION_ORDER:
        inputs = encoded[motion]
        captured = forward_with_rpb(model, inputs['input_ids'], rpb_collapse, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, capture_attn=True)
        for l in captured:
            z2_attn_stacked[l].append(captured[l])

    for l in z2_attn_stacked:
        z2_attn_stacked[l] = np.stack(z2_attn_stacked[l])

    z2_c5 = measure_attn_c5(z2_attn_stacked)

    print(f"\n  Collapse layer: {collapse_layer}", flush=True)
    print(f"  {'Layer':>6} | {'Normal k1':>10} {'Collapsed k1':>13} | {'Normal adj':>10} {'Collapsed adj':>13} | {'Phase shift':>11}", flush=True)
    print(f"  {'-'*75}", flush=True)

    for l in [0, max(collapse_layer-1,0), collapse_layer, min(collapse_layer+1, n_layers-1), last_layer]:
        if l not in c5_c5 or l not in z2_c5:
            continue
        cn = c5_c5[l]
        cz = z2_c5[l]
        shift = np.mean(np.abs(cn['sim_matrix'] - cz['sim_matrix']))
        tag = ""
        if l == collapse_layer: tag = " <-- Z2 flip"
        elif l == collapse_layer + 1: tag = " <-- 1 layer after flip"
        print(f"  {l:6d} | {cn['k1_ratio']:10.4f} {cz['k1_ratio']:13.4f} | "
              f"{cn['adj_sim']:10.4f} {cz['adj_sim']:13.4f} | {shift:11.4f}{tag}", flush=True)

    # ===== Generation comparison =====
    print("\n[8] Generation comparison (first prompt)...", flush=True)

    # Standard output
    with torch.no_grad():
        std_logits = model(**encoded[MOTION_ORDER[0]], output_attentions=False)
        std_text = tokenizer.decode(std_logits.logits.argmax(-1)[0], skip_special_tokens=True)

    # C5-RPB output
    with torch.no_grad():
        inputs_0 = encoded[MOTION_ORDER[0]]
        hidden = model.model.embed_tokens(inputs_0['input_ids'])
        bsz, sq, _ = hidden.shape
        for l_idx, layer in enumerate(model.model.layers):
            rpb = rpb_per_layer.get(l_idx)
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attn = layer.self_attn
            query = attn.q_proj(hidden)
            key = attn.k_proj(hidden)
            value = attn.v_proj(hidden)
            q = query.view(bsz, sq, n_heads, head_dim).transpose(1, 2)
            k = key.view(bsz, sq, n_kv_heads, head_dim).transpose(1, 2)
            v = value.view(bsz, sq, n_kv_heads, head_dim).transpose(1, 2)
            if n_kv_heads < n_heads:
                n_rep = n_heads // n_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, sq, head_dim)
                v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, sq, head_dim)
            position_ids = torch.arange(sq, device=hidden.device).unsqueeze(0)
            cos_r, sin_r = attn.rotary_emb(v, position_ids, seq_len=sq)
            def rotate_half(x):
                x1 = x[..., :x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
                return torch.cat((-x2, x1), dim=-1)
            q = q * cos_r + rotate_half(q) * sin_r
            k = k * cos_r + rotate_half(k) * sin_r
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            if rpb is not None:
                scores = scores + rpb[:, :sq, :sq].unsqueeze(0)
            mask = torch.triu(torch.full((sq, sq), float('-inf'), device=hidden.device, dtype=hidden.dtype), diagonal=1)
            scores = scores + mask
            attn_w = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_w, v)
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, sq, hidden_size)
            attn_out = attn.o_proj(attn_out)
            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
        hidden = model.model.norm(hidden)
        c5_text = tokenizer.decode(hidden.argmax(-1)[0], skip_special_tokens=True)

    print(f"\n  Standard:  {std_text[:200]}", flush=True)
    print(f"\n  C5-RPB:    {c5_text[:200]}", flush=True)

    # ===== Core conclusion =====
    print("\n" + "=" * 70, flush=True)
    print("CORE CONCLUSIONS", flush=True)
    print("=" * 70, flush=True)

    std_k1_final = std_c5.get(last_layer, {}).get('k1_ratio', 0)
    c5_k1_final = c5_c5.get(last_layer, {}).get('k1_ratio', 0)

    print(f"\n  Standard Attention k1 (last layer): {std_k1_final:.4f}", flush=True)
    print(f"  C5-RPB k1 (last layer):             {c5_k1_final:.4f}", flush=True)
    print(f"  Delta k1:                            {c5_k1_final - std_k1_final:.4f}", flush=True)

    if c5_k1_final > std_k1_final + 0.05:
        print("  >> C5-RPB SIGNIFICANTLY enhances C5 cyclic structure on trained model!", flush=True)
    elif c5_k1_final > std_k1_final + 0.01:
        print("  >> C5-RPB has modest enhancement effect", flush=True)
    else:
        print("  >> C5-RPB does not significantly enhance C5 structure (but attention naturally preserves phase)", flush=True)

    # Z2 verdict
    if collapse_layer in c5_c5 and collapse_layer + 1 in c5_c5:
        shift = np.mean(np.abs(c5_c5[collapse_layer+1]['sim_matrix'] - z2_c5[collapse_layer+1]['sim_matrix']))
        print(f"\n  Z2 collapse shift (layer {collapse_layer+1}): {shift:.4f}", flush=True)
        if shift > 0.1:
            print("  >> Z2 negation produces SIGNIFICANT collapse on trained model!", flush=True)
        elif shift > 0.03:
            print("  >> Z2 negation produces observable shift", flush=True)
        else:
            print("  >> Z2 negation shift still too small", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
