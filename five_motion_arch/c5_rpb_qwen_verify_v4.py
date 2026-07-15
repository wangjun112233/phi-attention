#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5-RPB Attention - Qwen2.5 Real Model Verification (v4)
FIXED: Measure head-level C5 structure (h%5 grouping) per prompt,
not motion-level structure across prompts.

Key insight: C5-RPB biases are added per head with phase = 2pi*(h%5)/5.
So heads with same h%5 share the same bias, and adjacent h%5 phases
should produce more similar attention patterns than non-adjacent ones.

Usage:
  python c5_rpb_qwen_verify_v4.py [--model_path PATH] [--rpb_amp FLOAT] [--device cpu|cuda]
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import argparse
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# C5-RPB Generation
# ============================================================================

def make_c5_rpb_tensor(n_heads, seq_len, amplitude=1.0, phi_shift=0.0, device='cpu', dtype=torch.float32):
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
# Prompts
# ============================================================================

PROMPTS = [
    "Analyze the mathematical structure of prime numbers and explain why they form the foundation of number theory.",
    "What if consciousness emerges not from complexity but from a specific geometric pattern in neural activity?",
    "When a civilization collapses, what are the last things that disappear and why do they persist?",
    "The contradiction between determinism and free will: can both be true simultaneously in a quantum framework?",
    "After all the known forces are accounted for, what remains unexplained about the structure of reality?",
]

# ============================================================================
# Manual forward with RPB
# ============================================================================

def run_model_with_rpb_capture(model, input_ids, attention_mask, rpb_per_layer, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, device):
    captured = {}
    
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        bsz, seq_len_q, _ = hidden.shape
        
        # Get rotary_emb
        rotary_emb = None
        if hasattr(model.model, 'rotary_emb'):
            rotary_emb = model.model.rotary_emb
        elif hasattr(model.model.layers[0].self_attn, 'rotary_emb'):
            rotary_emb = model.model.layers[0].self_attn.rotary_emb
        
        for l_idx in range(n_layers):
            layer = model.model.layers[l_idx]
            rpb = rpb_per_layer.get(l_idx) if rpb_per_layer else None
            
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            
            attn = layer.self_attn
            query = attn.q_proj(hidden)
            key = attn.k_proj(hidden)
            value = attn.v_proj(hidden)
            
            q = query.view(bsz, seq_len_q, n_heads, head_dim).transpose(1, 2)
            k = key.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)
            v = value.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)
            
            if n_kv_heads < n_heads:
                n_rep = n_heads // n_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)
                v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)
            
            # Apply RoPE
            if rotary_emb is not None:
                position_ids = torch.arange(seq_len_q, device=device).unsqueeze(0)
                try:
                    cos_r, sin_r = rotary_emb(v, position_ids, seq_len=seq_len_q)
                except TypeError:
                    try:
                        cos_r, sin_r = rotary_emb(v, position_ids)
                    except:
                        cos_r, sin_r = rotary_emb(position_ids, seq_len=seq_len_q)
                
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
            
            # Causal mask + attention mask
            causal_mask = torch.triu(
                torch.full((seq_len_q, seq_len_q), float('-inf'), device=device, dtype=hidden.dtype),
                diagonal=1
            )
            if attention_mask is not None:
                extended_mask = attention_mask[:, None, None, :].to(dtype=hidden.dtype)
                extended_mask = (1.0 - extended_mask) * torch.finfo(hidden.dtype).min
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) + extended_mask
            else:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            scores = scores + causal_mask
            attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            
            captured[l_idx] = attn_weights[0].cpu().float().numpy()  # [n_heads, seq, seq]
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len_q, hidden_size)
            attn_output = attn.o_proj(attn_output)
            
            hidden = residual + attn_output
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
        
        hidden = model.model.norm(hidden)
    
    return captured

# ============================================================================
# CORRECTED C5 Measurement: head-level, h%5 grouping
# ============================================================================

C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ  = [(0,2),(0,3),(1,3),(1,4),(2,4)]

def measure_head_phase_c5(attn_weights, n_heads):
    """Measure C5 cyclic structure between head groups (h%5).
    
    For each head, extract a feature vector from its attention pattern.
    Then group heads by h%5 (phase group) and measure:
    - Do heads in the same phase group have similar features? (intra-phase clustering)
    - Are adjacent phase groups more similar than non-adjacent? (C5 cyclic structure)
    - DFT k=1 energy in the 5-phase similarity matrix
    
    attn_weights: [n_heads, seq, seq] for a single prompt at a single layer
    """
    # Feature: for each head, use the attention distribution of the last token
    # This is a seq_len-dimensional vector, much more informative than a scalar
    features = np.zeros((n_heads, attn_weights.shape[1]))
    for h in range(n_heads):
        features[h] = attn_weights[h, -1, :]  # last token's attention distribution
    
    # Normalize each head's feature
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    features_norm = features / norms
    
    # Compute head-to-head similarity matrix
    head_sim = features_norm @ features_norm.T  # [n_heads, n_heads]
    
    # Group by h%5: compute phase-group centroid similarity
    n_phases = 5
    phase_centroids = np.zeros((n_phases, features.shape[1]))
    phase_counts = np.zeros(n_phases)
    
    for h in range(n_heads):
        p = h % 5
        phase_centroids[p] += features[h]
        phase_counts[p] += 1
    
    for p in range(n_phases):
        if phase_counts[p] > 0:
            phase_centroids[p] /= phase_counts[p]
    
    # Normalize centroids
    cnorms = np.linalg.norm(phase_centroids, axis=1, keepdims=True)
    cnorms = np.maximum(cnorms, 1e-10)
    phase_centroids_norm = phase_centroids / cnorms
    
    # Phase similarity matrix
    phase_sim = phase_centroids_norm @ phase_centroids_norm.T  # [5, 5]
    
    # C5 structure metrics
    adj_sim = np.mean([phase_sim[i,j] for i,j in C5_ADJACENT])
    nonadj_sim = np.mean([phase_sim[i,j] for i,j in C5_NONADJ])
    
    # DFT k=1 of phase similarity
    n = 5
    W_dft = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
    dft = W_dft @ phase_centroids
    freq_energy = np.array([np.mean(np.abs(dft[k])**2) for k in range(n)])
    total = freq_energy.sum()
    k1_ratio = (freq_energy[1] + freq_energy[4]) / max(total, 1e-10)
    
    # Intra-phase vs inter-phase similarity
    intra_sims = []
    inter_sims = []
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            if i % 5 == j % 5:
                intra_sims.append(head_sim[i, j])
            else:
                inter_sims.append(head_sim[i, j])
    
    intra_sim = np.mean(intra_sims) if intra_sims else 0
    inter_sim = np.mean(inter_sims) if inter_sims else 0
    
    # RPB effect: measure how much each head's attention shifts from mean
    # With C5-RPB, heads in different phases should deviate differently
    mean_attn = features.mean(axis=0, keepdims=True)  # mean across heads
    deviations = features - mean_attn  # [n_heads, seq]
    
    # Project deviations onto C5 phase directions
    phase_projection = np.zeros(n_phases)
    for p in range(n_phases):
        mask = np.array([1.0 if h % 5 == p else 0.0 for h in range(n_heads)])
        mask /= max(mask.sum(), 1e-10)
        phase_dev = mask @ deviations  # weighted average deviation for this phase
        phase_projection[p] = np.linalg.norm(phase_dev)
    
    # C5 structure in phase projections
    pp_norms = np.linalg.norm(phase_projection)
    if pp_norms > 1e-10:
        pp_normalized = phase_projection / pp_norms
    else:
        pp_normalized = phase_projection
    
    return {
        'k1_ratio': float(k1_ratio),
        'adj_sim': float(adj_sim),
        'nonadj_sim': float(nonadj_sim),
        'intra_phase_sim': float(intra_sim),
        'inter_phase_sim': float(inter_sim),
        'phase_sim_matrix': phase_sim,
        'phase_projection': phase_projection,
        'head_sim_matrix': head_sim,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r'C:\Users\WANGJUN\d10\ms_cache\models\Qwen--Qwen2.5-1.5B\snapshots\master')
    parser.add_argument('--rpb_amp', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    model_path = os.path.normpath(os.path.abspath(args.model_path))
    if not os.path.isdir(model_path):
        print(f"ERROR: Model path not found: {model_path}", flush=True)
        return

    print("=" * 70, flush=True)
    print("C5-RPB Attention - Qwen2.5 Real Model Verification (v4)", flush=True)
    print("FIXED: Head-level h%5 phase grouping measurement", flush=True)
    print("=" * 70, flush=True)
    print(f"\nModel: {model_path}", flush=True)
    print(f"RPB amplitude: {args.rpb_amp}", flush=True)
    print(f"Device: {args.device}", flush=True)

    # Load model
    print("\n[1] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  Tokenizer loaded.", flush=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
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
    print(f"  h%5 groups: {[h%5 for h in range(n_heads)]}", flush=True)

    # Prepare inputs
    print("\n[2] Preparing inputs...", flush=True)
    encoded = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors='pt', padding=False).to(args.device)
        encoded.append(inputs)
    
    max_seq = max(e['input_ids'].shape[1] for e in encoded)
    pad_id = tokenizer.pad_token_id
    for i, e in enumerate(encoded):
        cur_len = e['input_ids'].shape[1]
        if cur_len < max_seq:
            pad_len = max_seq - cur_len
            e['input_ids'] = torch.cat([e['input_ids'], torch.full((1, pad_len), pad_id, device=args.device)], dim=1)
            e['attention_mask'] = torch.cat([e['attention_mask'], torch.zeros((1, pad_len), device=args.device, dtype=torch.long)], dim=1)
    print(f"  {len(PROMPTS)} prompts, padded to seq_len={max_seq}", flush=True)

    # ===== Experiment 1: Standard Attention =====
    print("\n[3] Standard Attention...", flush=True)
    std_attn = {}  # {layer: {prompt_idx: ndarray[n_heads, seq, seq]}}
    
    with torch.no_grad():
        for pi, inputs in enumerate(encoded):
            output = model(**inputs, output_attentions=True)
            for l, attn in enumerate(output.attentions):
                if l not in std_attn:
                    std_attn[l] = {}
                std_attn[l][pi] = attn[0].cpu().float().numpy()
    print("  Done.", flush=True)

    # ===== Experiment 2: C5-RPB Attention =====
    print(f"\n[4] C5-RPB Attention (amp={args.rpb_amp})...", flush=True)
    
    rpb_normal = make_c5_rpb_tensor(n_heads, max_seq + 10, amplitude=args.rpb_amp, device=args.device, dtype=torch.float32)
    rpb_per_layer = {l: rpb_normal for l in range(n_layers)}

    c5_attn = {}
    for pi, inputs in enumerate(encoded):
        print(f"  Prompt {pi+1}/5...", flush=True)
        captured = run_model_with_rpb_capture(
            model, inputs['input_ids'], inputs.get('attention_mask'),
            rpb_per_layer, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, args.device
        )
        for l in captured:
            if l not in c5_attn:
                c5_attn[l] = {}
            c5_attn[l][pi] = captured[l]
    print("  Done.", flush=True)

    # ===== Experiment 3: Z2 collapse =====
    collapse_layer = n_layers // 2
    print(f"\n[5] Z2 collapse (flip at layer {collapse_layer})...", flush=True)
    
    rpb_z2 = make_c5_rpb_tensor(n_heads, max_seq + 10, amplitude=args.rpb_amp,
                                  phi_shift=math.pi, device=args.device, dtype=torch.float32)
    rpb_collapse = {l: (rpb_normal if l <= collapse_layer else rpb_z2) for l in range(n_layers)}

    z2_attn = {}
    for pi, inputs in enumerate(encoded):
        print(f"  Prompt {pi+1}/5...", flush=True)
        captured = run_model_with_rpb_capture(
            model, inputs['input_ids'], inputs.get('attention_mask'),
            rpb_collapse, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, args.device
        )
        for l in captured:
            if l not in z2_attn:
                z2_attn[l] = {}
            z2_attn[l][pi] = captured[l]
    print("  Done.", flush=True)

    # ===== Measure C5 structure =====
    print("\n[6] Head-level C5 structure measurement...", flush=True)
    
    # Average across prompts for each layer
    results = {'std': {}, 'c5': {}, 'z2': {}}
    
    for l in range(n_layers):
        for tag, attn_dict in [('std', std_attn), ('c5', c5_attn), ('z2', z2_attn)]:
            if l not in attn_dict:
                continue
            # Average measurement across prompts
            k1_vals = []
            adj_vals = []
            nonadj_vals = []
            intra_vals = []
            inter_vals = []
            phase_sims = []
            
            for pi in attn_dict[l]:
                m = measure_head_phase_c5(attn_dict[l][pi], n_heads)
                k1_vals.append(m['k1_ratio'])
                adj_vals.append(m['adj_sim'])
                nonadj_vals.append(m['nonadj_sim'])
                intra_vals.append(m['intra_phase_sim'])
                inter_vals.append(m['inter_phase_sim'])
                phase_sims.append(m['phase_sim_matrix'])
            
            results[tag][l] = {
                'k1_ratio': np.mean(k1_vals),
                'adj_sim': np.mean(adj_vals),
                'nonadj_sim': np.mean(nonadj_vals),
                'intra_phase_sim': np.mean(intra_vals),
                'inter_phase_sim': np.mean(inter_vals),
                'phase_sim_matrix': np.mean(phase_sims, axis=0),
            }

    # Print results
    print(f"\n  Head grouping: {[h%5 for h in range(n_heads)]}", flush=True)
    print(f"\n  {'Layer':>6} | {'Std k1':>8} {'C5 k1':>8} {'Z2 k1':>8} | {'Std adj':>8} {'C5 adj':>8} | {'Std intra':>9} {'C5 intra':>9} | {'Dk1':>8}", flush=True)
    print(f"  {'-'*90}", flush=True)
    
    for l in range(n_layers):
        s = results['std'].get(l, {})
        c = results['c5'].get(l, {})
        z = results['z2'].get(l, {})
        
        sk1 = s.get('k1_ratio', 0)
        ck1 = c.get('k1_ratio', 0)
        zk1 = z.get('k1_ratio', 0)
        sa = s.get('adj_sim', 0)
        ca = c.get('adj_sim', 0)
        si = s.get('intra_phase_sim', 0)
        ci = c.get('intra_phase_sim', 0)
        dk1 = ck1 - sk1
        
        tag = ""
        if l == collapse_layer: tag = " <- Z2 flip"
        elif l == collapse_layer + 1: tag = " <- after flip"
        
        print(f"  {l:6d} | {sk1:8.4f} {ck1:8.4f} {zk1:8.4f} | {sa:8.4f} {ca:8.4f} | {si:9.4f} {ci:9.4f} | {dk1:8.4f}{tag}", flush=True)

    # ===== Phase similarity matrices =====
    print(f"\n[7] Phase similarity matrices (last layer)...", flush=True)
    last_layer = n_layers - 1
    
    for tag in ['std', 'c5', 'z2']:
        if last_layer not in results[tag]:
            continue
        psim = results[tag][last_layer]['phase_sim_matrix']
        print(f"\n  {tag.upper()} phase similarity:", flush=True)
        for i in range(5):
            row = ' '.join(f'{psim[i,j]:7.3f}' for j in range(5))
            print(f"    phase {i}: {row}", flush=True)

    # ===== Z2 collapse effect =====
    print(f"\n[8] Z2 collapse effect...", flush=True)
    for l in [collapse_layer-1, collapse_layer, collapse_layer+1, last_layer]:
        if l < 0 or l not in results['c5'] or l not in results['z2']:
            continue
        cn = results['c5'][l]['phase_sim_matrix']
        cz = results['z2'][l]['phase_sim_matrix']
        shift = np.mean(np.abs(cn - cz))
        tag = " <-- Z2 flip" if l == collapse_layer else ""
        print(f"  Layer {l}: phase_sim_shift = {shift:.6f}{tag}", flush=True)

    # ===== Core conclusion =====
    print("\n" + "=" * 70, flush=True)
    print("CORE CONCLUSIONS", flush=True)
    print("=" * 70, flush=True)
    
    # Check if C5-RPB increases intra-phase similarity
    std_intra_final = results['std'].get(last_layer, {}).get('intra_phase_sim', 0)
    c5_intra_final = results['c5'].get(last_layer, {}).get('intra_phase_sim', 0)
    std_k1_final = results['std'].get(last_layer, {}).get('k1_ratio', 0)
    c5_k1_final = results['c5'].get(last_layer, {}).get('k1_ratio', 0)
    
    print(f"\n  Standard k1 (last layer): {std_k1_final:.4f}", flush=True)
    print(f"  C5-RPB  k1 (last layer): {c5_k1_final:.4f}", flush=True)
    print(f"  Delta k1:                 {c5_k1_final - std_k1_final:.4f}", flush=True)
    print(f"\n  Standard intra-phase sim: {std_intra_final:.4f}", flush=True)
    print(f"  C5-RPB  intra-phase sim: {c5_intra_final:.4f}", flush=True)
    print(f"  Delta intra:              {c5_intra_final - std_intra_final:.4f}", flush=True)
    
    dk1 = c5_k1_final - std_k1_final
    dintra = c5_intra_final - std_intra_final
    
    if dk1 > 0.05 or dintra > 0.02:
        print("\n  >> C5-RPB SIGNIFICANTLY enhances head-level C5 structure!", flush=True)
    elif dk1 > 0.01 or dintra > 0.005:
        print("\n  >> C5-RPB has modest enhancement on head-level C5 structure", flush=True)
    else:
        print("\n  >> C5-RPB does not significantly enhance C5 structure", flush=True)
        print("  >> Attention naturally preserves phase, but RPB bias is too weak vs trained weights", flush=True)

    # Z2 verdict
    if collapse_layer + 1 in results['c5'] and collapse_layer + 1 in results['z2']:
        cn = results['c5'][collapse_layer+1]['phase_sim_matrix']
        cz = results['z2'][collapse_layer+1]['phase_sim_matrix']
        shift = np.mean(np.abs(cn - cz))
        print(f"\n  Z2 collapse shift (layer {collapse_layer+1}): {shift:.6f}", flush=True)
        if shift > 0.05:
            print("  >> Z2 negation produces SIGNIFICANT collapse!", flush=True)
        elif shift > 0.01:
            print("  >> Z2 negation produces observable shift", flush=True)
        else:
            print("  >> Z2 negation shift still too small", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
