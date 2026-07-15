#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5-RPB Diagnostic Report Generator for LLMs
Runs full diagnostic pipeline and generates a formatted Markdown report.

Usage:
  python c5_diagnostic_report.py --model_path PATH --device cpu
  python c5_diagnostic_report.py --model_path PATH --device cpu --output report.md
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
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

DIAG_PROMPTS = [
    "The fundamental theorem of arithmetic establishes that every integer greater than one can be uniquely represented as a product of prime numbers.",
    "Climate change represents one of the most significant challenges facing humanity in the twenty-first century.",
    "The development of artificial intelligence has undergone several distinct phases since its formal inception.",
    "Quantum mechanics fundamentally altered our understanding of physical reality and measurement.",
    "The history of written language traces back to Sumerian cuneiform developed around 3400 BCE.",
]

PPL_TEXTS = [
    "The fundamental theorem of arithmetic establishes that every integer greater than one can be uniquely represented as a product of prime numbers. This result, while seemingly simple, has profound implications across mathematics. In algebraic number theory, the failure of unique factorization in general rings led to the development of ideal theory by Kummer and Dedekind. The theorem also underpins modern cryptographic systems, where the difficulty of factoring large composite numbers into primes forms the basis of RSA encryption and other public-key cryptosystems.",
    "Climate change represents one of the most significant challenges facing humanity in the twenty-first century. The scientific consensus, reflected in reports by the Intergovernmental Panel on Climate Change, indicates that global mean temperatures have risen by approximately one degree Celsius above pre-industrial levels. This warming is primarily attributed to anthropogenic greenhouse gas emissions, particularly carbon dioxide from fossil fuel combustion.",
    "The development of artificial intelligence has undergone several distinct phases since its formal inception at the Dartmouth Conference in 1956. Early approaches focused on symbolic reasoning and expert systems, which achieved notable success in constrained domains but struggled with the complexity of real-world problems. The resurgence of neural network approaches in the 2010s, driven by increased computational power and large datasets, led to breakthroughs in image recognition, natural language processing, and game playing.",
    "Quantum mechanics fundamentally altered our understanding of physical reality. The Schrodinger equation describes how quantum states evolve deterministically, yet measurement produces probabilistic outcomes that cannot be predicted with certainty. This tension between unitary evolution and measurement collapse remains one of the deepest puzzles in physics.",
    "Machine learning models trained on natural language data inevitably reflect the biases present in their training corpora. These biases can manifest in various ways: gender stereotypes in word associations, racial prejudices in sentiment analysis, and cultural assumptions in generated text. Addressing these biases requires careful curation of training data, development of fairness metrics, and ongoing evaluation of model outputs.",
    "The history of written language traces back to Sumerian cuneiform, developed around 3400 BCE in Mesopotamia. Initially used for administrative records, writing systems evolved to represent spoken language more directly. The Phoenician alphabet, emerging around 1050 BCE, was a revolutionary innovation that represented individual consonants with separate symbols.",
]

# ============================================================================
# C5 measurement (head-level h%5, from v4)
# ============================================================================

C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ  = [(0,2),(0,3),(1,3),(1,4),(2,4)]

def measure_head_phase_c5(attn_weights, n_heads):
    features = np.zeros((n_heads, attn_weights.shape[1]))
    for h in range(n_heads):
        features[h] = attn_weights[h, -1, :]
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-10)
    features_norm = features / norms
    head_sim = features_norm @ features_norm.T
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
    cnorms = np.maximum(np.linalg.norm(phase_centroids, axis=1, keepdims=True), 1e-10)
    phase_centroids_norm = phase_centroids / cnorms
    phase_sim = phase_centroids_norm @ phase_centroids_norm.T
    adj_sim = np.mean([phase_sim[i,j] for i,j in C5_ADJACENT])
    nonadj_sim = np.mean([phase_sim[i,j] for i,j in C5_NONADJ])
    n = 5
    W_dft = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
    dft = W_dft @ phase_centroids
    freq_energy = np.array([np.mean(np.abs(dft[k])**2) for k in range(n)])
    total = max(freq_energy.sum(), 1e-10)
    k1_ratio = (freq_energy[1] + freq_energy[4]) / total
    intra_sims, inter_sims = [], []
    for i in range(n_heads):
        for j in range(i+1, n_heads):
            if i % 5 == j % 5:
                intra_sims.append(head_sim[i, j])
            else:
                inter_sims.append(head_sim[i, j])
    return {
        'k1_ratio': float(k1_ratio),
        'adj_sim': float(adj_sim),
        'nonadj_sim': float(nonadj_sim),
        'intra_phase_sim': float(np.mean(intra_sims)) if intra_sims else 0,
        'inter_phase_sim': float(np.mean(inter_sims)) if inter_sims else 0,
        'phase_sim_matrix': phase_sim,
    }

# ============================================================================
# Manual forward with RPB (for k1 measurement)
# ============================================================================

def run_model_with_rpb_capture(model, input_ids, attention_mask, rpb_per_layer, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, device):
    captured = {}
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        bsz, seq_len_q, _ = hidden.shape
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
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            if rpb is not None:
                rpb_slice = rpb[:, :seq_len_q, :seq_len_q]
                scores = scores + rpb_slice.unsqueeze(0)
            causal_mask = torch.triu(
                torch.full((seq_len_q, seq_len_q), float('-inf'), device=device, dtype=hidden.dtype), diagonal=1
            )
            if attention_mask is not None:
                ext = attention_mask[:, None, None, :].to(dtype=hidden.dtype)
                ext = (1.0 - ext) * torch.finfo(hidden.dtype).min
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) + ext
            else:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask
            attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            captured[l_idx] = attn_weights[0].cpu().float().numpy()
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
# PPL computation
# ============================================================================

def compute_ppl_standard(model, tokenizer, texts, device, max_length=256):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            labels = inputs['input_ids'].clone()
            outputs = model(**inputs, labels=labels)
            n = inputs['input_ids'].shape[1]
            total_loss += outputs.loss.item() * n
            total_tokens += n
    avg = total_loss / max(total_tokens, 1)
    return math.exp(avg), avg

def compute_ppl_c5rpb(model, tokenizer, texts, rpb_amp, device, max_length=256, phi_shift=0.0):
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads
    vocab_size = model.config.vocab_size
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            input_ids = inputs['input_ids']
            bsz, seq_len = input_ids.shape
            rpb = make_c5_rpb_tensor(n_heads, seq_len+10, amplitude=rpb_amp, phi_shift=phi_shift, device=device, dtype=torch.float32)
            hidden = model.model.embed_tokens(input_ids)
            rotary_emb = None
            if hasattr(model.model, 'rotary_emb'):
                rotary_emb = model.model.rotary_emb
            elif hasattr(model.model.layers[0].self_attn, 'rotary_emb'):
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
                    try: cos_r, sin_r = rotary_emb(v, position_ids, seq_len=seq_len)
                    except TypeError:
                        try: cos_r, sin_r = rotary_emb(v, position_ids)
                        except: cos_r, sin_r = rotary_emb(position_ids, seq_len=seq_len)
                    def rh(x):
                        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                        return torch.cat((-x2, x1), dim=-1)
                    q = q * cos_r + rh(q) * sin_r
                    k = k * cos_r + rh(k) * sin_r
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                scores = scores + rpb[:, :seq_len, :seq_len].unsqueeze(0)
                cm = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=hidden.dtype), diagonal=1).unsqueeze(0).unsqueeze(0)
                scores = scores + cm
                aw = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                ao = torch.matmul(aw, v).transpose(1, 2).contiguous().reshape(bsz, seq_len, hidden_size)
                ao = attn.o_proj(ao)
                hidden = residual + ao
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().reshape(-1, vocab_size), input_ids[:, 1:].contiguous().reshape(-1))
            total_loss += loss.item() * (seq_len - 1)
            total_tokens += seq_len - 1
    avg = total_loss / max(total_tokens, 1)
    return math.exp(avg), avg

# ============================================================================
# Scoring
# ============================================================================

def grade_c5_compatibility(std_k1, c5_k1, z2_shift, ppl_pct):
    """Return A/B/C/D grade and description."""
    dk1 = c5_k1 - std_k1
    score = 0
    # k1 enhancement
    if dk1 > 0.25: score += 4
    elif dk1 > 0.15: score += 3
    elif dk1 > 0.05: score += 2
    elif dk1 > 0.01: score += 1
    # Z2 detectable
    if z2_shift > 0.15: score += 2
    elif z2_shift > 0.05: score += 1
    # PPL cost
    if ppl_pct < 2: score += 2
    elif ppl_pct < 8: score += 1

    if score >= 7: return "A", "Excellent C5 compatibility. Structure injects cleanly with minimal cost."
    elif score >= 5: return "B", "Good C5 compatibility. Structure injects well, modest PPL cost."
    elif score >= 3: return "C", "Moderate C5 compatibility. Structure injects but with significant cost or weak signal."
    else: return "D", "Low C5 compatibility. Head differentiation insufficient or cost too high."

def recommend_amp(ppl_results):
    """Find the highest amp with <5% PPL cost, or the lowest cost amp."""
    best_amp, best_pct = 0.5, 999
    for amp, ppl, pct in ppl_results:
        if pct < 5 and amp > best_amp:
            best_amp = amp
        if pct < best_pct:
            best_pct = pct
            best_amp_fallback = amp
    return best_amp if best_amp > 0.5 else best_amp_fallback

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r'C:\Users\WANGJUN\d10\ms_cache\models\Qwen--Qwen2.5-1.5B\snapshots\master')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    model_path = os.path.normpath(os.path.abspath(args.model_path))
    model_name = os.path.basename(os.path.dirname(model_path)).replace('--', '/')

    print("=" * 70, flush=True)
    print("C5-RPB Diagnostic Report Generator", flush=True)
    print("=" * 70, flush=True)

    # Load model
    print("\n[1/6] Loading model...", flush=True)
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

    print(f"  {model_name}: {n_layers}L, {n_heads}H, {n_kv_heads}KV, {hidden_size}d, {n_params:.1f}B params", flush=True)

    # Prepare inputs for k1 measurement
    print("\n[2/6] Preparing diagnostic inputs...", flush=True)
    encoded = []
    for prompt in DIAG_PROMPTS:
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

    # Standard attention k1
    print("\n[3/6] Standard attention measurement...", flush=True)
    std_k1_by_layer = {}
    with torch.no_grad():
        for pi, inputs in enumerate(encoded):
            output = model(**inputs, output_attentions=True)
            for l, attn in enumerate(output.attentions):
                if l not in std_k1_by_layer:
                    std_k1_by_layer[l] = []
                m = measure_head_phase_c5(attn[0].cpu().float().numpy(), n_heads)
                std_k1_by_layer[l].append(m['k1_ratio'])
    std_k1_final = np.mean(std_k1_by_layer[n_layers-1])
    std_k1_mean = np.mean([np.mean(v) for v in std_k1_by_layer.values()])
    print(f"  Standard k1 (final): {std_k1_final:.4f}, mean: {std_k1_mean:.4f}", flush=True)

    # C5-RPB attention k1
    print("\n[4/6] C5-RPB attention measurement (amp=0.5)...", flush=True)
    rpb_normal = make_c5_rpb_tensor(n_heads, max_seq+10, amplitude=0.5, device=args.device, dtype=torch.float32)
    rpb_per_layer = {l: rpb_normal for l in range(n_layers)}
    c5_k1_by_layer = {}
    for pi, inputs in enumerate(encoded):
        print(f"  Prompt {pi+1}/5...", flush=True)
        captured = run_model_with_rpb_capture(
            model, inputs['input_ids'], inputs.get('attention_mask'),
            rpb_per_layer, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, args.device
        )
        for l in captured:
            if l not in c5_k1_by_layer:
                c5_k1_by_layer[l] = []
            m = measure_head_phase_c5(captured[l], n_heads)
            c5_k1_by_layer[l].append(m['k1_ratio'])
    c5_k1_final = np.mean(c5_k1_by_layer[n_layers-1])
    c5_k1_mean = np.mean([np.mean(v) for v in c5_k1_by_layer.values()])
    dk1_final = c5_k1_final - std_k1_final
    print(f"  C5-RPB k1 (final): {c5_k1_final:.4f}, delta: {dk1_final:+.4f}", flush=True)

    # Z2 collapse
    collapse_layer = n_layers // 2
    print(f"\n[5/6] Z2 collapse test (flip at layer {collapse_layer})...", flush=True)
    rpb_z2 = make_c5_rpb_tensor(n_heads, max_seq+10, amplitude=0.5, phi_shift=math.pi, device=args.device, dtype=torch.float32)
    rpb_collapse = {l: (rpb_normal if l <= collapse_layer else rpb_z2) for l in range(n_layers)}
    z2_k1_by_layer = {}
    for pi, inputs in enumerate(encoded):
        print(f"  Prompt {pi+1}/5...", flush=True)
        captured = run_model_with_rpb_capture(
            model, inputs['input_ids'], inputs.get('attention_mask'),
            rpb_collapse, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, args.device
        )
        for l in captured:
            if l not in z2_k1_by_layer:
                z2_k1_by_layer[l] = []
            m = measure_head_phase_c5(captured[l], n_heads)
            z2_k1_by_layer[l].append(m['k1_ratio'])

    # Compute Z2 shift from phase similarity matrices at key layers
    # Re-run just for phase sim at collapse_layer+1
    z2_shift = 0.0
    check_layer = min(collapse_layer + 1, n_layers - 1)
    if check_layer in c5_k1_by_layer and check_layer in z2_k1_by_layer:
        z2_shift = abs(np.mean(z2_k1_by_layer[check_layer]) - np.mean(c5_k1_by_layer[check_layer]))
    # Better: re-run with phase sim capture (simplified - use k1 diff as proxy)
    # For more accurate Z2 shift, would need phase sim matrices
    print(f"  Z2 k1 shift (layer {check_layer}): {z2_shift:.4f}", flush=True)

    # Perplexity tests
    print("\n[6/6] Perplexity comparison...", flush=True)
    std_ppl, std_loss = compute_ppl_standard(model, tokenizer, PPL_TEXTS, args.device, args.max_length)
    print(f"  Standard PPL: {std_ppl:.2f}", flush=True)

    ppl_results = []
    for amp in [0.5, 1.0, 2.0]:
        c5_ppl, c5_loss = compute_ppl_c5rpb(model, tokenizer, PPL_TEXTS, amp, args.device, args.max_length)
        pct = ((c5_ppl - std_ppl) / std_ppl) * 100
        ppl_results.append((amp, c5_ppl, pct))
        print(f"  C5-RPB amp={amp}: PPL={c5_ppl:.2f} ({pct:+.1f}%)", flush=True)

    # Grade
    best_amp = recommend_amp(ppl_results)
    best_pct = [p for p in ppl_results if p[0] == best_amp][0][2]
    grade, grade_desc = grade_c5_compatibility(std_k1_final, c5_k1_final, z2_shift, abs(best_pct))

    # Head grouping info
    head_groups = [h % 5 for h in range(n_heads)]
    from collections import Counter
    group_counts = Counter(head_groups)

    # Generate report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# C5-RPB Diagnostic Report

**Model:** {model_name}  
**Date:** {now}  
**Grade: {grade}** — {grade_desc}

---

## 1. Model Profile

| Parameter | Value |
|-----------|-------|
| Layers | {n_layers} |
| Attention Heads | {n_heads} |
| KV Heads | {n_kv_heads} |
| Hidden Size | {hidden_size} |
| Head Dimension | {head_dim} |
| Parameters | {n_params:.1f}B |

## 2. Head Phase Grouping (h%5)

C5-RPB assigns heads to 5 phase groups based on head index modulo 5:

| Phase | Heads | Count |
|-------|-------|-------|
"""
    for p in range(5):
        heads_in_group = [h for h in range(n_heads) if h % 5 == p]
        report += f"| {p} | {', '.join(map(str, heads_in_group))} | {len(heads_in_group)} |\n"

    report += f"""
## 3. C5 Structure Metrics

| Metric | Standard | C5-RPB (amp=0.5) | Delta |
|--------|----------|-------------------|-------|
| DFT k1 (last layer) | {std_k1_final:.4f} | {c5_k1_final:.4f} | **{dk1_final:+.4f}** |
| DFT k1 (mean all layers) | {std_k1_mean:.4f} | {c5_k1_mean:.4f} | **{c5_k1_mean-std_k1_mean:+.4f}** |
| Z2 collapse shift | — | — | {z2_shift:.4f} |

**Interpretation:**
- k1 measures C5 cyclic structure in head attention patterns (0=none, 1=perfect pentagon)
- Standard k1 = {std_k1_final:.4f} → {'Near zero: heads are homogenized' if std_k1_final < 0.05 else 'Low: weak natural C5 structure' if std_k1_final < 0.15 else 'Moderate: some natural phase structure'}
- C5-RPB k1 = {c5_k1_final:.4f} → {'Strong C5 structure injected' if c5_k1_final > 0.2 else 'Moderate C5 structure injected' if c5_k1_final > 0.1 else 'Weak C5 structure'}
- Z2 shift = {z2_shift:.4f} → {'Significant: Z2 negation is detectable' if z2_shift > 0.05 else 'Weak: Z2 negation barely detectable'}

## 4. Perplexity Cost

| Config | PPL | Change |
|--------|-----|--------|
| Standard | {std_ppl:.2f} | — |
"""
    for amp, ppl, pct in ppl_results:
        tag = " ← recommended" if amp == best_amp else ""
        report += f"| C5-RPB (amp={amp}) | {ppl:.2f} | {pct:+.1f}%{tag} |\n"

    report += f"""
## 5. Recommendation

**Recommended amplitude: {best_amp}**

"""
    if grade == "A":
        report += "This model is an excellent candidate for C5-RPB integration. The structure injects cleanly with near-zero perplexity cost. Both training-time and inference-time integration are viable.\n"
    elif grade == "B":
        report += "This model is a good candidate for C5-RPB integration. Structure injects well with modest cost. Training-time integration is recommended for best results; inference-time injection is viable at amp=0.5.\n"
    elif grade == "C":
        report += "This model has moderate C5 compatibility. Structure can be injected but with either weak signal or significant PPL cost. Training-time integration with gradual ramp-up is recommended.\n"
    else:
        report += "This model has low C5 compatibility. Head differentiation may be insufficient, or PPL cost is too high for inference-time injection. Requires training-time integration from scratch.\n"

    report += f"""
## 6. Next Steps

1. **Inference-time**: Add C5-RPB at amp={best_amp} to your model's attention forward pass
2. **Training-time**: Integrate C5-RPB into your training code from the start for zero-cost structure
3. **Combined**: Use phi-Residual (changes "how much") + C5-RPB (changes "where to look") for dual control

---

*Report generated by [phi-attention](https://github.com/wangjun112233/phi-attention) diagnostic tool*
*Contact: fdr-factor@coze.email*
"""

    # Save
    output_path = args.output or f"c5_diagnostic_{model_name.replace('/', '_')}.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}", flush=True)
    print(f"\nGrade: {grade} — {grade_desc}", flush=True)

if __name__ == "__main__":
    main()
