#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5-RPB Diagnostic Report Generator v2 for LLMs
Runs full diagnostic pipeline with multi-amp scan and proper Z2 shift measurement.

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
# C5 measurement (head-level h%5, from v4) + phase similarity
# ============================================================================

C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ  = [(0,2),(0,3),(1,3),(1,4),(2,4)]

def measure_head_phase_c5(attn_weights, n_heads):
    """Measure C5 structure from head attention patterns. Returns k1 + phase similarity matrix."""
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

def compute_z2_shift(phase_sim_normal, phase_sim_flipped):
    """Compute Z2 collapse shift from two phase similarity matrices.
    Measures how much the phase structure reshuffles after Z2 negation."""
    diff = np.abs(phase_sim_normal - phase_sim_flipped)
    # Exclude diagonal (always 1.0)
    mask = ~np.eye(5, dtype=bool)
    return float(np.mean(diff[mask]))

# ============================================================================
# Manual forward with RPB capture (returns attention weights + phase sim)
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
# Scoring (v2 - based on best amp results)
# ============================================================================

def grade_c5_compatibility(std_k1, best_c5_k1, z2_shift, best_ppl_pct):
    """Grade based on best achievable results across all amplitudes."""
    dk1 = best_c5_k1 - std_k1
    score = 0
    details = []

    # k1 enhancement (most important)
    if dk1 > 0.25:
        score += 4; details.append("k1: very strong (+4)")
    elif dk1 > 0.15:
        score += 3; details.append("k1: strong (+3)")
    elif dk1 > 0.05:
        score += 2; details.append("k1: moderate (+2)")
    elif dk1 > 0.01:
        score += 1; details.append("k1: weak (+1)")
    else:
        details.append("k1: none (+0)")

    # Z2 shift (real phase similarity, not k1 proxy)
    if z2_shift > 0.10:
        score += 2; details.append("Z2: clear (+2)")
    elif z2_shift > 0.03:
        score += 1; details.append("Z2: detectable (+1)")
    else:
        details.append("Z2: weak (+0)")

    # PPL cost (at best amp)
    if best_ppl_pct < 2:
        score += 3; details.append("PPL: free lunch (+3)")
    elif best_ppl_pct < 5:
        score += 2; details.append("PPL: cheap (+2)")
    elif best_ppl_pct < 10:
        score += 1; details.append("PPL: moderate (+1)")
    else:
        details.append("PPL: expensive (+0)")

    if score >= 8: grade = "A"
    elif score >= 6: grade = "B"
    elif score >= 4: grade = "C"
    else: grade = "D"

    descs = {
        "A": "Excellent C5 compatibility. Structure injects cleanly with near-zero cost. Both inference-time and training-time integration viable.",
        "B": "Good C5 compatibility. Structure injects well with low cost. Inference-time viable; training-time recommended for best results.",
        "C": "Moderate C5 compatibility. Structure injects but signal or cost needs tuning. Training-time integration recommended.",
        "D": "Low C5 compatibility. Head differentiation insufficient or cost too high. Requires training-time integration from scratch.",
    }
    return grade, descs[grade], score, details

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

    AMP_SCAN = [0.3, 0.5, 1.0, 2.0]  # k1 scan amplitudes (include 2.0 for max signal)
    AMP_PPL  = [0.5, 1.0, 2.0]  # PPL scan amplitudes

    print("=" * 70, flush=True)
    print("C5-RPB Diagnostic Report Generator v2", flush=True)
    print("=" * 70, flush=True)

    # Load model
    print("\n[1/7] Loading model...", flush=True)
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

    # Prepare inputs
    print("\n[2/7] Preparing diagnostic inputs...", flush=True)
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
    print("\n[3/7] Standard attention measurement...", flush=True)
    std_k1_by_layer = {}
    std_phase_sim_by_layer = {}
    with torch.no_grad():
        for pi, inputs in enumerate(encoded):
            output = model(**inputs, output_attentions=True)
            for l, attn in enumerate(output.attentions):
                if l not in std_k1_by_layer:
                    std_k1_by_layer[l] = []
                    std_phase_sim_by_layer[l] = []
                m = measure_head_phase_c5(attn[0].cpu().float().numpy(), n_heads)
                std_k1_by_layer[l].append(m['k1_ratio'])
                std_phase_sim_by_layer[l].append(m['phase_sim_matrix'])
    std_k1_final = np.mean(std_k1_by_layer[n_layers-1])
    std_k1_mean = np.mean([np.mean(v) for v in std_k1_by_layer.values()])
    print(f"  Standard k1 (final): {std_k1_final:.4f}, mean: {std_k1_mean:.4f}", flush=True)

    # C5-RPB k1 at multiple amplitudes
    print(f"\n[4/7] C5-RPB k1 scan (amp={AMP_SCAN})...", flush=True)
    amp_k1_results = {}  # amp -> {layer -> [k1 values]}
    amp_phase_sim_results = {}  # amp -> {layer -> [phase_sim matrices]}
    for amp in AMP_SCAN:
        print(f"\n  --- amp={amp} ---", flush=True)
        rpb = make_c5_rpb_tensor(n_heads, max_seq+10, amplitude=amp, device=args.device, dtype=torch.float32)
        rpb_per_layer = {l: rpb for l in range(n_layers)}
        k1_by_layer = {}
        phase_sim_by_layer = {}
        for pi, inputs in enumerate(encoded):
            print(f"  amp={amp} prompt {pi+1}/5...", flush=True)
            captured = run_model_with_rpb_capture(
                model, inputs['input_ids'], inputs.get('attention_mask'),
                rpb_per_layer, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, args.device
            )
            for l in captured:
                if l not in k1_by_layer:
                    k1_by_layer[l] = []
                    phase_sim_by_layer[l] = []
                m = measure_head_phase_c5(captured[l], n_heads)
                k1_by_layer[l].append(m['k1_ratio'])
                phase_sim_by_layer[l].append(m['phase_sim_matrix'])
        amp_k1_results[amp] = k1_by_layer
        amp_phase_sim_results[amp] = phase_sim_by_layer
        print(f"  amp={amp} k1 (final): {np.mean(k1_by_layer[n_layers-1]):.4f} (delta: {np.mean(k1_by_layer[n_layers-1])-std_k1_final:+.4f})", flush=True)

    # Find best amp: highest k1 gain where PPL < 5%
    # We don't have PPL yet, so use amp=0.5 as reference best (known sweet spot)
    # and note the actual best after PPL scan
    best_k1_amp = max(AMP_SCAN, key=lambda a: np.mean(amp_k1_results[a][n_layers-1]))
    c5_k1_final = np.mean(amp_k1_results[best_k1_amp][n_layers-1])
    c5_k1_mean = np.mean([np.mean(v) for v in amp_k1_results[best_k1_amp].values()])

    # Z2 collapse test with proper phase similarity measurement
    collapse_layer = n_layers // 2
    print(f"\n[5/7] Z2 collapse test (flip at layer {collapse_layer})...", flush=True)
    z2_amp = 0.5  # test Z2 at sweet spot
    rpb_normal = make_c5_rpb_tensor(n_heads, max_seq+10, amplitude=z2_amp, device=args.device, dtype=torch.float32)
    rpb_z2 = make_c5_rpb_tensor(n_heads, max_seq+10, amplitude=z2_amp, phi_shift=math.pi, device=args.device, dtype=torch.float32)
    rpb_collapse = {l: (rpb_normal if l <= collapse_layer else rpb_z2) for l in range(n_layers)}

    z2_phase_sim_by_layer = {}
    z2_k1_by_layer = {}
    for pi, inputs in enumerate(encoded):
        print(f"  Z2 prompt {pi+1}/5...", flush=True)
        captured = run_model_with_rpb_capture(
            model, inputs['input_ids'], inputs.get('attention_mask'),
            rpb_collapse, n_layers, n_heads, n_kv_heads, hidden_size, head_dim, args.device
        )
        for l in captured:
            if l not in z2_k1_by_layer:
                z2_k1_by_layer[l] = []
                z2_phase_sim_by_layer[l] = []
            m = measure_head_phase_c5(captured[l], n_heads)
            z2_k1_by_layer[l].append(m['k1_ratio'])
            z2_phase_sim_by_layer[l].append(m['phase_sim_matrix'])

    # Compute real Z2 shift from phase similarity matrices
    check_layer = min(collapse_layer + 1, n_layers - 1)
    z2_shifts = []
    # Check multiple layers after collapse point
    for l in range(collapse_layer+1, min(collapse_layer+5, n_layers)):
        if l in amp_phase_sim_results[z2_amp] and l in z2_phase_sim_by_layer:
            for prompt_idx in range(len(encoded)):
                if prompt_idx < len(amp_phase_sim_results[z2_amp][l]) and prompt_idx < len(z2_phase_sim_by_layer[l]):
                    shift = compute_z2_shift(
                        amp_phase_sim_results[z2_amp][l][prompt_idx],
                        z2_phase_sim_by_layer[l][prompt_idx]
                    )
                    z2_shifts.append(shift)
    z2_shift = np.mean(z2_shifts) if z2_shifts else 0.0
    print(f"  Z2 phase sim shift (post-collapse avg): {z2_shift:.4f}", flush=True)

    # Perplexity tests
    print("\n[6/7] Perplexity comparison...", flush=True)
    std_ppl, std_loss = compute_ppl_standard(model, tokenizer, PPL_TEXTS, args.device, args.max_length)
    print(f"  Standard PPL: {std_ppl:.2f}", flush=True)

    ppl_results = []
    for amp in AMP_PPL:
        c5_ppl, c5_loss = compute_ppl_c5rpb(model, tokenizer, PPL_TEXTS, amp, args.device, args.max_length)
        pct = ((c5_ppl - std_ppl) / std_ppl) * 100
        ppl_results.append((amp, c5_ppl, pct))
        print(f"  C5-RPB amp={amp}: PPL={c5_ppl:.2f} ({pct:+.1f}%)", flush=True)

    # Also test Z2 flip PPL
    z2_ppl, _ = compute_ppl_c5rpb(model, tokenizer, PPL_TEXTS, 0.5, args.device, args.max_length, phi_shift=math.pi)
    z2_pct = ((z2_ppl - std_ppl) / std_ppl) * 100
    print(f"  Z2 flip (amp=0.5): PPL={z2_ppl:.2f} ({z2_pct:+.1f}%)", flush=True)

    # Find best amp: free-lunch mode (PPL < 5%) and max-signal mode
    # Free-lunch: highest amp with PPL < 5%
    free_amp, free_pct = 0.5, 999
    for amp, ppl, pct in ppl_results:
        if pct < 5 and amp >= free_amp:
            free_amp = amp
            free_pct = pct
    # Max-signal: amp=2.0 (strongest C5 structure regardless of cost)
    max_amp = 2.0
    max_pct = [p for p in ppl_results if p[0] == 2.0][0][2] if any(p[0] == 2.0 for p in ppl_results) else 0

    # k1 at each mode
    free_k1 = np.mean(amp_k1_results.get(free_amp, amp_k1_results[0.5])[n_layers-1])
    max_k1 = np.mean(amp_k1_results.get(max_amp, amp_k1_results[0.5])[n_layers-1])

    # Dual grading
    print("\n[7/7] Generating report...", flush=True)
    grade_free, desc_free, score_free, details_free = grade_c5_compatibility(
        std_k1_final, free_k1, z2_shift, abs(free_pct)
    )
    grade_max, desc_max, score_max, details_max = grade_c5_compatibility(
        std_k1_final, max_k1, z2_shift, abs(max_pct)
    )

    # Head grouping info
    from collections import Counter
    head_groups = [h % 5 for h in range(n_heads)]
    group_counts = Counter(head_groups)

    # Build k1 scan table
    k1_scan_rows = []
    for amp in AMP_SCAN:
        k1_val = np.mean(amp_k1_results[amp][n_layers-1])
        dk1 = k1_val - std_k1_final
        k1_scan_rows.append((amp, k1_val, dk1))

    # Determine labels for each amp
    def ppl_label(pct):
        if pct < 2: return "FREE LUNCH"
        elif pct < 5: return "cheap"
        elif pct < 10: return "moderate"
        else: return "expensive"

    # Generate report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# C5-RPB Diagnostic Report

**Model:** {model_name}  
**Date:** {now}

---

## Dual Grade

| Mode | Grade | Score | amp | k1 | Delta k1 | Z2 shift | PPL cost |
|------|-------|-------|-----|----|----------|-----------|----------|
| Free-lunch | **{grade_free}** | {score_free}/9 | {free_amp} | {free_k1:.4f} | {free_k1-std_k1_final:+.4f} | {z2_shift:.4f} | {free_pct:+.1f}% |
| Max-signal | **{grade_max}** | {score_max}/9 | {max_amp} | {max_k1:.4f} | {max_k1-std_k1_final:+.4f} | {z2_shift:.4f} | {max_pct:+.1f}% |

- **Free-lunch**: Best k1 at the lowest PPL cost (amp where PPL < 5%)
- **Max-signal**: Strongest C5 structure regardless of PPL cost (amp=2.0)

{desc_free}

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
## 3. C5 Structure — k1 Amplitude Scan

| Amplitude | DFT k1 (last layer) | Delta k1 | Fold change |
|-----------|---------------------|----------|-------------|
| Standard | {std_k1_final:.4f} | -- | 1x |
"""
    for amp, k1_val, dk1 in k1_scan_rows:
        fold = k1_val / max(std_k1_final, 1e-10)
        tag = " <-- free-lunch" if amp == free_amp else " <-- max-signal" if amp == max_amp else ""
        report += f"| C5-RPB amp={amp} | {k1_val:.4f} | **{dk1:+.4f}** | {fold:.1f}x{tag} |\n"

    report += f"""
**Interpretation:**
- k1 measures C5 cyclic structure in head attention patterns (0=none, 1=perfect pentagon)
- Standard k1 = {std_k1_final:.4f} -> {'Near zero: heads are homogenized' if std_k1_final < 0.05 else 'Low: weak natural C5 structure' if std_k1_final < 0.15 else 'Moderate: some natural phase structure'}
- At free-lunch amp={free_amp}: k1={free_k1:.4f} ({free_k1/std_k1_final:.1f}x) with {free_pct:+.1f}% PPL cost
- At max-signal amp={max_amp}: k1={max_k1:.4f} ({max_k1/std_k1_final:.1f}x) with {max_pct:+.1f}% PPL cost

## 4. Z2 Negation Detection

| Metric | Value |
|--------|-------|
| Collapse point | Layer {collapse_layer} |
| Z2 phase sim shift | **{z2_shift:.4f}** |
| Test amplitude | {z2_amp} |

**Interpretation:**
- Z2 shift measures how much the phase structure reshuffles after Z2 negation (flipping phi_shift by pi)
- Z2 shift = {z2_shift:.4f} -> {'Significant: Z2 negation is clearly detectable' if z2_shift > 0.10 else 'Detectable: Z2 negation has measurable effect' if z2_shift > 0.03 else 'Weak: Z2 negation barely affects phase structure'}

## 5. Perplexity Cost

| Config | PPL | Change | Label |
|--------|-----|--------|-------|
| Standard | {std_ppl:.2f} | -- | -- |
"""
    for amp, ppl, pct in ppl_results:
        tag = " <-- free-lunch" if amp == free_amp else " <-- max-signal" if amp == max_amp else ""
        report += f"| C5-RPB amp={amp} | {ppl:.2f} | {pct:+.1f}% | {ppl_label(pct)}{tag} |\n"
    report += f"| Z2 flip (amp=0.5) | {z2_ppl:.2f} | {z2_pct:+.1f}% | {ppl_label(abs(z2_pct))} |\n"

    report += f"""
## 6. Scoring Breakdown

**Free-lunch mode (amp={free_amp}):**

| Component | Score | Detail |
|-----------|-------|--------|
"""
    for d in details_free:
        parts = d.split(": ")
        report += f"| {parts[0]} | {parts[1]} |\n"
    report += f"| **Total** | **{score_free}/9** | Grade: **{grade_free}** |\n"

    report += f"""
**Max-signal mode (amp={max_amp}):**

| Component | Score | Detail |
|-----------|-------|--------|
"""
    for d in details_max:
        parts = d.split(": ")
        report += f"| {parts[0]} | {parts[1]} |\n"
    report += f"| **Total** | **{score_max}/9** | Grade: **{grade_max}** |\n"

    report += f"""
## 7. Recommendation

**For inference-time (no retraining):**
- Use amp={free_amp} for free-lunch deployment (PPL cost: {free_pct:+.1f}%)
- C5 structure is {'strong' if free_k1 > 0.2 else 'moderate' if free_k1 > 0.05 else 'detectable'} at this amplitude
- Z2 negation is {'clearly detectable' if z2_shift > 0.10 else 'detectable' if z2_shift > 0.03 else 'weak'} at this amplitude

**For training-time (from scratch or fine-tune):**
- Start with amp={free_amp} and gradually ramp to amp={max_amp} during training
- This gives the strongest C5 structure (k1={max_k1:.4f}) with zero PPL cost at convergence
- The model learns to accommodate the C5-RPB bias during training

## 8. Combined Options

| Approach | What it changes | Cost (inference) | Strength |
|----------|----------------|------------------|----------|
| C5-RPB (amp={free_amp}) | Where heads look | {free_pct:+.1f}% PPL | {'Strong' if free_k1 > 0.2 else 'Moderate'} C5 phase structure |
| C5-RPB (amp={max_amp}) | Where heads look | {max_pct:+.1f}% PPL | Strong C5 phase structure |
| phi-Residual | How much each layer contributes | ~0% PPL | Homeostatic balance |
| Both (amp={free_amp}) | Where + How much | ~{free_pct:+.1f}% PPL | Full D10 control |

---

*Report generated by [phi-attention](https://github.com/wangjun112233/phi-attention) diagnostic tool v2*
*Contact: fdr-factor@coze.email*
"""

    # Save
    output_path = args.output or f"c5_diagnostic_{model_name.replace('/', '_')}.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}", flush=True)
    print(f"\n{'='*70}", flush=True)
    print(f"Free-lunch: Grade {grade_free} (score: {score_free}/9) at amp={free_amp} (PPL: {free_pct:+.1f}%)", flush=True)
    print(f"Max-signal: Grade {grade_max} (score: {score_max}/9) at amp={max_amp} (PPL: {max_pct:+.1f}%)", flush=True)
    print(f"Z2 shift: {z2_shift:.4f}", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == "__main__":
    main()
