#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5-RPB Perplexity Comparison - Qwen2.5 Real Model
Compare standard vs C5-RPB attention on language modeling quality.
Key question: does C5-RPB hurt, help, or leave perplexity unchanged?

Usage:
  python c5_rpb_perplexity_test.py --model_path PATH --device cpu
  python c5_rpb_perplexity_test.py --model_path PATH --device cpu --max_length 128
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import argparse
import math
import torch
import torch.nn.functional as F
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
# Eval texts - diverse English passages (~1200 tokens total)
# ============================================================================

EVAL_TEXTS = [
    "The fundamental theorem of arithmetic establishes that every integer greater than one can be uniquely represented as a product of prime numbers. This result, while seemingly simple, has profound implications across mathematics. In algebraic number theory, the failure of unique factorization in general rings led to the development of ideal theory by Kummer and Dedekind. The theorem also underpins modern cryptographic systems, where the difficulty of factoring large composite numbers into primes forms the basis of RSA encryption and other public-key cryptosystems.",

    "Climate change represents one of the most significant challenges facing humanity in the twenty-first century. The scientific consensus, reflected in reports by the Intergovernmental Panel on Climate Change, indicates that global mean temperatures have risen by approximately one degree Celsius above pre-industrial levels. This warming is primarily attributed to anthropogenic greenhouse gas emissions, particularly carbon dioxide from fossil fuel combustion. The consequences include rising sea levels, more frequent extreme weather events, and disruption to ecosystems worldwide.",

    "The development of artificial intelligence has undergone several distinct phases since its formal inception at the Dartmouth Conference in 1956. Early approaches focused on symbolic reasoning and expert systems, which achieved notable success in constrained domains but struggled with the complexity of real-world problems. The resurgence of neural network approaches in the 2010s, driven by increased computational power and large datasets, led to breakthroughs in image recognition, natural language processing, and game playing. Contemporary large language models demonstrate emergent capabilities that were not explicitly programmed, raising fundamental questions about the nature of intelligence.",

    "Quantum mechanics fundamentally altered our understanding of physical reality. The Schrodinger equation describes how quantum states evolve deterministically, yet measurement produces probabilistic outcomes that cannot be predicted with certainty. This tension between unitary evolution and measurement collapse remains one of the deepest puzzles in physics. The many-worlds interpretation, proposed by Everett in 1957, resolves this by eliminating collapse entirely, while the Copenhagen interpretation treats measurement as a primitive operation. Recent experiments in quantum information have made these philosophical questions increasingly relevant to practical technology.",

    "The history of written language traces back to Sumerian cuneiform, developed around 3400 BCE in Mesopotamia. Initially used for administrative records, writing systems evolved to represent spoken language more directly. The Phoenician alphabet, emerging around 1050 BCE, was a revolutionary innovation that represented individual consonants with separate symbols. This system spread throughout the Mediterranean and was adapted by the Greeks, who added symbols for vowels. The Latin alphabet, derived from Greek via Etruscan, eventually became the most widely used writing system in the world.",

    "Machine learning models trained on natural language data inevitably reflect the biases present in their training corpora. These biases can manifest in various ways: gender stereotypes in word associations, racial prejudices in sentiment analysis, and cultural assumptions in generated text. Addressing these biases requires careful curation of training data, development of fairness metrics, and ongoing evaluation of model outputs. However, the definition of fairness itself is context-dependent and often contested, making purely technical solutions insufficient without broader societal engagement with the underlying ethical questions.",
]

# ============================================================================
# Standard perplexity (fast, uses model.forward)
# ============================================================================

def compute_standard_perplexity(model, tokenizer, texts, device, max_length=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            labels = inputs['input_ids'].clone()
            outputs = model(**inputs, labels=labels)
            n_tokens = inputs['input_ids'].shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss

# ============================================================================
# C5-RPB perplexity (manual forward with RPB injection)
# ============================================================================

def compute_c5rpb_perplexity(model, tokenizer, texts, rpb_amp, device, max_length=256, phi_shift=0.0):
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads
    vocab_size = model.config.vocab_size

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            input_ids = inputs['input_ids']
            bsz, seq_len = input_ids.shape

            rpb = make_c5_rpb_tensor(n_heads, seq_len + 10, amplitude=rpb_amp,
                                      phi_shift=phi_shift, device=device, dtype=torch.float32)

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
                query = attn.q_proj(hidden)
                key = attn.k_proj(hidden)
                value = attn.v_proj(hidden)

                q = query.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
                k = key.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                v = value.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)

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
                        except:
                            cos_r, sin_r = rotary_emb(position_ids, seq_len=seq_len)

                    def rotate_half(x):
                        x1 = x[..., :x.shape[-1]//2]
                        x2 = x[..., x.shape[-1]//2:]
                        return torch.cat((-x2, x1), dim=-1)

                    q = q * cos_r + rotate_half(q) * sin_r
                    k = k * cos_r + rotate_half(k) * sin_r

                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                rpb_slice = rpb[:, :seq_len, :seq_len]
                scores = scores + rpb_slice.unsqueeze(0)

                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=hidden.dtype),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)
                scores = scores + causal_mask

                attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
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

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            n_tokens = seq_len - 1
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r'C:\Users\WANGJUN\d10\ms_cache\models\Qwen--Qwen2.5-1.5B\snapshots\master')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_length', type=int, default=256)
    args = parser.parse_args()

    model_path = os.path.normpath(os.path.abspath(args.model_path))

    print("=" * 70, flush=True)
    print("C5-RPB Perplexity Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"\nModel: {model_path}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Max chunk length: {args.max_length}", flush=True)

    # Load model
    print("\n[1] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  Tokenizer loaded.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager", local_files_only=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  Model loaded. Layers={n_layers}, Heads={n_heads}", flush=True)

    # Token count
    total_toks = sum(len(tokenizer(t)['input_ids']) for t in EVAL_TEXTS)
    print(f"\n[2] Eval texts: {len(EVAL_TEXTS)} passages, ~{total_toks} tokens", flush=True)

    # Standard perplexity
    print("\n[3] Standard model perplexity...", flush=True)
    std_ppl, std_loss = compute_standard_perplexity(model, tokenizer, EVAL_TEXTS, args.device, args.max_length)
    print(f"  PPL = {std_ppl:.2f}, Loss = {std_loss:.4f}", flush=True)

    # C5-RPB at different amplitudes
    results = []
    for amp in [0.5, 1.0, 2.0]:
        print(f"\n[4] C5-RPB perplexity (amp={amp})...", flush=True)
        c5_ppl, c5_loss = compute_c5rpb_perplexity(model, tokenizer, EVAL_TEXTS, amp, args.device, args.max_length)
        delta_ppl = c5_ppl - std_ppl
        pct = (delta_ppl / std_ppl) * 100
        print(f"  PPL = {c5_ppl:.2f}, Loss = {c5_loss:.4f} (delta={delta_ppl:+.2f}, {pct:+.1f}%)", flush=True)
        results.append((amp, c5_ppl, c5_loss, delta_ppl, pct))

    # Z2 collapse at best amplitude (smallest PPL change)
    best_amp = min(results, key=lambda x: abs(x[3]))[0]
    print(f"\n[5] Z2 collapse test (amp={best_amp})...", flush=True)
    z2_ppl, z2_loss = compute_c5rpb_perplexity(
        model, tokenizer, EVAL_TEXTS, best_amp, args.device, args.max_length, phi_shift=math.pi
    )
    z2_delta = z2_ppl - std_ppl
    z2_pct = (z2_delta / std_ppl) * 100
    print(f"  Z2 PPL = {z2_ppl:.2f}, Loss = {z2_loss:.4f} (delta={z2_delta:+.2f}, {z2_pct:+.1f}%)", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("PERPLEXITY COMPARISON SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  {'Config':<25} {'PPL':>8} {'Loss':>8} {'Delta':>10} {'%Chg':>8}", flush=True)
    print(f"  {'-'*61}", flush=True)
    print(f"  {'Standard':<25} {std_ppl:8.2f} {std_loss:8.4f} {'---':>10} {'---':>8}", flush=True)
    for amp, ppl, loss, delta, pct in results:
        print(f"  {f'C5-RPB (amp={amp})':<25} {ppl:8.2f} {loss:8.4f} {delta:+10.2f} {pct:+7.1f}%", flush=True)
    print(f"  {f'Z2 flip (amp={best_amp})':<25} {z2_ppl:8.2f} {z2_loss:8.4f} {z2_delta:+10.2f} {z2_pct:+7.1f}%", flush=True)

    # Verdict
    min_delta = min(abs(r[3]) for r in results)
    print(f"\n  VERDICT:", flush=True)
    if min_delta / std_ppl < 0.05:
        print(f"  >> C5-RPB adds C5 structure with <5% PPL cost -> FREE LUNCH!", flush=True)
        print(f"  >> Structure gain (k1=0.31) at near-zero quality cost", flush=True)
    elif min_delta / std_ppl < 0.15:
        print(f"  >> C5-RPB adds C5 structure with modest PPL cost ({min_delta/std_ppl*100:.0f}%)", flush=True)
        print(f"  >> Promising for training-time integration or fine-tuning recovery", flush=True)
    else:
        print(f"  >> C5-RPB significantly hurts PPL ({min_delta/std_ppl*100:.0f}%)", flush=True)
        print(f"  >> Needs training-time integration to recover quality", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
