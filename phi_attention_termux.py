#!/usr/bin/env python3
"""
φ-Attention vs Standard: Termux/Phone Training Comparison
==========================================================
CPU-only, minimal dependencies, ~1-2 hours on phone.

Only needs: torch (numpy included)
No transformers/datasets needed — uses tiny shakespeare dataset auto-downloaded.

Install in Termux:
  pkg install python
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  python phi_attention_termux.py
"""

import math
import time
import json
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== C5 COUPLING ==========

PHI = (1 + math.sqrt(5)) / 2
COS72 = math.cos(math.radians(72))


def build_c5_coupling(n_heads: int) -> torch.Tensor:
    C5 = torch.eye(5) + COS72 * torch.tensor([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=torch.float32)
    n_groups = n_heads // 5
    n_remaining = n_heads % 5
    blocks = [C5] * n_groups
    if n_remaining > 0:
        blocks.append(torch.eye(n_remaining))
    M = torch.block_diag(*blocks)
    eigs = torch.linalg.eigvalsh(M)
    print(f"  C5 coupling eigenvalues: [{', '.join(f'{e:.4f}' for e in eigs)}] | λ_max={eigs[-1]:.6f} (φ={PHI:.6f})")
    return M


# ========== CHARACTER-LEVEL TOKENIZER ==========

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"  Vocab: {self.vocab_size} chars")

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i < self.vocab_size)


# ========== DATA ==========

def load_tiny_shakespeare(seq_len=128):
    """Download tiny shakespeare if not present, return tokenized dataset."""
    path = "input.txt"
    if not os.path.exists(path):
        print("  Downloading tiny shakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path
        )
    with open(path, 'r') as f:
        text = f.read()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    # 90/10 split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"  Train: {len(train_data):,} chars | Val: {len(val_data):,} chars")
    return tok, train_data, val_data


def get_batch(data, seq_len, batch_size):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


# ========== MODEL ==========

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_c5=False, max_seq_len=256):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_c5 = use_c5
        self.d_model = d_model

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .view(1, 1, max_seq_len, max_seq_len)
        )

        if use_c5:
            coupling = build_c5_coupling(n_heads)
            self.register_buffer('coupling_matrix', coupling)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # ★ C5 coupling before softmax ★
        if self.use_c5:
            scores = torch.einsum('hn,bnsq->bhsq', self.coupling_matrix, scores)

        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_c5=False, max_seq_len=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, use_c5, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLite(nn.Module):
    def __init__(self, vocab_size, d_model=160, n_heads=5, n_layers=4,
                 use_c5=False, max_seq_len=128):
        super().__init__()
        self.use_c5 = use_c5
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, use_c5, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        c5_label = "C5" if use_c5 else "STD"
        print(f"  [{c5_label}] {n_params/1e6:.2f}M params | {n_layers}L × {n_heads}H × d{d_model}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, labels=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        return logits, loss


# ========== EVAL ==========

@torch.no_grad()
def evaluate(model, val_data, seq_len, device, batch_size=8, n_batches=50):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, seq_len, batch_size)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        losses.append(loss.item())
    avg = sum(losses) / len(losses)
    ppl = math.exp(avg) if avg < 15 else float('inf')
    model.train()
    return ppl, avg


# ========== MAIN ==========

def main():
    device = torch.device("cpu")
    print(f"Device: {device} (CPU-only for Termux)")

    # ===== CONFIG (phone-friendly) =====
    D_MODEL = 160       # 5 heads × 32 head_dim (small!)
    N_HEADS = 5          # Perfect C5
    N_LAYERS = 4
    SEQ_LEN = 128
    BATCH_SIZE = 8
    STEPS = 5000
    EVAL_EVERY = 1000
    SEED = 42

    print("\nLoading data...")
    tok, train_data, val_data = load_tiny_shakespeare(SEQ_LEN)
    VOCAB_SIZE = tok.vocab_size

    # ===== STANDARD =====
    print("\n" + "=" * 50)
    print("STANDARD: Independent 5-head attention")
    print("=" * 50)

    torch.manual_seed(SEED)
    model_std = GPTLite(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS,
                        use_c5=False, max_seq_len=SEQ_LEN).to(device)

    ppl0_std, _ = evaluate(model_std, val_data, SEQ_LEN, device, BATCH_SIZE)
    print(f"  Pre-train PPL: {ppl0_std:.2f}")

    optimizer_std = torch.optim.AdamW(model_std.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    results_std = []

    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, SEQ_LEN, BATCH_SIZE)
        x, y = x.to(device), y.to(device)
        _, loss = model_std(x, labels=y)
        optimizer_std.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_std.parameters(), 1.0)
        optimizer_std.step()

        if step % 500 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            elapsed = time.time() - t0
            print(f"  [STD] {step}/{STEPS} | loss={loss.item():.4f} ppl={ppl:.1f} | {step/elapsed:.1f} steps/s")

        if step % EVAL_EVERY == 0:
            vp, vl = evaluate(model_std, val_data, SEQ_LEN, device, BATCH_SIZE)
            results_std.append({"step": step, "val_ppl": vp, "val_loss": vl})
            print(f"  *** [STD] Eval {step}: PPL={vp:.2f}")

    ppl_std, loss_std = evaluate(model_std, val_data, SEQ_LEN, device, BATCH_SIZE)
    print(f"  Final: PPL={ppl_std:.2f}, Loss={loss_std:.4f}")

    del model_std, optimizer_std

    # ===== C5 =====
    print("\n" + "=" * 50)
    print("φ-ATTENTION: C5-cycle coupled 5-head (cos72°)")
    print("=" * 50)

    torch.manual_seed(SEED)
    model_c5 = GPTLite(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS,
                       use_c5=True, max_seq_len=SEQ_LEN).to(device)

    ppl0_c5, _ = evaluate(model_c5, val_data, SEQ_LEN, device, BATCH_SIZE)
    print(f"  Pre-train PPL: {ppl0_c5:.2f}")

    optimizer_c5 = torch.optim.AdamW(model_c5.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    results_c5 = []

    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, SEQ_LEN, BATCH_SIZE)
        x, y = x.to(device), y.to(device)
        _, loss = model_c5(x, labels=y)
        optimizer_c5.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_c5.parameters(), 1.0)
        optimizer_c5.step()

        if step % 500 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            elapsed = time.time() - t0
            print(f"  [C5] {step}/{STEPS} | loss={loss.item():.4f} ppl={ppl:.1f} | {step/elapsed:.1f} steps/s")

        if step % EVAL_EVERY == 0:
            vp, vl = evaluate(model_c5, val_data, SEQ_LEN, device, BATCH_SIZE)
            results_c5.append({"step": step, "val_ppl": vp, "val_loss": vl})
            print(f"  *** [C5] Eval {step}: PPL={vp:.2f}")

    ppl_c5, loss_c5 = evaluate(model_c5, val_data, SEQ_LEN, device, BATCH_SIZE)
    print(f"  Final: PPL={ppl_c5:.2f}, Loss={loss_c5:.4f}")

    # ===== RESULTS =====
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    delta_ppl = ppl_c5 - ppl_std
    delta_pct = delta_ppl / ppl_std * 100
    print(f"  Standard PPL:   {ppl_std:.2f}")
    print(f"  φ-Attention PPL: {ppl_c5:.2f}")
    print(f"  ΔPPL: {delta_ppl:+.2f} ({delta_pct:+.2f}%)")
    print()
    print(f"  Standard Loss:   {loss_std:.4f}")
    print(f"  φ-Attention Loss: {loss_c5:.4f}")
    print(f"  ΔLoss: {loss_c5 - loss_std:+.4f}")

    if ppl_c5 < ppl_std:
        print("\n  ✅ φ-Attention WINS on PPL!")
    elif abs(delta_pct) < 2:
        print("\n  ➖ Roughly tied. May diverge at scale.")
    else:
        print("\n  ❌ Standard wins. Need different config or scale.")

    print("\nEval curves:")
    print(f"  {'Step':<8} {'Std PPL':>10} {'C5 PPL':>10} {'Δ':>10}")
    for r1, r2 in zip(results_std, results_c5):
        d = r2["val_ppl"] - r1["val_ppl"]
        print(f"  {r1['step']:<8} {r1['val_ppl']:>10.2f} {r2['val_ppl']:>10.2f} {d:>+10.2f}")

    # Save
    out = {
        "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                    "seq_len": SEQ_LEN, "steps": STEPS, "device": "cpu"},
        "standard": {"final_ppl": ppl_std, "final_loss": loss_std, "curve": results_std},
        "phi_attention": {"final_ppl": ppl_c5, "final_loss": loss_c5, "curve": results_c5},
        "delta_ppl_pct": delta_pct
    }
    with open("phi_attention_termux_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to phi_attention_termux_results.json")


if __name__ == "__main__":
    main()
