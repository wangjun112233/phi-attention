#!/usr/bin/env python3
"""
φ-Attention vs Standard Attention: Training Comparison
=======================================================
Google Colab T4 GPU compatible

Trains two small GPT-like models from scratch on Wikitext-2:
1. Standard: independent multi-head attention (5 heads)
2. φ-Attention: C5-cycle coupled attention (5 heads, cos72° coupling)

Both models have identical architecture except for the coupling matrix.
The C5 coupling is zero-parameter (fixed, non-trainable).

C5 coupling is applied to attention SCORES (before softmax), matching
the original φ-Attention formulation:
  coupled_scores = A @ raw_scores  (across heads)
  A = I + cos72° * C5-cycle adjacency
  λ_max(A) = φ (golden ratio, algebraically necessary)

Expected runtime on T4: ~45 minutes total

Usage on Colab:
1. Runtime → Change runtime type → T4 GPU
2. !pip install datasets transformers
3. Upload this script or copy-paste into a cell
4. Run
"""

import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ========== C5 COUPLING ==========

PHI = (1 + math.sqrt(5)) / 2
COS72 = math.cos(math.radians(72))


def build_c5_coupling(n_heads: int) -> torch.Tensor:
    """Build C5-cycle coupling matrix.

    For n_heads=5: I + cos72° * C5-cycle adjacency
    Max eigenvalue = φ (golden ratio), algebraically necessary.
    cos72° is the UNIQUE weight that achieves this.

    For n_heads % 5 != 0: block-diagonal with C5 groups + identity for remainder.
    """
    cos72 = COS72

    # 5-cycle adjacency with cos72° weights + identity
    # This gives λ_max = 1 + 2*cos72° = 1 + 2/(1+√5) = φ
    C5 = torch.eye(5) + cos72 * torch.tensor([
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

    # Verify eigenvalues
    eigs = torch.linalg.eigvalsh(M)
    print(f"C5 coupling: {n_groups}×C5 + {n_remaining}×I | "
          f"eigenvalues: [{', '.join(f'{e:.4f}' for e in eigs)}] | "
          f"λ_max={eigs[-1]:.6f} (φ={PHI:.6f})")

    return M


# ========== MODEL ==========

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_c5=False, max_seq_len=512):
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

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .view(1, 1, max_seq_len, max_seq_len)
        )

        # C5 coupling on attention scores
        if use_c5:
            coupling = build_c5_coupling(n_heads)
            self.register_buffer('coupling_matrix', coupling)

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to multi-head: (B, T, nh, hd) -> (B, nh, T, hd)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, nh, T, T)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # ★ C5 coupling on scores (before softmax) ★
        if self.use_c5:
            scores = torch.einsum('hn,bnsq->bhsq', self.coupling_matrix, scores)

        # Causal mask
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Output
        out = torch.matmul(attn_weights, v)  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_c5=False, max_seq_len=512):
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
    def __init__(self, vocab_size, d_model=320, n_heads=5, n_layers=6,
                 use_c5=False, max_seq_len=512):
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

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Init
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        c5_label = "φ-Attention (C5)" if use_c5 else "Standard"
        print(f"GPTLite [{c5_label}]: {n_params/1e6:.1f}M params, "
              f"{n_layers} layers, {n_heads} heads, d_model={d_model}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
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


# ========== DATA ==========

def load_wikitext2(tokenizer, seq_len=256):
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // seq_len) * seq_len
        result = {
            k: [concatenated[k][i:i + seq_len]
                for i in range(0, total_len, seq_len)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(group, batched=True)

    return grouped["train"], grouped["validation"], grouped["test"]


# ========== EVALUATION ==========

@torch.no_grad()
def evaluate(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)
        _, loss = model(input_ids, labels=labels)
        n = (labels != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 15 else float('inf')
    model.train()
    return ppl, avg_loss


# ========== TRAINING ==========

def train(model, train_ds, val_ds, device, steps=10000, lr=3e-4,
          eval_every=2000, batch_size=16):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    data_iter = iter(train_loader)

    best_val_ppl = float('inf')
    results = []
    t0 = time.time()

    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)

        _, loss = model(input_ids, labels=labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            elapsed = time.time() - t0
            sps = step / elapsed
            c5_tag = "[C5]" if model.use_c5 else "[STD]"
            print(f"  {c5_tag} Step {step}/{steps} | "
                  f"Loss: {loss.item():.4f} | PPL: {ppl:.2f} | "
                  f"{sps:.1f} steps/s")

        if step % eval_every == 0:
            val_ppl, val_loss = evaluate(model, val_ds, device, batch_size)
            results.append({"step": step, "val_ppl": val_ppl, "val_loss": val_loss})
            c5_tag = "C5" if model.use_c5 else "STD"
            print(f"  *** [{c5_tag}] Eval step {step}: PPL={val_ppl:.2f} Loss={val_loss:.4f}")
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
            model.train()

    return results, best_val_ppl


# ========== MAIN ==========

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ===== CONFIG =====
    VOCAB_SIZE = 50257  # GPT-2 tokenizer
    D_MODEL = 320       # 5 heads × 64 head_dim
    N_HEADS = 5         # Clean C5 coupling
    N_LAYERS = 6
    SEQ_LEN = 256
    BATCH_SIZE = 16
    STEPS = 10000
    EVAL_EVERY = 2000
    SEED = 42

    # Tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Data
    print("\nLoading Wikitext-2...")
    train_ds, val_ds, test_ds = load_wikitext2(tokenizer, seq_len=SEQ_LEN)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ===== STANDARD MODEL =====
    print("\n" + "=" * 60)
    print("STANDARD: Independent multi-head attention (5 heads)")
    print("=" * 60)

    torch.manual_seed(SEED)
    model_std = GPTLite(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS,
                        use_c5=False, max_seq_len=SEQ_LEN).to(device)

    ppl_std_before, _ = evaluate(model_std, test_ds, device, BATCH_SIZE)
    print(f"Before training: PPL={ppl_std_before:.2f}")

    results_std, best_std = train(
        model_std, train_ds, val_ds, device,
        steps=STEPS, batch_size=BATCH_SIZE, eval_every=EVAL_EVERY
    )

    ppl_std_after, loss_std = evaluate(model_std, test_ds, device, BATCH_SIZE)
    print(f"\nFinal test: PPL={ppl_std_after:.2f}, Loss={loss_std:.4f}")

    del model_std
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== C5 MODEL =====
    print("\n" + "=" * 60)
    print("φ-ATTENTION: C5-cycle coupled attention (5 heads, cos72°)")
    print("=" * 60)

    torch.manual_seed(SEED)
    model_c5 = GPTLite(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS,
                       use_c5=True, max_seq_len=SEQ_LEN).to(device)

    ppl_c5_before, _ = evaluate(model_c5, test_ds, device, BATCH_SIZE)
    print(f"Before training: PPL={ppl_c5_before:.2f}")

    results_c5, best_c5 = train(
        model_c5, train_ds, val_ds, device,
        steps=STEPS, batch_size=BATCH_SIZE, eval_every=EVAL_EVERY
    )

    ppl_c5_after, loss_c5 = evaluate(model_c5, test_ds, device, BATCH_SIZE)
    print(f"\nFinal test: PPL={ppl_c5_after:.2f}, Loss={loss_c5:.4f}")

    # ===== COMPARISON =====
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Standard':>12} {'φ-Attention':>12} {'Δ':>12}")
    print("-" * 60)
    print(f"{'Pre-training PPL':<25} {ppl_std_before:>12.2f} {ppl_c5_before:>12.2f} "
          f"{ppl_c5_before - ppl_std_before:>+12.2f}")
    print(f"{'Post-training PPL':<25} {ppl_std_after:>12.2f} {ppl_c5_after:>12.2f} "
          f"{ppl_c5_after - ppl_std_after:>+12.2f}")
    print(f"{'Post-training Loss':<25} {loss_std:>12.4f} {loss_c5:>12.4f} "
          f"{loss_c5 - loss_std:>+12.4f}")
    print(f"{'Best Val PPL':<25} {best_std:>12.2f} {best_c5:>12.2f} "
          f"{best_c5 - best_std:>+12.2f}")

    delta_pct = (ppl_c5_after - ppl_std_after) / ppl_std_after * 100
    print(f"\nΔPPL: {delta_pct:+.2f}% (negative = φ-Attention wins)")

    if ppl_c5_after < ppl_std_after:
        print("✅ φ-Attention WINS! C5 coupling improves language modeling.")
    elif abs(delta_pct) < 1:
        print("➖ Roughly tied. Effect may emerge at larger scale.")
    else:
        print("❌ Standard wins. C5 coupling may need scale or different hyperparameters.")

    # Eval curves
    print("\nEval curves (Val PPL):")
    print(f"{'Step':<10} {'Std PPL':>12} {'C5 PPL':>12} {'Δ':>12}")
    print("-" * 45)
    for r1, r2 in zip(results_std, results_c5):
        delta = r2["val_ppl"] - r1["val_ppl"]
        print(f"{r1['step']:<10} {r1['val_ppl']:>12.2f} {r2['val_ppl']:>12.2f} {delta:>+12.2f}")

    # Save results
    results_data = {
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "steps": STEPS,
            "seed": SEED
        },
        "standard": {
            "pre_ppl": ppl_std_before, "post_ppl": ppl_std_after,
            "post_loss": loss_std, "best_val_ppl": best_std,
            "eval_curve": results_std
        },
        "phi_attention": {
            "pre_ppl": ppl_c5_before, "post_ppl": ppl_c5_after,
            "post_loss": loss_c5, "best_val_ppl": best_c5,
            "eval_curve": results_c5
        },
        "delta_ppl_pct": delta_pct
    }
    with open("phi_attention_colab_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to phi_attention_colab_results.json")


if __name__ == "__main__":
    main()
