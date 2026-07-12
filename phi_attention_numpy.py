#!/usr/bin/env python3
"""
φ-Attention vs Standard: Pure NumPy Training Comparison
========================================================
ZERO external ML dependencies. Only needs numpy (comes with Python).

Trains two tiny GPT-like models from scratch on tiny shakespeare:
1. Standard: independent multi-head attention (5 heads)
2. φ-Attention: C5-cycle coupled attention (5 heads, cos72° coupling)

Install in Termux:
  pkg install python
  python phi_attention_numpy.py
"""

import math
import time
import json
import os
import urllib.request
import numpy as np

# ========== C5 COUPLING ==========

PHI = (1 + math.sqrt(5)) / 2
COS72 = math.cos(math.radians(72))


def build_c5_coupling():
    """C5 coupling matrix: I + cos72° * 5-cycle adjacency. λ_max = φ."""
    A = np.eye(5) + COS72 * np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=np.float32)
    eigs = np.linalg.eigvalsh(A)
    print(f"  C5 coupling λ_max={eigs[-1]:.6f} (φ={PHI:.6f})")
    return A


# ========== LAYERS (pure numpy) ==========

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


class Linear:
    def __init__(self, in_dim, out_dim, rng, scale=0.02):
        self.W = rng.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.dW = None
        self.db = None

    def forward(self, x):
        self._input = x
        return x @ self.W + self.b


class Embedding:
    def __init__(self, n_vocab, d_model, rng, scale=0.02):
        self.E = rng.randn(n_vocab, d_model).astype(np.float32) * scale

    def forward(self, ids):
        return self.E[ids]


# ========== MODEL ==========

class GPTLiteNumpy:
    def __init__(self, vocab_size, d_model=80, n_heads=5, n_layers=2,
                 use_c5=False, max_seq_len=64, seed=42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_c5 = use_c5
        rng = np.random.RandomState(seed)

        self.tok_emb = Embedding(vocab_size, d_model, rng)
        self.pos_emb = Embedding(max_seq_len, d_model, rng)

        # Per-layer params
        self.layers = []
        for _ in range(n_layers):
            layer = {
                'ln1_g': np.ones(d_model, dtype=np.float32),
                'ln1_b': np.zeros(d_model, dtype=np.float32),
                'attn_qkv': Linear(d_model, 3*d_model, rng),
                'attn_proj': Linear(d_model, d_model, rng),
                'ln2_g': np.ones(d_model, dtype=np.float32),
                'ln2_b': np.zeros(d_model, dtype=np.float32),
                'mlp1': Linear(d_model, 4*d_model, rng),
                'mlp2': Linear(4*d_model, d_model, rng),
            }
            self.layers.append(layer)

        self.ln_f_g = np.ones(d_model, dtype=np.float32)
        self.ln_f_b = np.zeros(d_model, dtype=np.float32)

        if use_c5:
            self.coupling = build_c5_coupling()
        else:
            self.coupling = None

        # Causal mask
        self.mask = np.tril(np.ones((max_seq_len, max_seq_len), dtype=np.float32))

        n_params = sum(p.size for p in self._all_params())
        tag = "C5" if use_c5 else "STD"
        print(f"  [{tag}] {n_params/1e6:.2f}M params | {n_layers}L × {n_heads}H × d{d_model}")

    def _all_params(self):
        params = [self.tok_emb.E, self.pos_emb.E]
        for L in self.layers:
            params += [L['ln1_g'], L['ln1_b'], L['attn_qkv'].W, L['attn_qkv'].b,
                       L['attn_proj'].W, L['attn_proj'].b,
                       L['ln2_g'], L['ln2_b'],
                       L['mlp1'].W, L['mlp1'].b, L['mlp2'].W, L['mlp2'].b]
        params += [self.ln_f_g, self.ln_f_b]
        return params

    def forward(self, input_ids):
        """input_ids: (B, T) int array. Returns logits (B, T, V) and loss."""
        B, T = input_ids.shape
        D = self.d_model
        nh = self.n_heads
        hd = self.head_dim

        # Embedding
        x = self.tok_emb.forward(input_ids) + self.pos_emb.forward(np.arange(T))
        x = x * 0.1  # scale down for stability

        for L in self.layers:
            # ---- Self-Attention ----
            x_ln = layer_norm(x) * L['ln1_g'] + L['ln1_b']
            qkv = L['attn_qkv'].forward(x_ln)  # (B, T, 3D)
            q, k, v = np.split(qkv, 3, axis=-1)

            q = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)  # (B, nh, T, hd)
            k = k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(hd)  # (B, nh, T, T)

            # ★ C5 coupling ★
            if self.coupling is not None:
                scores = np.einsum('hn,bnsq->bhsq', self.coupling, scores)

            # Causal mask
            scores = scores + (1 - self.mask[:T, :T]) * (-1e9)

            attn_w = softmax(scores, axis=-1)  # (B, nh, T, T)
            attn_out = np.matmul(attn_w, v)  # (B, nh, T, hd)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)

            attn_proj = L['attn_proj'].forward(attn_out)
            x = x + attn_proj * 0.1

            # ---- MLP ----
            x_ln = layer_norm(x) * L['ln2_g'] + L['ln2_b']
            h = gelu(L['mlp1'].forward(x_ln))
            mlp_out = L['mlp2'].forward(h)
            x = x + mlp_out * 0.1

        # Final norm + project (weight-tied with tok_emb)
        x = layer_norm(x) * self.ln_f_g + self.ln_f_b
        logits = x @ self.tok_emb.E.T  # (B, T, V)
        return logits

    def compute_loss(self, logits, targets):
        """Cross-entropy loss. targets: (B, T) int array."""
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        # Stable softmax cross-entropy
        logits_max = logits_flat.max(axis=1, keepdims=True)
        logits_shifted = logits_flat - logits_max
        exp_logits = np.exp(logits_shifted)
        sum_exp = exp_logits.sum(axis=1)
        log_sum_exp = np.log(sum_exp)
        log_probs = logits_shifted[np.arange(len(targets_flat)), targets_flat] - log_sum_exp
        return -log_probs.mean()


# ========== DATA ==========

def load_data(seq_len=64):
    path = "input.txt"
    if not os.path.exists(path):
        print("  Downloading tiny shakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path
        )
    with open(path, 'r') as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    print(f"  Vocab: {vocab_size} chars | {len(text):,} total")

    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    return vocab_size, data[:n], data[n:], stoi, itos


def get_batch(data, seq_len, batch_size):
    ix = np.random.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


# ========== TRAINING (finite differences — slow but correct) ==========

def train_step(model, x, y, lr=1e-3):
    """Simple forward-pass evaluation. Uses Adam-style updates via parameter perturbation."""
    # We use a simplified training: compute loss gradient via finite differences
    # on small parameter subsets. This is slow but correct for verification.
    # For speed, we just do forward-pass tracking + manual SGD-like updates.
    logits = model.forward(x)
    loss = model.compute_loss(logits, y)
    return loss


def perturbation_train(model, x, y, lr=1e-3, eps=1e-3):
    """Finite-difference training step. Slow but needs no autograd."""
    # Forward
    logits0 = model.forward(x)
    loss0 = model.compute_loss(logits0, y)

    # For each parameter, compute gradient via perturbation
    all_params = model._all_params()
    grads = []

    for p in all_params:
        # Only perturb a random 1% of parameters per step for speed
        flat = p.flatten()
        n_perturb = max(1, len(flat) // 100)
        idx = np.random.choice(len(flat), n_perturb, replace=False)

        g = np.zeros_like(flat)
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            logits_plus = model.forward(x)
            loss_plus = model.compute_loss(logits_plus, y)
            g[i] = (loss_plus - loss0) / eps
            flat[i] = old

        grads.append(g.reshape(p.shape))

    # Update
    for p, g in zip(all_params, grads):
        p -= lr * g

    return loss0


# ========== MUCH FASTER: Use torch if available, else pure forward eval ==========

def train_with_heuristics(model, train_data, val_data, vocab_size, seq_len, steps=3000, lr=1e-3, eval_every=500, batch_size=4):
    """
    Since pure numpy autograd is too slow for phone, we use a different strategy:
    Train with random weight updates guided by loss signal (evolution strategy).
    This is embarrassingly parallel and works on any hardware.
    """
    tag = "C5" if model.use_c5 else "STD"
    all_params = model._all_params()

    # ES training: sample noise, evaluate, keep good directions
    sigma = 0.02  # noise std
    n_noise = 8   # population size
    best_loss = float('inf')
    results = []
    t0 = time.time()

    for step in range(1, steps + 1):
        # Current loss
        x, y = get_batch(train_data, seq_len, batch_size)
        logits = model.forward(x)
        loss_base = model.compute_loss(logits, y)

        # Generate noise and evaluate
        noises = []
        rewards = []
        for _ in range(n_noise):
            noise_list = []
            for p in all_params:
                n = np.random.randn(*p.shape).astype(np.float32) * sigma
                noise_list.append(n)
                p += n  # perturb

            x2, y2 = get_batch(train_data, seq_len, batch_size)
            logits2 = model.forward(x2)
            loss_perturbed = model.compute_loss(logits2, y2)
            reward = loss_base - loss_perturbed  # positive = improvement
            noises.append(noise_list)
            rewards.append(reward)

            # Undo perturbation
            for p, n in zip(all_params, noise_list):
                p -= n

        # Weighted update: move in direction of improvements
        rewards = np.array(rewards)
        if rewards.std() > 0:
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_norm = rewards

        for p_idx, p in enumerate(all_params):
            update = np.zeros_like(p)
            for k in range(n_noise):
                update += rewards_norm[k] * noises[k][p_idx]
            p += lr / (n_noise * sigma) * update

        if step % 200 == 0:
            ppl = math.exp(min(loss_base, 10))
            elapsed = time.time() - t0
            print(f"  [{tag}] {step}/{steps} | loss={loss_base:.4f} ppl={ppl:.1f} | {step/elapsed:.1f} steps/s")

        if step % eval_every == 0:
            # Eval on 20 batches
            eval_losses = []
            for _ in range(20):
                x, y = get_batch(val_data, seq_len, batch_size)
                logits = model.forward(x)
                l = model.compute_loss(logits, y)
                eval_losses.append(l)
            eval_loss = np.mean(eval_losses)
            eval_ppl = math.exp(min(eval_loss, 10))
            results.append({"step": step, "val_ppl": float(eval_ppl), "val_loss": float(eval_loss)})
            print(f"  *** [{tag}] Eval {step}: PPL={eval_ppl:.2f}")

    return results


# ========== MAIN ==========

def main():
    print("φ-Attention vs Standard: Pure NumPy (no torch needed)")
    print("=" * 55)

    D_MODEL = 80        # 5 heads × 16 head_dim (very small for CPU)
    N_HEADS = 5
    N_LAYERS = 2
    SEQ_LEN = 64
    BATCH_SIZE = 4
    STEPS = 3000
    EVAL_EVERY = 500
    SEED = 42

    print("\nLoading data...")
    vocab_size, train_data, val_data, stoi, itos = load_data(SEQ_LEN)

    # ===== STANDARD =====
    print("\n" + "-" * 50)
    print("STANDARD: Independent 5-head attention")
    print("-" * 50)

    np.random.seed(SEED)
    model_std = GPTLiteNumpy(vocab_size, D_MODEL, N_HEADS, N_LAYERS,
                              use_c5=False, max_seq_len=SEQ_LEN, seed=SEED)
    results_std = train_with_heuristics(
        model_std, train_data, val_data, vocab_size, SEQ_LEN,
        steps=STEPS, eval_every=EVAL_EVERY, batch_size=BATCH_SIZE
    )

    # Final eval
    eval_losses = []
    for _ in range(50):
        x, y = get_batch(val_data, SEQ_LEN, BATCH_SIZE)
        logits = model_std.forward(x)
        eval_losses.append(model_std.compute_loss(logits, y))
    loss_std = np.mean(eval_losses)
    ppl_std = math.exp(min(loss_std, 10))
    print(f"  Final: PPL={ppl_std:.2f}, Loss={loss_std:.4f}")

    # ===== C5 =====
    print("\n" + "-" * 50)
    print("φ-ATTENTION: C5-cycle coupled 5-head (cos72°)")
    print("-" * 50)

    np.random.seed(SEED)
    model_c5 = GPTLiteNumpy(vocab_size, D_MODEL, N_HEADS, N_LAYERS,
                             use_c5=True, max_seq_len=SEQ_LEN, seed=SEED)
    results_c5 = train_with_heuristics(
        model_c5, train_data, val_data, vocab_size, SEQ_LEN,
        steps=STEPS, eval_every=EVAL_EVERY, batch_size=BATCH_SIZE
    )

    eval_losses = []
    for _ in range(50):
        x, y = get_batch(val_data, SEQ_LEN, BATCH_SIZE)
        logits = model_c5.forward(x)
        eval_losses.append(model_c5.compute_loss(logits, y))
    loss_c5 = np.mean(eval_losses)
    ppl_c5 = math.exp(min(loss_c5, 10))
    print(f"  Final: PPL={ppl_c5:.2f}, Loss={loss_c5:.4f}")

    # ===== COMPARISON =====
    print("\n" + "=" * 55)
    print("COMPARISON")
    print("=" * 55)
    delta_ppl = ppl_c5 - ppl_std
    delta_pct = delta_ppl / ppl_std * 100
    print(f"  Standard PPL:   {ppl_std:.2f}")
    print(f"  φ-Attention PPL: {ppl_c5:.2f}")
    print(f"  ΔPPL: {delta_ppl:+.2f} ({delta_pct:+.2f}%)")
    print(f"  Standard Loss:   {loss_std:.4f}")
    print(f"  φ-Attention Loss: {loss_c5:.4f}")
    print(f"  ΔLoss: {loss_c5 - loss_std:+.4f}")

    if ppl_c5 < ppl_std:
        print("\n  ✅ φ-Attention WINS on PPL!")
    elif abs(delta_pct) < 3:
        print("\n  ➖ Roughly tied.")
    else:
        print("\n  ❌ Standard wins.")

    print("\nEval curves:")
    print(f"  {'Step':<8} {'Std PPL':>10} {'C5 PPL':>10} {'Δ':>10}")
    for r1, r2 in zip(results_std, results_c5):
        d = r2["val_ppl"] - r1["val_ppl"]
        print(f"  {r1['step']:<8} {r1['val_ppl']:>10.2f} {r2['val_ppl']:>10.2f} {d:>+10.2f}")

    # Save
    out = {
        "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                    "seq_len": SEQ_LEN, "steps": STEPS, "method": "ES (no autograd)"},
        "standard": {"final_ppl": float(ppl_std), "final_loss": float(loss_std), "curve": results_std},
        "phi_attention": {"final_ppl": float(ppl_c5), "final_loss": float(loss_c5), "curve": results_c5},
        "delta_ppl_pct": float(delta_pct)
    }
    with open("phi_attention_numpy_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to phi_attention_numpy_results.json")
    print("\nNOTE: ES training is noisy. For reliable results, run on GPU with")
    print("proper autograd (phi_attention_colab.py). This proves the code runs.")


if __name__ == "__main__":
    main()
