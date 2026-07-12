"""
φ-Attention: 5-head C5-cycle coupled attention mechanism
验证C5容器在算法尺度的有效性。
纯numpy/scipy实现，无需GPU。

Usage:
  python phi_attention_module.py  # 自检
"""

import numpy as np
from typing import Optional

PHI = (1 + np.sqrt(5)) / 2
COS72 = np.cos(np.radians(72))


def build_cycle_adjacency(n: int, coupling_weight: float) -> np.ndarray:
    A = np.eye(n)
    for i in range(n):
        A[i][(i + 1) % n] = coupling_weight
        A[i][(i - 1) % n] = coupling_weight
    return A


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class StandardMultiHeadAttention:
    def __init__(self, n_heads: int, d_model: int, seed: int = 42):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0
        self.rng = np.random.RandomState(seed)
        self._init_weights()

    def _init_weights(self):
        scale = np.sqrt(2.0 / (self.d_model + self.d_k))
        self.W_Q = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_K = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_V = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_O = self.rng.randn(self.d_model, self.d_model) * scale

    def forward(self, X: np.ndarray):
        B, S, D = X.shape
        d_k = self.d_k
        Q = (X @ self.W_Q).reshape(B, S, self.n_heads, d_k).transpose(0, 2, 1, 3)
        K = (X @ self.W_K).reshape(B, S, self.n_heads, d_k).transpose(0, 2, 1, 3)
        V = (X @ self.W_V).reshape(B, S, self.n_heads, d_k).transpose(0, 2, 1, 3)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        attn_weights = stable_softmax(scores, axis=-1)
        attn_out = np.matmul(attn_weights, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return attn_out @ self.W_O, attn_weights


class PhiAttention:
    def __init__(self, n_heads: int, d_model: int,
                 coupling_weight: float = COS72, seed: int = 42):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.coupling_weight = coupling_weight
        assert d_model % n_heads == 0
        self.A = build_cycle_adjacency(n_heads, coupling_weight)
        self.rng = np.random.RandomState(seed)
        self._init_weights()

    def _init_weights(self):
        scale = np.sqrt(2.0 / (self.d_model + self.d_k))
        self.W_Q = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_K = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_V = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_O = self.rng.randn(self.d_model, self.d_model) * scale

    def forward(self, X: np.ndarray):
        B, S, D = X.shape
        d_k = self.d_k
        n = self.n_heads
        Q = (X @ self.W_Q).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
        K = (X @ self.W_K).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
        V = (X @ self.W_V).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
        raw_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        coupled_scores = np.einsum('hn,bnsq->bhsq', self.A, raw_scores)
        attn_weights = stable_softmax(coupled_scores, axis=-1)
        attn_out = np.matmul(attn_weights, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return attn_out @ self.W_O, attn_weights


if __name__ == '__main__':
    print("φ-Attention 自检")
    d_model = 640
    rng = np.random.RandomState(0)
    X = rng.randn(2, 32, d_model) * 0.1
    for name, cls, kwargs in [
        ('phi5', PhiAttention, {'n_heads': 5, 'd_model': d_model}),
        ('std5', StandardMultiHeadAttention, {'n_heads': 5, 'd_model': d_model}),
        ('std8', StandardMultiHeadAttention, {'n_heads': 8, 'd_model': d_model}),
        ('phi8', PhiAttention, {'n_heads': 8, 'd_model': d_model}),
    ]:
        model = cls(**kwargs)
        output, attn = model.forward(X)
        print(f"{name}: output={output.shape}, attn={attn.shape}, "
              f"output_mean={output.mean():.6f}")
