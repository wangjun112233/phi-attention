"""
φ-Attention: 5-head C5-cycle coupled attention mechanism

验证C5容器在算法尺度的有效性。
- 5个head通过5-cycle邻接矩阵耦合，耦合权重cos72°=(1-φ)/2≈0.309
- 对照：标准8-head/5-head无耦合、8-head用8-cycle耦合

纯numpy/scipy实现，无需GPU。

Author: φ-Attention Prototype
Date: 2026-07-12
"""

import numpy as np
from typing import Optional

# ─── 黄金比例常量 ───────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2          # φ ≈ 1.618
COS72 = np.cos(np.radians(72))      # cos72° ≈ 0.309 = (1-φ)/2


# ─── 邻接矩阵构建 ────────────────────────────────────────────────
def build_cycle_adjacency(n: int, coupling_weight: float) -> np.ndarray:
    """构建n-cycle邻接矩阵（自环=1，相邻=权重，其余=0）"""
    A = np.eye(n)
    for i in range(n):
        A[i][(i + 1) % n] = coupling_weight
        A[i][(i - 1) % n] = coupling_weight
    return A


def build_c5_adjacency(coupling_weight: float = COS72) -> np.ndarray:
    """构建C5-5-cycle邻接矩阵，默认耦合权重=cos72°≈0.309"""
    return build_cycle_adjacency(5, coupling_weight)


def build_c8_adjacency(coupling_weight: float = COS72) -> np.ndarray:
    """构建C8-8-cycle邻接矩阵，用于验证n>5弥散"""
    return build_cycle_adjacency(8, coupling_weight)


# ─── Softmax ────────────────────────────────────────────────────
def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ─── 标准多头注意力（无耦合） ─────────────────────────────────────
class StandardMultiHeadAttention:
    """标准多头注意力，head间无耦合（标准Transformer行为）"""

    def __init__(self, n_heads: int, d_model: int, seed: int = 42):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.rng = np.random.RandomState(seed)
        self._init_weights()

    def _init_weights(self):
        scale = np.sqrt(2.0 / (self.d_model + self.d_k))
        # W_Q, W_K, W_V: (d_model, d_model) = 各head拼接
        self.W_Q = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_K = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_V = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_O = self.rng.randn(self.d_model, self.d_model) * scale

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        B, S, D = X.shape
        d_k = self.d_k

        Q = X @ self.W_Q  # (B, S, D)
        K = X @ self.W_K
        V = X @ self.W_V

        # reshape to (B, S, n_heads, d_k) -> (B, n_heads, S, d_k)
        Q = Q.reshape(B, S, self.n_heads, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, self.n_heads, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, self.n_heads, d_k).transpose(0, 2, 1, 3)

        # attention scores: (B, n_heads, S, S)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        attn_weights = stable_softmax(scores, axis=-1)

        # attention output: (B, n_heads, S, d_k)
        attn_out = np.matmul(attn_weights, V)

        # concat heads: (B, S, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
        output = attn_out @ self.W_O

        return output, attn_weights  # attn_weights: (B, n_heads, S, S)


# ─── φ-Attention: C5-cycle耦合多头注意力 ──────────────────────────
class PhiAttention:
    """
    φ-Attention: head间通过n-cycle邻接矩阵耦合
    
    核心机制：每个head的attention score在softmax前，
    叠加相邻head的attention score（加权coupling_weight），
    模拟C5中sector间d场的干涉。
    """

    def __init__(self, n_heads: int, d_model: int,
                 coupling_weight: float = COS72,
                 seed: int = 42):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.coupling_weight = coupling_weight
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        # 构建邻接矩阵
        self.A = build_cycle_adjacency(n_heads, coupling_weight)

        self.rng = np.random.RandomState(seed)
        self._init_weights()

    def _init_weights(self):
        scale = np.sqrt(2.0 / (self.d_model + self.d_k))
        self.W_Q = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_K = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_V = self.rng.randn(self.d_model, self.d_model) * scale
        self.W_O = self.rng.randn(self.d_model, self.d_model) * scale

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (batch, seq_len, d_model)
        returns: output (B, S, d_model), attn_weights (B, n_heads, S, S)
        """
        B, S, D = X.shape
        d_k = self.d_k
        n = self.n_heads

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q = Q.reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, n, d_k).transpose(0, 2, 1, 3)

        # 原始attention scores: (B, n_heads, S, S)
        raw_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        # ─── C5耦合：softmax前叠加相邻head的scores ───
        # A[i][j] 表示head i 叠加 head j 的权重
        # coupled_scores[h] = sum_j A[h][j] * raw_scores[j]
        # 用einsum: (n, n) x (B, n, S, S) -> (B, n, S, S)
        coupled_scores = np.einsum('hn,bnsq->bhsq', self.A, raw_scores)

        # softmax得到耦合后的注意力权重
        attn_weights = stable_softmax(coupled_scores, axis=-1)

        # attention output
        attn_out = np.matmul(attn_weights, V)

        # concat heads
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
        output = attn_out @ self.W_O

        return output, attn_weights


# ─── 工厂函数 ─────────────────────────────────────────────────────
def create_model(model_type: str, d_model: int = 640, seed: int = 42,
                 coupling_weight: float = COS72):
    """
    创建指定类型的注意力模型
    
    model_type:
      - 'phi5':    5-head φ-Attention (C5-cycle耦合, w=cos72°)
      - 'phi8':    8-head φ-Attention (C8-cycle耦合, 验证弥散)
      - 'std8':    标准8-head注意力 (无耦合)
      - 'std5':    标准5-head注意力 (消融对照)
    """
    models = {
        'phi5': lambda: PhiAttention(5, d_model, coupling_weight=COS72, seed=seed),
        'phi8': lambda: PhiAttention(8, d_model, coupling_weight=COS72, seed=seed),
        'std8': lambda: StandardMultiHeadAttention(8, d_model, seed=seed),
        'std5': lambda: StandardMultiHeadAttention(5, d_model, seed=seed),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}, choose from {list(models.keys())}")
    return models[model_type]()


# ─── 测试指标 ─────────────────────────────────────────────────────

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个概率分布间的Jensen-Shannon散度"""
    from scipy.spatial.distance import jensenshannon
    return jensenshannon(p, q) ** 2  # JSD = JS距离的平方


def long_range_consistency(attn_weights: np.ndarray) -> dict:
    """
    测量不同head对同一位置关注一致性（Jensen-Shannon散度）
    
    attn_weights: (B, n_heads, S, S)
    
    返回:
      - mean_jsd: 所有head对间的平均JSD（越低越一致）
      - std_jsd: JSD标准差
      - pairwise_jsd: (n_heads, n_heads) 成对JSD矩阵
    """
    B, H, S, _ = attn_weights.shape
    # 对batch维度取平均，得到每个head的平均注意力分布
    # attn_avg[h]: (S, S) - head h在所有batch上的平均注意力
    attn_avg = attn_weights.mean(axis=0)  # (H, S, S)

    n_pairs = 0
    total_jsd = 0.0
    pairwise_jsd = np.zeros((H, H))

    for i in range(H):
        for j in range(i + 1, H):
            # 对每个query位置计算JSD，然后取平均
            jsds = []
            for pos in range(S):
                p = attn_avg[i, pos, :]  # head i 对位置pos的注意力分布
                q = attn_avg[j, pos, :]  # head j 对位置pos的注意力分布
                # 确保是合法概率分布（加小量避免零）
                p = p + 1e-10
                q = q + 1e-10
                p = p / p.sum()
                q = q / q.sum()
                jsds.append(jensen_shannon_divergence(p, q))
            mean_jsd_ij = np.mean(jsds)
            pairwise_jsd[i, j] = mean_jsd_ij
            pairwise_jsd[j, i] = mean_jsd_ij
            total_jsd += mean_jsd_ij
            n_pairs += 1

    mean_jsd = total_jsd / n_pairs if n_pairs > 0 else 0.0
    # 标准差
    all_jsds = [pairwise_jsd[i, j] for i in range(H) for j in range(i + 1, H)]
    std_jsd = np.std(all_jsds) if all_jsds else 0.0

    return {
        'mean_jsd': mean_jsd,
        'std_jsd': std_jsd,
        'pairwise_jsd': pairwise_jsd,
        'n_pairs': n_pairs,
    }


def effective_rank(attn_weights: np.ndarray, threshold_ratio: float = 0.01) -> dict:
    """
    计算每个head的有效秩（参与比）
    
    对每个head的注意力矩阵做SVD，计算满足 λ_i/λ_1 > threshold_ratio 的奇异值数量
    
    attn_weights: (B, n_heads, S, S)
    
    返回:
      - per_head_erank: 每个head的有效秩
      - mean_erank: 平均有效秩
      - participation_ratio: 参与比 = erank / S
    """
    B, H, S, _ = attn_weights.shape
    # batch维度取平均
    attn_avg = attn_weights.mean(axis=0)  # (H, S, S)

    per_head_erank = []
    for h in range(H):
        U, s, Vt = np.linalg.svd(attn_avg[h], full_matrices=False)
        # 有效秩：s_i / s_0 > threshold_ratio 的个数
        s_normalized = s / (s[0] + 1e-10)
        erank = np.sum(s_normalized > threshold_ratio)
        per_head_erank.append(float(erank))

    per_head_erank = np.array(per_head_erank)
    mean_erank = float(np.mean(per_head_erank))
    participation_ratio = mean_erank / S

    return {
        'per_head_erank': per_head_erank,
        'mean_erank': mean_erank,
        'participation_ratio': participation_ratio,
    }


def vitality_sweep(model_type: str, d_model: int = 640,
                   n_sequences: int = 50, seq_len: int = 128,
                   seed: int = 42) -> dict:
    """
    退化测试：扫描耦合权重从0到1，测量活力度曲线
    
    只适用于phi类型模型
    
    返回:
      - weights: 耦合权重数组
      - vitality_scores: 对应的活力度指标
    """
    if not model_type.startswith('phi'):
        return {'error': 'vitality_sweep only applies to phi-type models'}

    n_heads = 5 if '5' in model_type else 8
    weights = np.linspace(0, 1, 51)
    vitality_scores = []

    rng = np.random.RandomState(seed)

    for w in weights:
        model = PhiAttention(n_heads, d_model, coupling_weight=w, seed=seed)
        # 生成随机序列
        X = rng.randn(1, seq_len, d_model) * 0.1

        output, attn_weights = model.forward(X)

        # 活力度指标：head间JSD + 有效秩的加权组合
        # 活力度 = 一致性(1/JSD) × 效率(参与比)
        consistency = long_range_consistency(attn_weights)
        efficiency = effective_rank(attn_weights)

        # 活力度 V = (1 - normalized_JSD) × participation_ratio
        # 归一化JSD到[0,1]范围（经验上限约0.2）
        norm_jsd = min(consistency['mean_jsd'] / 0.2, 1.0)
        V = (1 - norm_jsd) * efficiency['participation_ratio']
        vitality_scores.append(float(V))

    return {
        'weights': weights,
        'vitality_scores': np.array(vitality_scores),
        'peak_weight': float(weights[np.argmax(vitality_scores)]),
        'peak_vitality': float(np.max(vitality_scores)),
    }


if __name__ == '__main__':
    # 快速自测
    print("=" * 60)
    print("φ-Attention 自检")
    print("=" * 60)

    d_model = 640  # LCM(5,8)=40, 640=16×40
    seq_len = 32
    batch = 2

    rng = np.random.RandomState(0)
    X = rng.randn(batch, seq_len, d_model) * 0.1

    for name in ['phi5', 'std8', 'std5', 'phi8']:
        model = create_model(name, d_model=d_model, seed=42)
        output, attn = model.forward(X)
        print(f"\n{name}: output={output.shape}, attn={attn.shape}")
        print(f"  output mean={output.mean():.6f}, std={output.std():.6f}")
        print(f"  attn sum per head (last token): {attn[0, :, -1, :].sum(axis=-1)}")
