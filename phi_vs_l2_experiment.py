"""
φ-Attention vs L2 Regularization: Structure Effect or Regularization Effect?

关键问题：φ-Attention的优势是C5结构效应还是单纯正则化效应？

4组对照：
  (A) φ5:       5-head C5-coupled (cos72°权重) — 主实验
  (B) Std5:     5-head uncoupled — 基线
  (C) Std5-L2:  5-head uncoupled + L2正则化，扫描lambda使DOF≈4.5
  (D) Std5-DOF-drop: 5-head uncoupled + head mixing，使DOF≈4.5

判据：
  - 如果 Std5-L2 的 JSD 和谱间隙都能接近 φ5 → 正则化效应
  - 如果 Std5-L2 的 DOF≈4.5 但 JSD 和谱间隙仍显著差于 φ5 → C5结构有独特价值

纯numpy/scipy实现，无PyTorch/GPU依赖。
"""

import numpy as np
import sys
import os
import time
import json
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_attention_module import StandardMultiHeadAttention, PhiAttention, stable_softmax, COS72

PHI = (1 + np.sqrt(5)) / 2
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def sinusoidal_position_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def generate_structured_input(batch, seq_len, d_model, rng, scale=1.0):
    X = rng.randn(batch, seq_len, d_model) * scale
    pe = sinusoidal_position_encoding(seq_len, d_model)
    X = X + pe[np.newaxis, :, :]
    return X


# ═══════════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════════

def jsd_between_dists(p, q):
    """Jensen-Shannon散度（JS距离的平方）"""
    return jensenshannon(p, q) ** 2


def compute_metrics(attn_weights):
    """
    从注意力权重计算三大指标：JSD, 有效DOF, 谱间隙

    attn_weights: (B, n_heads, S, S)
    """
    B, H, S, _ = attn_weights.shape

    # ─── 1. JSD (长程一致性) ───
    attn_avg = attn_weights.mean(axis=0)  # (H, S, S)
    jsd_values = []
    for i in range(H):
        for j in range(i + 1, H):
            pos_jsds = []
            for pos in range(S):
                p = attn_avg[i, pos, :] + 1e-10
                q = attn_avg[j, pos, :] + 1e-10
                p = p / p.sum()
                q = q / q.sum()
                pos_jsds.append(jsd_between_dists(p, q))
            jsd_values.append(np.mean(pos_jsds))

    mean_jsd = float(np.mean(jsd_values))
    std_jsd = float(np.std(jsd_values))

    # ─── 2. 有效DOF + 谱间隙 (head相关矩阵) ───
    flat = attn_weights.reshape(H, -1)
    corr = np.corrcoef(flat)  # (H, H)

    eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.maximum(eigenvalues, 0)

    total = np.sum(eigenvalues)
    effective_dof = float(total ** 2 / (np.sum(eigenvalues ** 2) + 1e-10))
    spectral_gap = float((eigenvalues[0] - eigenvalues[1]) / (total + 1e-10)) if len(eigenvalues) >= 2 else 0.0
    concentration = float((eigenvalues[0] + eigenvalues[1]) / total) if total > 0 else 1.0

    # ─── 3. 注意力熵 ───
    eps = 1e-10
    p = attn_weights + eps
    entropy = -np.sum(p * np.log(p), axis=-1)
    max_entropy = np.log(S)
    norm_entropy = float(np.mean(entropy / max_entropy))

    return {
        'mean_jsd': mean_jsd,
        'std_jsd': std_jsd,
        'effective_dof': effective_dof,
        'spectral_gap': spectral_gap,
        'concentration': concentration,
        'norm_entropy': norm_entropy,
        'eigenvalues': eigenvalues.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# 模型变体
# ═══════════════════════════════════════════════════════════════════

def forward_std5_l2(model, X, l2_lambda):
    """
    标准attention + L2正则化模拟

    机制：L2 weight decay等价于在训练中缩小权重。
    对Q/K投影施加衰减：W_Q, W_K *= 1/(1 + l2_lambda * C)
    缩小Q/K → 缩小attention scores → softmax温度降低 → 更均匀注意力

    这模拟了L2正则化对注意力模式的核心效应。
    """
    decay = 1.0 / (1.0 + l2_lambda * 500.0)

    B, S, D = X.shape
    d_k = model.d_k
    n = model.n_heads

    # Q/K施加衰减（模拟L2 weight decay的核心效应）
    W_Q_decay = model.W_Q * decay
    W_K_decay = model.W_K * decay

    Q = (X @ W_Q_decay).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
    K = (X @ W_K_decay).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
    V = (X @ model.W_V).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)

    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = stable_softmax(scores, axis=-1)

    attn_out = np.matmul(attn_weights, V)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
    output = attn_out @ model.W_O

    return output, attn_weights


def forward_std5_dof_drop(model, X, mix_alpha):
    """
    标准attention + Head混合（直接削减DOF）

    机制：将一个head的注意力部分替换为其他head的平均
    attn[4] = alpha * attn[4] + (1-alpha) * mean(attn[0:4])

    alpha=1 → 5个独立head, DOF=5
    alpha=0 → head4完全由其他head决定, DOF≈4
    中间值 → DOF在4-5之间

    这是另一个正则化基线：通过直接消除head间独立性来降DOF，
    而不引入C5结构耦合。
    """
    B, S, D = X.shape
    d_k = model.d_k
    n = model.n_heads

    Q = (X @ model.W_Q).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
    K = (X @ model.W_K).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)
    V = (X @ model.W_V).reshape(B, S, n, d_k).transpose(0, 2, 1, 3)

    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = stable_softmax(scores, axis=-1)

    # Head混合：将head 4替换为自身与head 0-3均值的加权
    mean_others = attn_weights[:, :4, :, :].mean(axis=1, keepdims=True)
    attn_weights[:, 4:5, :, :] = (
        mix_alpha * attn_weights[:, 4:5, :, :] +
        (1.0 - mix_alpha) * mean_others
    )

    attn_out = np.matmul(attn_weights, V)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
    output = attn_out @ model.W_O

    return output, attn_weights


# ═══════════════════════════════════════════════════════════════════
# 实验流程
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(forward_fn, model, n_sequences, seq_len, d_model, rng,
                   **forward_kwargs):
    """对指定模型跑n_sequences条序列，返回平均指标"""
    all_metrics = []
    for i in range(n_sequences):
        X = generate_structured_input(1, seq_len, d_model, rng, scale=1.0)
        _, attn_weights = forward_fn(model, X, **forward_kwargs)
        metrics = compute_metrics(attn_weights)
        all_metrics.append(metrics)

    # 聚合
    result = {}
    for key in ['mean_jsd', 'std_jsd', 'effective_dof', 'spectral_gap',
                'concentration', 'norm_entropy']:
        vals = [m[key] for m in all_metrics]
        result[f'avg_{key}'] = float(np.mean(vals))
        result[f'std_{key}'] = float(np.std(vals))

    return result


def phase1_l2_scan(n_scan=25, n_sequences=30, seq_len=128, d_model=640, seed=42):
    """
    Phase 1: L2 lambda扫描，找到DOF≈4.5的lambda值

    扫描范围：1e-5 到 1e-2，对数均匀
    """
    print("\n" + "=" * 70)
    print("Phase 1: L2 Lambda 扫描 (寻找 DOF ≈ 4.5)")
    print("=" * 70)

    lambdas = np.logspace(-5, -2, n_scan)
    model = StandardMultiHeadAttention(n_heads=5, d_model=d_model, seed=seed)
    rng = np.random.RandomState(seed)

    scan_results = []
    for li, lam in enumerate(lambdas):
        metrics = evaluate_model(
            forward_std5_l2, model, n_sequences, seq_len, d_model, rng,
            l2_lambda=lam
        )
        scan_results.append({
            'lambda': float(lam),
            'dof': metrics['avg_effective_dof'],
            'jsd': metrics['avg_mean_jsd'],
            'spectral_gap': metrics['avg_spectral_gap'],
        })
        print(f"  λ={lam:.2e}: DOF={metrics['avg_effective_dof']:.3f}, "
              f"JSD={metrics['avg_mean_jsd']:.4f}, "
              f"谱间隙={metrics['avg_spectral_gap']:.4f}")

    # 找到DOF最接近4.5的lambda
    best_idx = int(np.argmin([abs(r['dof'] - 4.5) for r in scan_results]))
    best_lambda = scan_results[best_idx]['lambda']
    best_dof = scan_results[best_idx]['dof']

    print(f"\n  ★ 最佳L2 λ = {best_lambda:.4e} (DOF = {best_dof:.3f})")

    return lambdas, scan_results, best_lambda


def phase2_dof_drop_scan(n_scan=20, n_sequences=30, seq_len=128, d_model=640, seed=42):
    """
    Phase 2: Head混合系数扫描，找到DOF≈4.5的alpha值

    mix_alpha: 0.0 (完全混合, DOF≈4) 到 1.0 (独立, DOF=5)
    """
    print("\n" + "=" * 70)
    print("Phase 2: Head混合系数扫描 (寻找 DOF ≈ 4.5)")
    print("=" * 70)

    alphas = np.linspace(0.0, 1.0, n_scan + 2)[1:-1]  # 去掉0和1
    model = StandardMultiHeadAttention(n_heads=5, d_model=d_model, seed=seed)
    rng = np.random.RandomState(seed)

    scan_results = []
    for ai, alpha in enumerate(alphas):
        metrics = evaluate_model(
            forward_std5_dof_drop, model, n_sequences, seq_len, d_model, rng,
            mix_alpha=alpha
        )
        scan_results.append({
            'alpha': float(alpha),
            'dof': metrics['avg_effective_dof'],
            'jsd': metrics['avg_mean_jsd'],
            'spectral_gap': metrics['avg_spectral_gap'],
        })
        print(f"  α={alpha:.3f}: DOF={metrics['avg_effective_dof']:.3f}, "
              f"JSD={metrics['avg_mean_jsd']:.4f}, "
              f"谱间隙={metrics['avg_spectral_gap']:.4f}")

    # 找到DOF最接近4.5的alpha
    best_idx = int(np.argmin([abs(r['dof'] - 4.5) for r in scan_results]))
    best_alpha = scan_results[best_idx]['alpha']
    best_dof = scan_results[best_idx]['dof']

    print(f"\n  ★ 最佳混合系数 α = {best_alpha:.3f} (DOF = {best_dof:.3f})")

    return alphas, scan_results, best_alpha


def phase3_full_comparison(best_l2_lambda, best_dof_alpha,
                           n_sequences=80, seq_len=128, d_model=640, seed=42):
    """
    Phase 3: 4组完整对比

    (A) φ5:       5-head C5-coupled
    (B) Std5:     5-head uncoupled
    (C) Std5-L2:  5-head uncoupled + L2 (lambda=best_l2_lambda)
    (D) Std5-DOF-drop: 5-head uncoupled + head mixing (alpha=best_dof_alpha)
    """
    print("\n" + "=" * 70)
    print("Phase 3: 四组完整对比")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    results = {}

    # (A) φ5
    print("\n  [A] φ5: 5-head C5-coupled (cos72°)...")
    phi5_model = PhiAttention(n_heads=5, d_model=d_model, coupling_weight=COS72, seed=seed)
    rng_a = np.random.RandomState(seed)
    results['phi5'] = evaluate_model(
        lambda m, x: m.forward(x), phi5_model,
        n_sequences, seq_len, d_model, rng_a
    )
    print(f"    JSD={results['phi5']['avg_mean_jsd']:.4f}, "
          f"DOF={results['phi5']['avg_effective_dof']:.2f}, "
          f"谱间隙={results['phi5']['avg_spectral_gap']:.4f}")

    # (B) Std5
    print("\n  [B] Std5: 5-head uncoupled...")
    std5_model = StandardMultiHeadAttention(n_heads=5, d_model=d_model, seed=seed)
    rng_b = np.random.RandomState(seed)
    results['std5'] = evaluate_model(
        lambda m, x: m.forward(x), std5_model,
        n_sequences, seq_len, d_model, rng_b
    )
    print(f"    JSD={results['std5']['avg_mean_jsd']:.4f}, "
          f"DOF={results['std5']['avg_effective_dof']:.2f}, "
          f"谱间隙={results['std5']['avg_spectral_gap']:.4f}")

    # (C) Std5-L2
    print(f"\n  [C] Std5-L2: 5-head + L2 (λ={best_l2_lambda:.4e})...")
    std5_l2_model = StandardMultiHeadAttention(n_heads=5, d_model=d_model, seed=seed)
    rng_c = np.random.RandomState(seed)
    results['std5_l2'] = evaluate_model(
        forward_std5_l2, std5_l2_model,
        n_sequences, seq_len, d_model, rng_c,
        l2_lambda=best_l2_lambda
    )
    print(f"    JSD={results['std5_l2']['avg_mean_jsd']:.4f}, "
          f"DOF={results['std5_l2']['avg_effective_dof']:.2f}, "
          f"谱间隙={results['std5_l2']['avg_spectral_gap']:.4f}")

    # (D) Std5-DOF-drop
    print(f"\n  [D] Std5-DOF-drop: 5-head + head mixing (α={best_dof_alpha:.3f})...")
    std5_dof_model = StandardMultiHeadAttention(n_heads=5, d_model=d_model, seed=seed)
    rng_d = np.random.RandomState(seed)
    results['std5_dof_drop'] = evaluate_model(
        forward_std5_dof_drop, std5_dof_model,
        n_sequences, seq_len, d_model, rng_d,
        mix_alpha=best_dof_alpha
    )
    print(f"    JSD={results['std5_dof_drop']['avg_mean_jsd']:.4f}, "
          f"DOF={results['std5_dof_drop']['avg_effective_dof']:.2f}, "
          f"谱间隙={results['std5_dof_drop']['avg_spectral_gap']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════════════

def generate_report(l2_scan, dof_scan, comparison, best_l2_lambda, best_dof_alpha,
                    n_sequences, seq_len, d_model):
    """生成完整对比报告"""
    r = []
    r.append("# φ-Attention vs L2 正则化：结构效应还是正则化效应？\n")
    r.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    r.append(f"**实验参数**: n_sequences={n_sequences}, seq_len={seq_len}, d_model={d_model}\n")
    r.append("---\n")

    # ─── 1. 实验动机 ───
    r.append("## 1. 实验动机\n")
    r.append("φ-Attention (5-head C5-coupled) 在之前实验中展现出三大优势：")
    r.append("1. 更低的JSD（head间更一致）")
    r.append("2. 更低的有效DOF（信息更集中）")
    r.append("3. 更高的谱间隙（结构更刚性）\n")
    r.append("但这些优势可能来自两种不同的机制：")
    r.append("- **正则化假说**：C5耦合只是约束了模型容量（降低DOF），任何正则化手段都能达到同样效果")
    r.append("- **结构效应假说**：C5耦合引入了独特的head间信息流结构，不只是降DOF，还改变了相关模式\n")
    r.append("本实验通过引入两个正则化基线来区分这两种假说。\n")

    # ─── 2. 实验设计 ───
    r.append("## 2. 实验设计\n")
    r.append("### 2.1 四组对照\n")
    r.append("| 编号 | 模型 | head数 | 机制 | DOF调控方式 |")
    r.append("|------|------|--------|------|------------|")
    r.append("| (A) φ5 | 5-head C5-coupled | 5 | cos72°邻接矩阵耦合 | 结构效应自然降DOF |")
    r.append("| (B) Std5 | 5-head uncoupled | 5 | 无 | DOF=5（上界） |")
    r.append("| (C) Std5-L2 | 5-head + L2正则化 | 5 | Q/K权重衰减 → softmax温度降低 | 扫描λ使DOF≈4.5 |")
    r.append("| (D) Std5-DOF-drop | 5-head + head混合 | 5 | head4部分替换为其他head均值 | 扫描α使DOF≈4.5 |\n")
    r.append("### 2.2 判据\n")
    r.append("- 如果 (C) 或 (D) 的 **JSD和谱间隙** 都接近 (A) → φ-Attention优势 = 正则化效应")
    r.append("- 如果 (C) 或 (D) 的 DOF≈4.5 但 **JSD或谱间隙** 仍显著差于 (A) → C5结构有独特价值\n")
    r.append("### 2.3 L2正则化模拟机制\n")
    r.append("L2 weight decay在训练中缩小权重。对Q/K投影矩阵施加衰减：\n")
    r.append("```")
    r.append("W_Q_eff = W_Q / (1 + λ × 500)")
    r.append("W_K_eff = W_K / (1 + λ × 500)")
    r.append("```\n")
    r.append("缩小Q/K → 缩小attention scores → softmax温度降低 → 注意力更均匀 → DOF降低。\n")
    r.append("### 2.4 Head混合（DOF-drop）机制\n")
    r.append("将一个head的注意力部分替换为其他head的均值：\n")
    r.append("```")
    r.append("attn[4] = α × attn[4] + (1-α) × mean(attn[0:4])")
    r.append("```\n")
    r.append("α=1时DOF=5，α=0时DOF≈4。中间值使DOF在4-5之间。\n")

    # ─── 3. L2扫描结果 ───
    r.append("## 3. L2 Lambda 扫描\n")
    r.append(f"扫描范围：1e-5 ~ 1e-2，{len(l2_scan[1])}个对数均匀采样点\n")

    r.append("| λ | DOF | JSD | 谱间隙 |")
    r.append("|----|-----|-----|--------|")
    for sr in l2_scan[1]:
        marker = " ★" if abs(sr['dof'] - 4.5) < 0.15 else ""
        r.append(f"| {sr['lambda']:.2e} | {sr['dof']:.3f} | {sr['jsd']:.4f} | "
                 f"{sr['spectral_gap']:.4f} |{marker}")
    r.append(f"\n**最佳λ = {best_l2_lambda:.4e}** (DOF最接近4.5)\n")

    # ─── 4. DOF-drop扫描结果 ───
    r.append("## 4. Head混合系数扫描\n")
    r.append(f"扫描范围：0~1，{len(dof_scan[1])}个均匀采样点\n")

    r.append("| α | DOF | JSD | 谱间隙 |")
    r.append("|----|-----|-----|--------|")
    for sr in dof_scan[1]:
        marker = " ★" if abs(sr['dof'] - 4.5) < 0.15 else ""
        r.append(f"| {sr['alpha']:.3f} | {sr['dof']:.3f} | {sr['jsd']:.4f} | "
                 f"{sr['spectral_gap']:.4f} |{marker}")
    r.append(f"\n**最佳α = {best_dof_alpha:.3f}** (DOF最接近4.5)\n")

    # ─── 5. 核心对比 ───
    r.append("## 5. 四组完整对比\n")

    labels = {
        'phi5': '(A) φ5: 5-head C5-coupled',
        'std5': '(B) Std5: 5-head uncoupled',
        'std5_l2': '(C) Std5-L2: 5-head + L2正则化',
        'std5_dof_drop': '(D) Std5-DOF-drop: 5-head + head混合',
    }

    r.append("| 模型 | JSD ↓ | DOF | 谱间隙 ↑ | 浓度 | 归一化熵 ↓ |")
    r.append("|------|-------|-----|----------|------|------------|")
    for key in ['phi5', 'std5', 'std5_l2', 'std5_dof_drop']:
        c = comparison[key]
        r.append(f"| {labels[key]} | "
                 f"{c['avg_mean_jsd']:.4f}±{c['std_mean_jsd']:.4f} | "
                 f"{c['avg_effective_dof']:.2f}±{c['std_effective_dof']:.2f} | "
                 f"{c['avg_spectral_gap']:.4f}±{c['std_spectral_gap']:.4f} | "
                 f"{c['avg_concentration']:.4f} | "
                 f"{c['avg_norm_entropy']:.4f} |")
    r.append("")

    # ─── 6. 关键对比分析 ───
    r.append("## 6. 关键对比分析\n")

    phi5 = comparison['phi5']
    std5 = comparison['std5']
    std5_l2 = comparison['std5_l2']
    std5_dof = comparison['std5_dof_drop']

    r.append("### 6.1 DOF匹配验证\n")
    r.append("| 模型 | 有效DOF | 与φ5的DOF差 |")
    r.append("|------|---------|-------------|")
    for key, label in labels.items():
        dof_diff = comparison[key]['avg_effective_dof'] - phi5['avg_effective_dof']
        r.append(f"| {label} | {comparison[key]['avg_effective_dof']:.3f} | {dof_diff:+.3f} |")
    r.append("")

    # 判断DOF是否匹配
    l2_dof_match = abs(std5_l2['avg_effective_dof'] - phi5['avg_effective_dof']) < 0.3
    dof_drop_match = abs(std5_dof['avg_effective_dof'] - phi5['avg_effective_dof']) < 0.3

    if l2_dof_match:
        r.append(f"✅ Std5-L2的DOF({std5_l2['avg_effective_dof']:.3f})与φ5({phi5['avg_effective_dof']:.3f})匹配\n")
    else:
        r.append(f"⚠️ Std5-L2的DOF({std5_l2['avg_effective_dof']:.3f})与φ5({phi5['avg_effective_dof']:.3f})不匹配\n")

    if dof_drop_match:
        r.append(f"✅ Std5-DOF-drop的DOF({std5_dof['avg_effective_dof']:.3f})与φ5({phi5['avg_effective_dof']:.3f})匹配\n")
    else:
        r.append(f"⚠️ Std5-DOF-drop的DOF({std5_dof['avg_effective_dof']:.3f})与φ5({phi5['avg_effective_dof']:.3f})不匹配\n")

    r.append("### 6.2 JSD对比（DOF匹配条件下）\n")
    r.append("| 模型 | JSD | 与φ5的JSD差 | JSD改善(相对Std5) |")
    r.append("|------|-----|-------------|-------------------|")
    for key, label in labels.items():
        jsd_diff = comparison[key]['avg_mean_jsd'] - phi5['avg_mean_jsd']
        jsd_improve = (std5['avg_mean_jsd'] - comparison[key]['avg_mean_jsd']) / std5['avg_mean_jsd'] * 100
        r.append(f"| {label} | {comparison[key]['avg_mean_jsd']:.4f} | {jsd_diff:+.4f} | {jsd_improve:+.1f}% |")
    r.append("")

    # 判断JSD差异
    l2_jsd_gap = std5_l2['avg_mean_jsd'] - phi5['avg_mean_jsd']
    dof_jsd_gap = std5_dof['avg_mean_jsd'] - phi5['avg_mean_jsd']
    std5_jsd_gap = std5['avg_mean_jsd'] - phi5['avg_mean_jsd']
    jsd_threshold = 0.02  # 显著差异阈值

    r.append("### 6.3 谱间隙对比（DOF匹配条件下）\n")
    r.append("| 模型 | 谱间隙 | 与φ5的谱间隙差 | 谱间隙改善(相对Std5) |")
    r.append("|------|--------|----------------|---------------------|")
    for key, label in labels.items():
        sgap_diff = comparison[key]['avg_spectral_gap'] - phi5['avg_spectral_gap']
        sgap_improve = (comparison[key]['avg_spectral_gap'] - std5['avg_spectral_gap']) / (std5['avg_spectral_gap'] + 1e-10) * 100
        r.append(f"| {label} | {comparison[key]['avg_spectral_gap']:.4f} | {sgap_diff:+.4f} | {sgap_improve:+.1f}% |")
    r.append("")

    # 判断谱间隙差异
    l2_sgap_gap = phi5['avg_spectral_gap'] - std5_l2['avg_spectral_gap']
    dof_sgap_gap = phi5['avg_spectral_gap'] - std5_dof['avg_spectral_gap']
    sgap_threshold = 0.01

    r.append("### 6.4 结构效应 vs 正则化效应判断\n")

    # 综合判断
    l2_captures_jsd = l2_jsd_gap < jsd_threshold
    l2_captures_sgap = l2_sgap_gap < sgap_threshold
    dof_captures_jsd = dof_jsd_gap < jsd_threshold
    dof_captures_sgap = dof_sgap_gap < sgap_threshold

    r.append("**Std5-L2 (Q/K权重衰减)**：\n")
    r.append(f"- DOF匹配: {'✅' if l2_dof_match else '⚠️'} ({std5_l2['avg_effective_dof']:.3f} vs φ5的{phi5['avg_effective_dof']:.3f})")
    r.append(f"- JSD能否达到φ5水平: {'✅' if l2_captures_jsd else '❌'} (差{l2_jsd_gap:+.4f})")
    r.append(f"- 谱间隙能否达到φ5水平: {'✅' if l2_captures_sgap else '❌'} (差{l2_sgap_gap:+.4f})\n")

    r.append("**Std5-DOF-drop (head混合)**：\n")
    r.append(f"- DOF匹配: {'✅' if dof_drop_match else '⚠️'} ({std5_dof['avg_effective_dof']:.3f} vs φ5的{phi5['avg_effective_dof']:.3f})")
    r.append(f"- JSD能否达到φ5水平: {'✅' if dof_captures_jsd else '❌'} (差{dof_jsd_gap:+.4f})")
    r.append(f"- 谱间隙能否达到φ5水平: {'✅' if dof_captures_sgap else '❌'} (差{dof_sgap_gap:+.4f})\n")

    # ─── 7. 最终结论 ───
    r.append("## 7. 最终结论\n")

    # 判断逻辑
    both_capture = (l2_captures_jsd and l2_captures_sgap) or (dof_captures_jsd and dof_captures_sgap)
    neither_capture = (not l2_captures_jsd and not l2_captures_sgap) and (not dof_captures_jsd and not dof_captures_sgap)

    if both_capture:
        r.append("### 📊 结论：φ-Attention的优势主要是正则化效应\n")
        r.append("在DOF匹配条件下，正则化基线（L2或head混合）能够复现φ5的JSD和谱间隙优势。")
        r.append("C5-cycle耦合并未提供超越简单DOF约束的独特结构价值。\n")
    elif neither_capture:
        r.append("### 📊 结论：φ-Attention的优势是C5结构效应\n")
        r.append("即使DOF匹配到≈4.5，正则化基线仍无法复现φ5的JSD和谱间隙优势。")
        r.append("C5-cycle耦合引入了独特的head间信息流结构，不只是简单地约束DOF。\n")
        r.append("**具体证据：**\n")

        if not l2_captures_jsd:
            r.append(f"- L2正则化：JSD差{l2_jsd_gap:+.4f}（φ5更一致），说明C5结构降低了head间分歧\n")
        if not l2_captures_sgap:
            r.append(f"- L2正则化：谱间隙差{l2_sgap_gap:+.4f}（φ5更刚性），说明C5结构创造了更强的主导本征模式\n")
        if not dof_captures_jsd:
            r.append(f"- Head混合：JSD差{dof_jsd_gap:+.4f}（φ5更一致），说明C5的cycle结构比简单head平均更有效地协调注意力\n")
        if not dof_captures_sgap:
            r.append(f"- Head混合：谱间隙差{dof_sgap_gap:+.4f}（φ5更刚性），说明C5的邻接结构创造了简单混合无法实现的相关模式\n")

        r.append("\n**物理解释**：C5-cycle邻接矩阵的最大本征值=φ(黄金比例)，")
        r.append("这使得head间信息流在最主导模式上的增益恰好是φ——")
        r.append("既不过强（锁死）也不过弱（弥散）。")
        r.append("这种精确的增益结构是简单正则化无法构造的。\n")
    else:
        r.append("### 📊 结论：φ-Attention的优势是结构与正则化的混合效应\n")
        r.append("部分指标（JSD或谱间隙之一）可被正则化基线复现，")
        r.append("但另一个指标仍显著差于φ5，说明C5结构在特定维度有独特价值。\n")

        if l2_captures_jsd and not l2_captures_sgap:
            r.append("- L2能复现JSD但无法复现谱间隙：C5结构的主要独特价值在于创造刚性相关模式\n")
        elif not l2_captures_jsd and l2_captures_sgap:
            r.append("- L2能复现谱间隙但无法复现JSD：C5结构的主要独特价值在于协调head间一致性\n")

    # ─── 附录：C5邻接矩阵本征值 ───
    r.append("---\n")
    r.append("## 附录：C5-cycle邻接矩阵本征值\n")
    A = np.eye(5)
    for i in range(5):
        A[i][(i + 1) % 5] = COS72
        A[i][(i - 1) % 5] = COS72
    eigvals = np.sort(np.real(np.linalg.eigvalsh(A)))[::-1]
    r.append(f"权重=cos72°={COS72:.4f}\n")
    r.append(f"本征值: [{', '.join(f'{v:.4f}' for v in eigvals)}]")
    r.append(f"\n最大本征值 λ₀ = {eigvals[0]:.4f} ≈ φ = {PHI:.4f}")
    r.append(f"\n谱间隙 λ₀-λ₁ = {eigvals[0]-eigvals[1]:.4f}\n")

    r.append("---\n")
    r.append(f"*本报告由φ-Attention vs L2实验脚本自动生成，{time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    report_text = "\n".join(r)
    report_path = os.path.join(OUTPUT_DIR, 'phi_vs_l2_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

async def main():
    result_mode = sys.argv[1] if len(sys.argv) > 1 else "display_only"
    n_sequences = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    d_model = int(sys.argv[4]) if len(sys.argv) > 4 else 640
    seed = 42

    print(f"[参数] result_mode={result_mode}, n_sequences={n_sequences}, "
          f"seq_len={seq_len}, d_model={d_model}")

    from codeact_sdk import CodeActSDK
    sdk = CodeActSDK()

    try:
        t0 = time.time()

        # Phase 1: L2 lambda扫描
        l2_scan_results = phase1_l2_scan(
            n_scan=25, n_sequences=30, seq_len=seq_len, d_model=d_model, seed=seed
        )
        lambdas, l2_scan_data, best_l2_lambda = l2_scan_results

        # Phase 2: Head混合系数扫描
        dof_scan_results = phase2_dof_drop_scan(
            n_scan=20, n_sequences=30, seq_len=seq_len, d_model=d_model, seed=seed
        )
        alphas, dof_scan_data, best_dof_alpha = dof_scan_results

        # Phase 3: 完整四组对比
        comparison = phase3_full_comparison(
            best_l2_lambda, best_dof_alpha,
            n_sequences=n_sequences, seq_len=seq_len, d_model=d_model, seed=seed
        )

        elapsed = time.time() - t0
        print(f"\n所有实验完成，耗时 {elapsed:.1f}s")

        # 生成报告
        report_path = generate_report(
            l2_scan_results, dof_scan_results, comparison,
            best_l2_lambda, best_dof_alpha,
            n_sequences, seq_len, d_model
        )

        # 保存原始数据
        data = {
            'l2_scan': {
                'lambdas': [float(x) for x in lambdas],
                'results': l2_scan_data,
                'best_lambda': float(best_l2_lambda),
            },
            'dof_scan': {
                'alphas': [float(x) for x in alphas],
                'results': dof_scan_data,
                'best_alpha': float(best_dof_alpha),
            },
            'comparison': comparison,
            'meta': {
                'n_sequences': n_sequences,
                'seq_len': seq_len,
                'd_model': d_model,
                'seed': seed,
                'cos72': float(COS72),
                'phi': float(PHI),
                'elapsed_seconds': elapsed,
            }
        }
        data_path = os.path.join(OUTPUT_DIR, 'phi_vs_l2_data.json')
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"数据已保存: {data_path}")

        # 生成摘要
        phi5 = comparison['phi5']
        std5 = comparison['std5']
        std5_l2 = comparison['std5_l2']
        std5_dof = comparison['std5_dof_drop']

        l2_jsd_gap = std5_l2['avg_mean_jsd'] - phi5['avg_mean_jsd']
        l2_sgap_gap = phi5['avg_spectral_gap'] - std5_l2['avg_spectral_gap']
        dof_jsd_gap = std5_dof['avg_mean_jsd'] - phi5['avg_mean_jsd']
        dof_sgap_gap = phi5['avg_spectral_gap'] - std5_dof['avg_spectral_gap']

        summary = (
            f"φ-Attention vs L2 正则化实验完成\n\n"
            f"**四组对比 (DOF匹配到≈4.5)：**\n"
            f"| 模型 | JSD | DOF | 谱间隙 |\n"
            f"|------|-----|-----|--------|\n"
            f"| φ5 (C5) | {phi5['avg_mean_jsd']:.4f} | {phi5['avg_effective_dof']:.2f} | {phi5['avg_spectral_gap']:.4f} |\n"
            f"| Std5 | {std5['avg_mean_jsd']:.4f} | {std5['avg_effective_dof']:.2f} | {std5['avg_spectral_gap']:.4f} |\n"
            f"| Std5-L2 | {std5_l2['avg_mean_jsd']:.4f} | {std5_l2['avg_effective_dof']:.2f} | {std5_l2['avg_spectral_gap']:.4f} |\n"
            f"| Std5-DOF-drop | {std5_dof['avg_mean_jsd']:.4f} | {std5_dof['avg_effective_dof']:.2f} | {std5_dof['avg_spectral_gap']:.4f} |\n\n"
            f"**关键发现：**\n"
            f"- L2正则化(λ={best_l2_lambda:.2e})使DOF≈{std5_l2['avg_effective_dof']:.2f}，"
            f"但JSD差φ5 {l2_jsd_gap:+.4f}，谱间隙差{l2_sgap_gap:+.4f}\n"
            f"- Head混合(α={best_dof_alpha:.2f})使DOF≈{std5_dof['avg_effective_dof']:.2f}，"
            f"但JSD差φ5 {dof_jsd_gap:+.4f}，谱间隙差{dof_sgap_gap:+.4f}\n"
        )

        # 判断结论
        jsd_thresh = 0.02
        sgap_thresh = 0.01
        l2_captures = (l2_jsd_gap < jsd_thresh) and (l2_sgap_gap < sgap_thresh)
        dof_captures = (dof_jsd_gap < jsd_thresh) and (dof_sgap_gap < sgap_thresh)

        if l2_captures or dof_captures:
            summary += "→ **结论：φ-Attention优势主要是正则化效应**\n"
        elif (not (l2_jsd_gap < jsd_thresh) or not (l2_sgap_gap < sgap_thresh)) and \
             (not (dof_jsd_gap < jsd_thresh) or not (dof_sgap_gap < sgap_thresh)):
            summary += "→ **结论：φ-Attention优势是C5结构效应**\n"
        else:
            summary += "→ **结论：混合效应（结构与正则化共同作用）**\n"

        actual_mode = result_mode if result_mode != "auto" else "display_only"
        await sdk.submit_result(
            result_mode=actual_mode,
            status="success",
            message=summary,
            data={
                "report_path": report_path,
                "data_path": data_path,
                "best_l2_lambda": float(best_l2_lambda),
                "best_dof_alpha": float(best_dof_alpha),
                "phi5_dof": phi5['avg_effective_dof'],
                "phi5_jsd": phi5['avg_mean_jsd'],
                "phi5_sgap": phi5['avg_spectral_gap'],
                "std5_l2_dof": std5_l2['avg_effective_dof'],
                "std5_l2_jsd": std5_l2['avg_mean_jsd'],
                "std5_l2_sgap": std5_l2['avg_spectral_gap'],
                "conclusion": "structure" if not (l2_captures or dof_captures) else "regularization",
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        await sdk.submit_result(
            result_mode="notify",
            status="error",
            message=f"φ vs L2实验失败: {e}",
            data={"error_type": type(e).__name__},
        )


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
