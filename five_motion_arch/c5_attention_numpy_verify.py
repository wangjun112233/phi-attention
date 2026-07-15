#!/usr/bin/env python3
"""
C5-Coupled Attention vs 标准Attention — numpy验证
===================================================
核心洞察：C5结构放在残差连接上会被洗掉，放在注意力机制上每层重新涌现

为什么注意力是正确载体：
- 残差连接：h_{l+1} = h_l + Δ, Δ被后续层洗掉
- 注意力：Q·K^T 每层重新计算，结构不会累积衰减
- C5耦合在Q的head维度上 → 5个head之间有φ旋转关系 → C5相位每层重新生成

设计：
1. 标准Attention: 5个head独立计算
2. C5-Attention: 5个head的Q之间用C5循环矩阵耦合
3. Z₂塌缩: 在C5-Attention中，特定层对Q向量施加Z₂否定 → 注意力模式突变

验证：
- C5相位结构是否每层保持(不被洗掉)
- Z₂否定是否在C5-Attention中产生可观测的注意力模式变化
- 与残差连接版本对比
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2

C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ = [(0,2),(0,3),(1,3),(1,4),(2,4)]
MOTION_LABELS = ["认", "遇", "落", "裂", "余"]

# ============================================================================
# C5旋转矩阵 (5×5, 作用于head维度)
# ============================================================================

def make_c5_rotation(theta=2*math.pi/5):
    """C5旋转: 每个head旋转θ
    本征值: e^{2πik/5}, k=0,1,2,3,4
    """
    R = np.zeros((5, 5))
    for i in range(5):
        R[i, (i+1)%5] = math.cos(theta)
        R[i, (i-1)%5] = math.sin(theta)  # 不对, 用标准旋转
    # 标准C5循环排列+相位
    R = np.zeros((5, 5))
    for i in range(5):
        R[i, (i+1)%5] = 1.0
    # 这只是排列矩阵, 加上相位耦合
    # 用circulant形式: R_{ij} = cos((i-j)*θ) + sin((i-j)*θ) 不对
    # 最简单: R = 排列矩阵 * 相位对角矩阵
    perm = np.zeros((5, 5))
    for i in range(5):
        perm[i, (i+1)%5] = 1.0
    phase = np.diag([1.0, math.cos(theta), math.cos(2*theta), math.cos(2*theta), math.cos(theta)])
    R = perm @ phase
    return R

def make_c5_circulant():
    """C5循环耦合矩阵 (邻接/2)"""
    C = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=np.float64) / 2.0
    return C

def make_z2_negation():
    """Z₂ reflection: 反转C5方向"""
    return np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.float64)

# ============================================================================
# 模拟参数
# ============================================================================

D_MODEL = 500
N_HEADS = 5
D_HEAD = D_MODEL // N_HEADS  # 100
N_LAYERS = 28
N_PROMPTS = 20
SEQ_LEN = 8  # 短序列, 够用
C5_COUPLING = 0.1  # C5耦合强度
SEED = 42

# ============================================================================
# Attention模拟
# ============================================================================

def standard_attention(Q, K, V):
    """标准多头注意力: 5个head独立
    Q, K, V: [batch, n_heads, seq_len, d_head]
    返回: [batch, d_model] (取最后一个token)
    """
    # Scaled dot-product attention
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)  # [b, h, s, s]
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Apply to V
    out = np.matmul(attn_weights, V)  # [b, h, s, d_head]
    
    # 取最后一个token, 展平heads
    last_token = out[:, :, -1, :]  # [b, h, d_head]
    result = last_token.reshape(last_token.shape[0], -1)  # [b, d_model]
    return result, attn_weights

def c5_attention(Q, K, V, c5_coupling_matrix, coupling_strength=0.1):
    """C5耦合注意力: 5个head的Q之间有C5循环耦合
    
    核心改动: Q' = Q + coupling_strength * C5_mix(Q_across_heads)
    
    C5_mix在head维度上混合Q向量, 使得5个head不再是独立的,
    而是有C5循环相位关系
    """
    # Q: [batch, n_heads, seq_len, d_head]
    # 在head维度上混合Q
    # Q: [b, h, s, d] → 对h维度用C5混合
    Q_mixed = Q.copy()
    
    # C5耦合: Q'[h] = Q[h] + s * sum_j C5[h,j] * Q[j]
    for h in range(N_HEADS):
        coupling = np.zeros_like(Q[:, h, :, :])
        for j in range(N_HEADS):
            coupling += c5_coupling_matrix[h, j] * Q[:, j, :, :]
        Q_mixed[:, h, :, :] = Q[:, h, :, :] + coupling_strength * coupling
    
    # 然后正常算attention
    return standard_attention(Q_mixed, K, V)

def c5_attention_with_collapse(Q, K, V, c5_coupling_matrix, z2_neg, coupling_strength=0.1):
    """C5注意力 + Z₂否定(塌缩): 先耦合, 再塌缩
    
    塌缩 = 对Q施加Z₂否定 → 反转C5循环方向
    效果: 认↔余互换, 遇↔裂互换 → 注意力模式突变
    """
    # 先C5耦合
    Q_mixed = Q.copy()
    for h in range(N_HEADS):
        coupling = np.zeros_like(Q[:, h, :, :])
        for j in range(N_HEADS):
            coupling += c5_coupling_matrix[h, j] * Q[:, j, :, :]
        Q_mixed[:, h, :, :] = Q[:, h, :, :] + coupling_strength * coupling
    
    # Z₂否定: 在head维度上反转
    Q_collapsed = np.zeros_like(Q_mixed)
    for h in range(N_HEADS):
        for j in range(N_HEADS):
            Q_collapsed[:, h, :, :] += z2_neg[h, j] * Q_mixed[:, j, :, :]
    
    return standard_attention(Q_collapsed, K, V)

# ============================================================================
# 完整模型模拟
# ============================================================================

def make_layer_params(rng):
    """生成一层的参数: Wq, Wk, Wv, Wo, Wff1, Wff2"""
    d = D_MODEL
    dh = D_HEAD
    params = {
        'Wq': rng.standard_normal((N_HEADS * dh, d)) / math.sqrt(d),
        'Wk': rng.standard_normal((N_HEADS * dh, d)) / math.sqrt(d),
        'Wv': rng.standard_normal((N_HEADS * dh, d)) / math.sqrt(d),
        'Wo': rng.standard_normal((d, N_HEADS * dh)) / math.sqrt(d),
        'Wff1': rng.standard_normal((d * 4, d)) / math.sqrt(d),
        'Wff2': rng.standard_normal((d, d * 4)) / math.sqrt(d),
    }
    return params

def forward_layer(h, params, mode='standard', c5_matrix=None, z2=None,
                  coupling_strength=0.1, collapse=False):
    """单层前向传播
    h: [batch, seq_len, d_model]
    """
    batch = h.shape[0]
    
    # Self-attention
    Q_all = h @ params['Wq'].T  # [b, s, n_heads*d_head]
    K_all = h @ params['Wk'].T
    V_all = h @ params['Wv'].T
    
    # Reshape to multi-head
    Q = Q_all.reshape(batch, SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)  # [b, h, s, d]
    K = K_all.reshape(batch, SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    V = V_all.reshape(batch, SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    
    # Attention
    if mode == 'standard':
        attn_out, attn_weights = standard_attention(Q, K, V)
    elif mode == 'c5' and not collapse:
        attn_out, attn_weights = c5_attention(Q, K, V, c5_matrix, coupling_strength)
    elif mode == 'c5' and collapse:
        attn_out, attn_weights = c5_attention_with_collapse(Q, K, V, c5_matrix, z2, coupling_strength)
    else:
        attn_out, attn_weights = standard_attention(Q, K, V)
    
    # Output projection
    proj = attn_out @ params['Wo'].T  # [b, d]
    
    # FFN (简化: 只对最后一个token)
    # 这里简化处理: 用残差连接
    h_last = h[:, -1, :]  # [b, d]
    out = h_last + proj  # 残差
    
    # LayerNorm (简化)
    mean = out.mean(axis=-1, keepdims=True)
    std = out.std(axis=-1, keepdims=True) + 1e-5
    out = (out - mean) / std
    
    return out, attn_weights

def run_model(h0_seq, layer_params_list, mode='standard', c5_matrix=None, z2=None,
              coupling_strength=0.1, collapse_layers=None):
    """运行完整模型
    h0_seq: [batch, seq_len, d_model]
    返回: 每层的输出 + 每层的注意力权重
    """
    layer_outputs = []
    layer_attn_weights = []
    
    h = h0_seq.copy()
    
    for l in range(len(layer_params_list)):
        collapse = (collapse_layers is not None and l in collapse_layers)
        out, weights = forward_layer(h, layer_params_list[l], 
                                      mode=mode, c5_matrix=c5_matrix, z2=z2,
                                      coupling_strength=coupling_strength,
                                      collapse=collapse)
        layer_outputs.append(out)
        layer_attn_weights.append(weights)
        
        # 重建序列表示 (用最后一个token的输出扩展回序列)
        h_new = np.zeros_like(h)
        h_new[:, -1, :] = out
        h_new[:, :-1, :] = h[:, :-1, :]  # 简化: 前面token不变
        h = h_new
    
    return layer_outputs, layer_attn_weights

# ============================================================================
# C5结构测量
# ============================================================================

def extract_phase_from_attn(attn_weights, motion_labels):
    """从注意力权重提取5动相位向量
    
    attn_weights: [batch, n_heads, seq_len, seq_len]
    取每个motion对应prompt的平均注意力模式
    """
    # 取每个head对最后一个token的注意力分布
    # attn_weights[:, :, -1, :] → [batch, n_heads, seq_len]
    last_attn = attn_weights[:, :, -1, :]  # [b, h, s]
    
    phase_vecs = np.zeros((5, N_HEADS * SEQ_LEN))
    counts = np.zeros(5)
    
    for idx, mi in motion_labels:
        flat = last_attn[idx].flatten()  # [h*s]
        phase_vecs[mi] += flat
        counts[mi] += 1
    
    for k in range(5):
        if counts[k] > 0:
            phase_vecs[k] /= counts[k]
    
    return phase_vecs

def extract_phase_from_output(layer_output, motion_labels):
    """从层输出(最后一个token)提取5动相位向量"""
    phase_vecs = np.zeros((5, D_MODEL))
    counts = np.zeros(5)
    
    for idx, mi in motion_labels:
        phase_vecs[mi] += layer_output[idx]
        counts[mi] += 1
    
    for k in range(5):
        if counts[k] > 0:
            phase_vecs[k] /= counts[k]
    
    return phase_vecs

def measure_c5_structure(phase_vecs):
    """测量C5循环结构"""
    norms = np.linalg.norm(phase_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = phase_vecs / norms
    
    sim = normalized @ normalized.T
    
    adj_sim = np.mean([sim[i,j] for i,j in C5_ADJACENT])
    nonadj_sim = np.mean([sim[i,j] for i,j in C5_NONADJ])
    circular_ratio = adj_sim / max(nonadj_sim, 1e-10)
    
    nearest_c5 = 0
    for i in range(5):
        sims = sim[i].copy()
        sims[i] = -999
        nearest = np.argmax(sims)
        if nearest in [(i+1)%5, (i-1)%5]:
            nearest_c5 += 1
    
    # DFT k=1
    n = 5
    W_dft = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
    dft = W_dft @ phase_vecs
    freq_energy = np.array([np.mean(np.abs(dft[k])**2) for k in range(n)])
    total = freq_energy.sum()
    k1_ratio = (freq_energy[1] + freq_energy[4]) / max(total, 1e-10)
    
    return {
        'circular_ratio': float(circular_ratio),
        'adj_sim': float(adj_sim),
        'nonadj_sim': float(nonadj_sim),
        'nearest_c5': nearest_c5,
        'k1_ratio': float(k1_ratio),
        'sim_matrix': sim,
    }

def measure_head_c5_structure(attn_weights, motion_labels):
    """测量head之间的C5结构: 不同motion在不同head上的激活差异"""
    # attn_weights: [batch, n_heads, seq_len, seq_len]
    # 对每个motion, 取5个head的激活均值
    head_activations = np.zeros((5, N_HEADS))
    counts = np.zeros(5)
    
    for idx, mi in motion_labels:
        # 每个head的平均注意力权重
        for h in range(N_HEADS):
            head_activations[mi, h] += attn_weights[idx, h, -1, :].mean()
        counts[mi] += 1
    
    for k in range(5):
        if counts[k] > 0:
            head_activations[k] /= counts[k]
    
    return measure_c5_structure(head_activations)

# ============================================================================
# 主实验
# ============================================================================

def main():
    print("=" * 70)
    print("C5-Coupled Attention vs 标准Attention — numpy验证")
    print("=" * 70)
    
    rng = np.random.default_rng(SEED)
    c5_matrix = make_c5_circulant()
    z2_neg = make_z2_negation()
    
    print(f"\n参数: d_model={D_MODEL}, n_heads={N_HEADS}, d_head={D_HEAD}, layers={N_LAYERS}")
    print(f"C5耦合强度: {C5_COUPLING}")
    print(f"C5循环矩阵本征值: {np.linalg.eigvalsh(c5_matrix).round(4)}")
    
    # ===== 准备5种motion的输入序列 =====
    print("\n[1] 生成5种motion的输入序列...")
    
    motion_means = rng.standard_normal((5, D_MODEL)) * 2.0
    
    inputs = []
    motion_labels = []
    for mi in range(5):
        for _ in range(N_PROMPTS):
            # 生成序列: 每个token是motion_mean + noise
            seq = np.zeros((SEQ_LEN, D_MODEL))
            for t in range(SEQ_LEN):
                noise = rng.standard_normal(D_MODEL) * 0.5
                seq[t] = motion_means[mi] + noise
                # 加入一些上下文变化
                if t > 0:
                    seq[t] += seq[t-1] * 0.3
            inputs.append(seq)
            motion_labels.append((len(inputs)-1, mi))
    
    inputs = np.array(inputs)  # [100, seq_len, d_model]
    print(f"  总输入: {inputs.shape}")
    
    # ===== 生成层参数 =====
    print("\n[2] 生成层参数...")
    layer_params = [make_layer_params(rng) for _ in range(N_LAYERS)]
    
    # ===== 实验1: 标准Attention =====
    print("\n[3] 标准Attention模型...")
    std_outputs, std_attn = run_model(inputs, layer_params, mode='standard')
    
    # ===== 实验2: C5-Coupled Attention =====
    print("\n[4] C5-Coupled Attention模型...")
    c5_outputs, c5_attn = run_model(inputs, layer_params, mode='c5',
                                      c5_matrix=c5_matrix, coupling_strength=C5_COUPLING)
    
    # ===== 实验3: C5-Attention + Z₂塌缩 =====
    print("\n[5] C5-Attention + Z₂塌缩 (层14)...")
    collapse_layers = {14}
    c5c_outputs, c5c_attn = run_model(inputs, layer_params, mode='c5',
                                        c5_matrix=c5_matrix, z2=z2_neg,
                                        coupling_strength=C5_COUPLING,
                                        collapse_layers=collapse_layers)
    
    # ===== 测量C5结构 =====
    print("\n[6] C5结构测量...")
    
    # 用输出向量测
    print("\n  === 输出向量C5结构 ===")
    print(f"  {'层':>4} | {'标准循环比':>10} {'标准k1':>8} {'标准最近邻':>8} | {'C5循环比':>10} {'C5k1':>8} {'C5最近邻':>8}")
    print(f"  {'-'*70}")
    
    for l in [0, 1, 5, 10, 14, 20, 27]:
        if l >= len(std_outputs):
            continue
        
        pv_std = extract_phase_from_output(std_outputs[l], motion_labels)
        c5_std = measure_c5_structure(pv_std)
        
        pv_c5 = extract_phase_from_output(c5_outputs[l], motion_labels)
        c5_c5 = measure_c5_structure(pv_c5)
        
        marker = ""
        if l == 14: marker = " ← 塌缩层"
        
        print(f"  {l:4d} | {c5_std['circular_ratio']:10.4f} {c5_std['k1_ratio']:8.4f} {c5_std['nearest_c5']:8d} | "
              f"{c5_c5['circular_ratio']:10.4f} {c5_c5['k1_ratio']:8.4f} {c5_c5['nearest_c5']:8d}{marker}")
    
    # 用注意力权重测
    print("\n  === 注意力权重C5结构 ===")
    print(f"  {'层':>4} | {'标准循环比':>10} {'标准k1':>8} {'标准最近邻':>8} | {'C5循环比':>10} {'C5k1':>8} {'C5最近邻':>8}")
    print(f"  {'-'*70}")
    
    for l in [0, 1, 5, 10, 14, 20, 27]:
        if l >= len(std_attn):
            continue
        
        c5_std_a = measure_head_c5_structure(std_attn[l], motion_labels)
        c5_c5_a = measure_head_c5_structure(c5_attn[l], motion_labels)
        
        marker = ""
        if l == 14: marker = " ← 塌缩层"
        
        print(f"  {l:4d} | {c5_std_a['circular_ratio']:10.4f} {c5_std_a['k1_ratio']:8.4f} {c5_std_a['nearest_c5']:8d} | "
              f"{c5_c5_a['circular_ratio']:10.4f} {c5_c5_a['k1_ratio']:8.4f} {c5_c5_a['nearest_c5']:8d}{marker}")
    
    # ===== Z₂塌缩效果 =====
    print("\n[7] Z₂否定(塌缩)效果对比...")
    print(f"  塌缩层: 14")
    
    for l in [14, 15, 19, 24]:
        if l >= len(c5_outputs):
            continue
        
        pv_no = extract_phase_from_output(c5_outputs[l], motion_labels)
        c5_no = measure_c5_structure(pv_no)
        pv_yes = extract_phase_from_output(c5c_outputs[l], motion_labels)
        c5_yes = measure_c5_structure(pv_yes)
        
        shift = np.mean(np.abs(c5_no['sim_matrix'] - c5_yes['sim_matrix']))
        
        tag = " (塌缩层)" if l == 14 else f" (塌缩后{l-14}层)"
        print(f"\n  层{l}{tag}:")
        print(f"    无塌缩: 循环比={c5_no['circular_ratio']:.4f}, k1={c5_no['k1_ratio']:.4f}, 最近邻={c5_no['nearest_c5']}/5")
        print(f"    有塌缩: 循环比={c5_yes['circular_ratio']:.4f}, k1={c5_yes['k1_ratio']:.4f}, 最近邻={c5_yes['nearest_c5']}/5")
        print(f"    相位偏移: {shift:.4f}")
    
    # ===== 最终层相似度矩阵 =====
    print("\n[8] 最终层(层27)相似度矩阵...")
    
    pv_std_final = extract_phase_from_output(std_outputs[-1], motion_labels)
    c5_std_final = measure_c5_structure(pv_std_final)
    
    pv_c5_final = extract_phase_from_output(c5_outputs[-1], motion_labels)
    c5_c5_final = measure_c5_structure(pv_c5_final)
    
    print("\n  标准Attention:")
    sim = c5_std_final['sim_matrix']
    for i in range(5):
        row = ' '.join(f'{sim[i,j]:.3f}' for j in range(5))
        print(f"    {MOTION_LABELS[i]}: {row}")
    
    print("\n  C5-Attention:")
    sim = c5_c5_final['sim_matrix']
    for i in range(5):
        row = ' '.join(f'{sim[i,j]:.3f}' for j in range(5))
        print(f"    {MOTION_LABELS[i]}: {row}")
    
    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    # 对比最终层
    print(f"\n  最终层对比:")
    print(f"  {'指标':<15} {'标准Attn':>10} {'C5-Attn':>10} {'判定':>10}")
    print(f"  {'-'*50}")
    
    cr_s = c5_std_final['circular_ratio']
    cr_c = c5_c5_final['circular_ratio']
    k1_s = c5_std_final['k1_ratio']
    k1_c = c5_c5_final['k1_ratio']
    nn_s = c5_std_final['nearest_c5']
    nn_c = c5_c5_final['nearest_c5']
    
    def judge(val_c, val_s, threshold=0.01):
        if val_c > val_s + threshold: return '✅'
        elif val_c > val_s: return '⚠️'
        else: return '❌'
    
    print(f"  {'循环比':<15} {cr_s:10.4f} {cr_c:10.4f} {judge(cr_c, cr_s):>10}")
    print(f"  {'DFT k=1':<15} {k1_s:10.4f} {k1_c:10.4f} {judge(k1_c, k1_s):>10}")
    print(f"  {'最近邻C5':<15} {nn_s:10d} {nn_c:10d} {'✅' if nn_c > nn_s else '❌':>10}")
    
    # C5保持性: 浅层vs深层的变化幅度
    print(f"\n  C5结构保持性 (浅层→深层):")
    for tag, outputs in [("标准", std_outputs), ("C5", c5_outputs)]:
        k1_values = []
        for l in range(min(28, len(outputs))):
            pv = extract_phase_from_output(outputs[l], motion_labels)
            c5_m = measure_c5_structure(pv)
            k1_values.append(c5_m['k1_ratio'])
        
        k1_early = np.mean(k1_values[:5])
        k1_late = np.mean(k1_values[-5:])
        decay = k1_early - k1_late
        print(f"    {tag}: k1早期={k1_early:.4f}, k1晚期={k1_late:.4f}, 衰减={decay:.4f}")
    
    # Z₂塌缩判定
    collapse_shifts = []
    for l in [15, 19, 24]:
        if l < len(c5_outputs):
            pv_no = extract_phase_from_output(c5_outputs[l], motion_labels)
            c5_no = measure_c5_structure(pv_no)
            pv_yes = extract_phase_from_output(c5c_outputs[l], motion_labels)
            c5_yes = measure_c5_structure(pv_yes)
            collapse_shifts.append(np.mean(np.abs(c5_no['sim_matrix'] - c5_yes['sim_matrix'])))
    
    max_shift = max(collapse_shifts) if collapse_shifts else 0
    
    print(f"\n  Z₂塌缩判定:")
    if max_shift > 0.1:
        print(f"  ✅✅ Z₂否定产生显著相位偏移: {max_shift:.4f} — 注意力叠加态可塌缩!")
    elif max_shift > 0.02:
        print(f"  ⚠️ Z₂否定产生可观测偏移: {max_shift:.4f}")
    else:
        print(f"  ❌ Z₂否定偏移: {max_shift:.4f}")
    
    # 核心结论
    print(f"\n  核心结论:")
    if cr_c > cr_s + 0.02 or k1_c > k1_s + 0.02:
        print("  ✅ C5-Attention比标准Attention产生更强的C5相位结构")
        print("  → 注意力是C5结构的正确载体, 残差连接不是")
    else:
        print("  ❌ C5-Attention也没产生显著C5结构")
        print("  → 需要更强的耦合机制或不同维度的设计")
    
    if max_shift > 0.02:
        print("  ✅ Z₂否定在C5-Attention中有可观测效果")
        print("  → 叠加态+塌缩机制可行")
    
    print(f"\n  与残差连接版本对比:")
    print(f"  残差连接: k1在第5层就衰减到0.07, 10层后归零")
    attn_k1s = []
    for l in range(min(28, len(c5_outputs))):
        pv = extract_phase_from_output(c5_outputs[l], motion_labels)
        c5_m = measure_c5_structure(pv)
        attn_k1s.append(c5_m['k1_ratio'])
    print(f"  C5-Attention: k1在第5层={attn_k1s[4]:.4f}, 第10层={attn_k1s[9]:.4f}, 第27层={attn_k1s[27]:.4f}")
    
    if attn_k1s[9] > 0.05:
        print("  ✅ 注意力保持C5结构的能力显著优于残差连接")
    else:
        print("  ❌ 注意力版本的C5结构也在衰减")

if __name__ == "__main__":
    main()
