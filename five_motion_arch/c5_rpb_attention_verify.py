#!/usr/bin/env python3
"""
C5-RPB Attention — 相对位置偏置方案验证
=========================================
核心洞察: 
- v1(v0.1耦合): Q向量混合太弱, 随机投影主导 → C5被淹没
- v4模拟结论: C5-RPB是正确机制
- RPB直接加在attention score上, 不经过线性变换, 结构不会被洗掉

设计:
- 标准Attention: score = Q·K^T / √d
- C5-RPB: score = Q·K^T / √d + B_c5(position, head)
  B_c5的5个head有C5循环相位关系:
  B[h] = A * cos(2πh/5 + φ(position))
  
- Z₂塌缩: φ → -φ (反转相位), 在特定层触发

验证:
1. C5-RPB是否产生比标准Attention更强的C5循环结构
2. C5-RPB + Z₂是否产生显著的注意力模式突变
3. RPB幅度 vs C5结构强度的关系
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2
C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ = [(0,2),(0,3),(1,3),(1,4),(2,4)]
MOTION_LABELS = ["认", "遇", "落", "裂", "余"]

D_MODEL = 500
N_HEADS = 5
D_HEAD = D_MODEL // N_HEADS
N_LAYERS = 28
N_PROMPTS = 20
SEQ_LEN = 8
SEED = 42

def make_c5_rpb(seq_len, n_heads, amplitude=1.0, phi_shift=0.0):
    """生成C5结构的相对位置偏置
    
    B[h, i, j] = A * cos(2πh/5 + φ_shift + π*(i-j)/seq_len)
    
    - h: head index (0-4), 天然有C5相位 2πh/5
    - i-j: 相对位置, 引入位置依赖的相位
    - φ_shift: 全局相位偏移 (Z₂塌缩: φ_shift += π)
    
    C5结构: head之间的2π/5相位差 → 邻接head相似, 远离head不相似
    Z₂否定: φ_shift += π → 反转所有head的相位 → 认↔余互换
    """
    B = np.zeros((n_heads, seq_len, seq_len))
    for h in range(n_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos = (i - j) / seq_len
                phase = 2 * math.pi * h / 5 + phi_shift + math.pi * rel_pos
                B[h, i, j] = amplitude * math.cos(phase)
    return B

def make_z2_rpb(seq_len, n_heads, amplitude=1.0):
    """Z₂否定的RPB: 全局相位+π"""
    return make_c5_rpb(seq_len, n_heads, amplitude, phi_shift=math.pi)

def standard_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    out = np.matmul(attn_weights, V)
    last_token = out[:, :, -1, :]
    result = last_token.reshape(last_token.shape[0], -1)
    return result, attn_weights

def c5_rpb_attention(Q, K, V, rpb_matrix):
    """C5-RPB注意力: 在attention score上加C5结构的偏置"""
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    # 加入C5-RPB偏置
    # rpb_matrix: [n_heads, seq_len, seq_len] → 广播到 [batch, n_heads, seq_len, seq_len]
    scores = scores + rpb_matrix[np.newaxis, :, :, :]
    
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    out = np.matmul(attn_weights, V)
    last_token = out[:, :, -1, :]
    result = last_token.reshape(last_token.shape[0], -1)
    return result, attn_weights

def make_layer_params(rng):
    d = D_MODEL
    dh = D_HEAD
    return {
        'Wq': rng.standard_normal((N_HEADS * dh, d)) / math.sqrt(d),
        'Wk': rng.standard_normal((N_HEADS * dh, d)) / math.sqrt(d),
        'Wv': rng.standard_normal((N_HEADS * dh, d)) / math.sqrt(d),
        'Wo': rng.standard_normal((d, N_HEADS * dh)) / math.sqrt(d),
    }

def forward_layer(h, params, rpb=None):
    """单层前向, rpb=None时用标准attention"""
    batch = h.shape[0]
    Q_all = h @ params['Wq'].T
    K_all = h @ params['Wk'].T
    V_all = h @ params['Wv'].T
    Q = Q_all.reshape(batch, SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    K = K_all.reshape(batch, SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    V = V_all.reshape(batch, SEQ_LEN, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    
    if rpb is not None:
        attn_out, attn_weights = c5_rpb_attention(Q, K, V, rpb)
    else:
        attn_out, attn_weights = standard_attention(Q, K, V)
    
    proj = attn_out @ params['Wo'].T
    h_last = h[:, -1, :]
    out = h_last + proj
    mean = out.mean(axis=-1, keepdims=True)
    std = out.std(axis=-1, keepdims=True) + 1e-5
    out = (out - mean) / std
    return out, attn_weights

def run_model(h0_seq, layer_params_list, rpb_per_layer=None):
    """运行完整模型
    rpb_per_layer: dict {layer_idx: rpb_matrix}, None表示标准attention
    """
    layer_outputs = []
    layer_attn_weights = []
    h = h0_seq.copy()
    for l in range(len(layer_params_list)):
        rpb = rpb_per_layer.get(l) if rpb_per_layer else None
        out, weights = forward_layer(h, layer_params_list[l], rpb=rpb)
        layer_outputs.append(out)
        layer_attn_weights.append(weights)
        h_new = np.zeros_like(h)
        h_new[:, -1, :] = out
        h_new[:, :-1, :] = h[:, :-1, :]
        h = h_new
    return layer_outputs, layer_attn_weights

def extract_phase_from_output(layer_output, motion_labels):
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

def measure_attn_head_c5(attn_weights, motion_labels):
    """从注意力权重提取5动×5head矩阵, 测C5结构"""
    # attn_weights: [batch, n_heads, seq_len, seq_len]
    # 取每个motion在每个head上的激活模式
    head_act = np.zeros((5, N_HEADS))
    counts = np.zeros(5)
    for idx, mi in motion_labels:
        for h in range(N_HEADS):
            head_act[mi, h] += attn_weights[idx, h, -1, :].mean()
        counts[mi] += 1
    for k in range(5):
        if counts[k] > 0:
            head_act[k] /= counts[k]
    return measure_c5_structure(head_act)

def main():
    print("=" * 70)
    print("C5-RPB Attention — 相对位置偏置验证")
    print("=" * 70)
    
    rng = np.random.default_rng(SEED)
    
    # 准备输入
    motion_means = rng.standard_normal((5, D_MODEL)) * 2.0
    inputs = []
    motion_labels = []
    for mi in range(5):
        for _ in range(N_PROMPTS):
            seq = np.zeros((SEQ_LEN, D_MODEL))
            for t in range(SEQ_LEN):
                noise = rng.standard_normal(D_MODEL) * 0.5
                seq[t] = motion_means[mi] + noise
                if t > 0:
                    seq[t] += seq[t-1] * 0.3
            inputs.append(seq)
            motion_labels.append((len(inputs)-1, mi))
    inputs = np.array(inputs)
    
    layer_params = [make_layer_params(rng) for _ in range(N_LAYERS)]
    
    # ===== 实验1: 标准Attention =====
    print("\n[1] 标准Attention baseline...")
    std_outputs, std_attn = run_model(inputs, layer_params)
    
    # ===== 实验2: C5-RPB, 不同幅度扫描 =====
    amplitudes = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("\n[2] C5-RPB幅度扫描...")
    print(f"\n  {'幅度':>5} | {'k1(层0)':>8} {'k1(层14)':>9} {'k1(层27)':>9} | {'adj':>7} {'nonadj':>7} | {'最近邻':>6} | {'head-k1':>8}")
    print(f"  {'-'*80}")
    
    # 标准baseline
    k1_std = []
    for l in range(N_LAYERS):
        pv = extract_phase_from_output(std_outputs[l], motion_labels)
        k1_std.append(measure_c5_structure(pv)['k1_ratio'])
    pv_std_f = extract_phase_from_output(std_outputs[-1], motion_labels)
    c5_std_f = measure_c5_structure(pv_std_f)
    head_c5_std = measure_attn_head_c5(std_attn[-1], motion_labels)
    print(f"  {'标准':>5} | {k1_std[0]:8.4f} {k1_std[14]:9.4f} {k1_std[27]:9.4f} | "
          f"{c5_std_f['adj_sim']:7.4f} {c5_std_f['nonadj_sim']:7.4f} | {c5_std_f['nearest_c5']:6d} | {head_c5_std['k1_ratio']:8.4f}")
    
    results = {}
    
    for amp in amplitudes:
        # 生成C5-RPB (所有层共用同一个)
        rpb = make_c5_rpb(SEQ_LEN, N_HEADS, amplitude=amp)
        rpb_all = {l: rpb for l in range(N_LAYERS)}
        
        c5_outputs, c5_attn = run_model(inputs, layer_params, rpb_per_layer=rpb_all)
        
        # k1曲线
        k1_values = []
        for l in range(N_LAYERS):
            pv = extract_phase_from_output(c5_outputs[l], motion_labels)
            k1_values.append(measure_c5_structure(pv)['k1_ratio'])
        
        # 最终层
        pv_final = extract_phase_from_output(c5_outputs[-1], motion_labels)
        c5_final = measure_c5_structure(pv_final)
        
        # Head维度的C5结构
        head_c5 = measure_attn_head_c5(c5_attn[-1], motion_labels)
        
        results[amp] = {
            'k1_values': k1_values,
            'c5_final': c5_final,
            'head_c5': head_c5,
            'outputs': c5_outputs,
            'attn': c5_attn,
        }
        
        print(f"  {amp:5.1f} | {k1_values[0]:8.4f} {k1_values[14]:9.4f} {k1_values[27]:9.4f} | "
              f"{c5_final['adj_sim']:7.4f} {c5_final['nonadj_sim']:7.4f} | {c5_final['nearest_c5']:6d} | {head_c5['k1_ratio']:8.4f}")
    
    # ===== 实验3: Z₂塌缩 — 在层14切换RPB =====
    print("\n[3] Z₂塌缩 (层14: C5-RPB → Z₂-RPB)...")
    
    amp_best = 2.0  # 中等幅度
    
    rpb_normal = make_c5_rpb(SEQ_LEN, N_HEADS, amplitude=amp_best)
    rpb_z2 = make_z2_rpb(SEQ_LEN, N_HEADS, amplitude=amp_best)
    
    # 正常C5-RPB全部层
    rpb_all_normal = {l: rpb_normal for l in range(N_LAYERS)}
    outputs_normal, attn_normal = run_model(inputs, layer_params, rpb_per_layer=rpb_all_normal)
    
    # 层0-13: C5-RPB, 层14-27: Z₂-RPB
    rpb_collapse = {}
    for l in range(14):
        rpb_collapse[l] = rpb_normal
    for l in range(14, N_LAYERS):
        rpb_collapse[l] = rpb_z2
    outputs_collapse, attn_collapse = run_model(inputs, layer_params, rpb_per_layer=rpb_collapse)
    
    print(f"\n  {'层':>4} | {'正常k1':>8} {'塌缩k1':>8} | {'正常adj':>8} {'塌缩adj':>8} | {'相位偏移':>8}")
    print(f"  {'-'*60}")
    
    for l in [0, 5, 13, 14, 15, 19, 24, 27]:
        pv_n = extract_phase_from_output(outputs_normal[l], motion_labels)
        c5_n = measure_c5_structure(pv_n)
        pv_c = extract_phase_from_output(outputs_collapse[l], motion_labels)
        c5_c = measure_c5_structure(pv_c)
        shift = np.mean(np.abs(c5_n['sim_matrix'] - c5_c['sim_matrix']))
        
        tag = ""
        if l == 14: tag = " ← Z₂翻转"
        elif l > 14: tag = f" ← 翻转后{l-14}层"
        
        print(f"  {l:4d} | {c5_n['k1_ratio']:8.4f} {c5_c['k1_ratio']:8.4f} | "
              f"{c5_n['adj_sim']:8.4f} {c5_c['adj_sim']:8.4f} | {shift:8.4f}{tag}")
    
    # ===== 实验4: 不同幅度下Z₂塌缩偏移 =====
    print("\n[4] Z₂塌缩偏移 vs RPB幅度...")
    
    z2_results = {}
    for amp in [0.5, 1.0, 2.0, 5.0, 10.0]:
        rpb_n = make_c5_rpb(SEQ_LEN, N_HEADS, amplitude=amp)
        rpb_z = make_z2_rpb(SEQ_LEN, N_HEADS, amplitude=amp)
        
        # 正常
        rpb_all_n = {l: rpb_n for l in range(N_LAYERS)}
        out_n, _ = run_model(inputs, layer_params, rpb_per_layer=rpb_all_n)
        
        # 塌缩
        rpb_c = {}
        for l in range(14):
            rpb_c[l] = rpb_n
        for l in range(14, N_LAYERS):
            rpb_c[l] = rpb_z
        out_c, _ = run_model(inputs, layer_params, rpb_per_layer=rpb_c)
        
        # 层15的偏移
        pv_n = extract_phase_from_output(out_n[15], motion_labels)
        c5_n = measure_c5_structure(pv_n)
        pv_c = extract_phase_from_output(out_c[15], motion_labels)
        c5_c = measure_c5_structure(pv_c)
        shift_15 = np.mean(np.abs(c5_n['sim_matrix'] - c5_c['sim_matrix']))
        
        # 层27的偏移
        pv_n27 = extract_phase_from_output(out_n[27], motion_labels)
        c5_n27 = measure_c5_structure(pv_n27)
        pv_c27 = extract_phase_from_output(out_c[27], motion_labels)
        c5_c27 = measure_c5_structure(pv_c27)
        shift_27 = np.mean(np.abs(c5_n27['sim_matrix'] - c5_c27['sim_matrix']))
        
        z2_results[amp] = {'shift_15': shift_15, 'shift_27': shift_27}
        print(f"  幅度={amp:5.1f}: Z₂偏移(层15)={shift_15:.4f}, Z₂偏移(层27)={shift_27:.4f}")
    
    # ===== 实验5: Head维度的C5结构 =====
    print("\n[5] Head维度C5结构 (最终层)...")
    print(f"\n  {'方法':>10} | {'head-k1':>8} {'head-adj':>8} {'head-nn':>8}")
    print(f"  {'-'*45}")
    
    head_c5_std = measure_attn_head_c5(std_attn[-1], motion_labels)
    print(f"  {'标准':>10} | {head_c5_std['k1_ratio']:8.4f} {head_c5_std['adj_sim']:8.4f} {head_c5_std['nearest_c5']:8d}")
    
    for amp in [1.0, 5.0, 10.0]:
        hc = results[amp]['head_c5']
        print(f"  {'RPB='+str(amp):>10} | {hc['k1_ratio']:8.4f} {hc['adj_sim']:8.4f} {hc['nearest_c5']:8d}")
    
    # ===== 相似度矩阵对比 =====
    print("\n[6] 相似度矩阵对比 (RPB=5.0 vs 标准)...")
    
    sim_std = c5_std_f['sim_matrix']
    sim_rpb = results[5.0]['c5_final']['sim_matrix']
    
    print("\n  标准Attention:")
    for i in range(5):
        row = ' '.join(f'{sim_std[i,j]:7.3f}' for j in range(5))
        print(f"    {MOTION_LABELS[i]}: {row}")
    
    print("\n  C5-RPB (amp=5.0):")
    for i in range(5):
        row = ' '.join(f'{sim_rpb[i,j]:7.3f}' for j in range(5))
        print(f"    {MOTION_LABELS[i]}: {row}")
    
    # ===== RPB本身的结构可视化 =====
    print("\n[7] C5-RPB结构 (amp=2.0, head 0-4, position [last, :])...")
    rpb_vis = make_c5_rpb(SEQ_LEN, N_HEADS, amplitude=2.0)
    for h in range(N_HEADS):
        vals = ' '.join(f'{rpb_vis[h, -1, j]:6.3f}' for j in range(SEQ_LEN))
        print(f"    head{h}({MOTION_LABELS[h]}): {vals}")
    
    # ===== 核心结论 =====
    print("\n" + "=" * 70)
    print("核心结论")
    print("=" * 70)
    
    # 1. RPB效果
    best_amp = max(results.keys(), key=lambda a: results[a]['c5_final']['k1_ratio'])
    best_k1 = results[best_amp]['c5_final']['k1_ratio']
    print(f"\n  1. C5-RPB效果:")
    print(f"     标准k1(终层): {k1_std[-1]:.4f}")
    print(f"     最佳RPB k1(终层): {best_k1:.4f} (amp={best_amp})")
    print(f"     Δk1: {best_k1 - k1_std[-1]:.4f}")
    if best_k1 > k1_std[-1] + 0.05:
        print(f"     ✅✅ C5-RPB显著增强C5循环结构!")
    elif best_k1 > k1_std[-1] + 0.01:
        print(f"     ⚠️ C5-RPB有微弱增强")
    else:
        print(f"     ❌ C5-RPB对输出向量C5结构无显著效果")
    
    # 2. Head维度
    best_head_amp = max(results.keys(), key=lambda a: results[a]['head_c5']['k1_ratio'])
    best_head_k1 = results[best_head_amp]['head_c5']['k1_ratio']
    print(f"\n  2. Head维度C5结构:")
    print(f"     标准head-k1: {head_c5_std['k1_ratio']:.4f}")
    print(f"     最佳RPB head-k1: {best_head_k1:.4f} (amp={best_head_amp})")
    if best_head_k1 > head_c5_std['k1_ratio'] + 0.05:
        print(f"     ✅✅ C5-RPB在注意力head维度产生显著C5结构!")
    elif best_head_k1 > head_c5_std['k1_ratio'] + 0.01:
        print(f"     ⚠️ C5-RPB在head维度有微弱C5结构")
    else:
        print(f"     ❌ Head维度也没有显著C5结构")
    
    # 3. Z₂塌缩
    max_z2_shift = max(z2_results.values(), key=lambda x: x['shift_15'])['shift_15']
    max_z2_amp = max(z2_results.keys(), key=lambda a: z2_results[a]['shift_15'])
    print(f"\n  3. Z₂塌缩 (RPB版):")
    print(f"     最大偏移: {max_z2_shift:.4f} (amp={max_z2_amp})")
    print(f"     对比Q耦合版: 最大偏移0.017")
    if max_z2_shift > 0.1:
        print(f"     ✅✅ RPB版Z₂塌缩显著强于Q耦合版!")
    elif max_z2_shift > 0.03:
        print(f"     ⚠️ RPB版Z₂塌缩略强于Q耦合版")
    else:
        print(f"     ❌ Z₂塌缩仍然太弱")
    
    # 4. 总结
    print(f"\n  4. 载体对比总结:")
    print(f"     残差连接: k1衰减到0, Z₂偏移=0.000 → ❌ 死载体")
    print(f"     Q耦合(0.1): k1不衰减但Δk1≈0, Z₂偏移=0.017 → ❌ 耦合太弱")
    print(f"     C5-RPB:     k1不衰减Δk1={best_k1-k1_std[-1]:.4f}, Z₂偏移={max_z2_shift:.4f} → 待判定")
    print(f"\n     关键问题: numpy随机权重+无训练 → 5个head不特异化")
    print(f"     真实模型中: 训练会让head学到不同pattern → C5-RPB效果会放大")

if __name__ == "__main__":
    main()
