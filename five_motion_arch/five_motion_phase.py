#!/usr/bin/env python3
"""
Five-Motion Phase Structure Experiment
=======================================
验证: 同一组层内, 五动是否形成C5旋转相位结构

核心区别:
- 旧实验(错误): 5动→5个空间parcel (并列分类)
- 新实验(正确): 5动→同一空间内5个旋转相位 (循环结构)

方法:
1. 5类motion prompt × 5个 = 25次forward
2. 抓每层hidden state (5×hidden矩阵)
3. 每层做:
   - PCA: 5点在PC1-PC2平面是否成五边形
   - 相邻vs非相邻余弦相似度
   - DFT: 5-cycle频率是否主导
"""

import sys
import os
import time
import json
import math
import numpy as np
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# ============================================================================
# 配置
# ============================================================================

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B"
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

PHI = (1 + math.sqrt(5)) / 2  # 黄金比例

MOTION_NAMES = ["认(Recognize)", "遇(Encounter)", "落(Settle)", "裂(Split)", "余(Residue)"]
MOTION_KEYS = ["ren", "yu", "luo", "lie", "yuu"]
MOTION_LABELS = ["认", "遇", "落", "裂", "余"]

# C5循环邻接: 认→遇→落→裂→余→认
C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]  # 相邻
C5_NONADJ = [(0,2),(0,3),(1,3),(1,4),(2,4)]     # 非相邻(隔1)
C5_OPPOSITE = [(0,2),(0,3)]                       # 对面(隔1和2)

# 每个motion: 多个prompt (取最后一个token的hidden state后平均)
PROMPTS = {
    "ren": [
        "The pattern in this sequence is clearly recognizable as",
        "Looking at the data, I can identify that the structure is",
        "The key feature that distinguishes this object is its",
        "Classification: this specimen belongs to the category of",
        "The fingerprint of this chemical reaction is uniquely",
    ],
    "yu": [
        "When I encountered this problem, the first connection I made was",
        "The unexpected link between these two phenomena is surprisingly",
        "This reminds me of something I saw before, specifically",
        "By associating these two ideas, we discover that they",
        "The surprising intersection of these concepts creates a",
    ],
    "luo": [
        "After long deliberation, the committee finally settled on the",
        "The solution converges to a stable point at which",
        "When the dust settles, what remains is the fundamental",
        "The system reaches equilibrium when all forces are",
        "After all the noise fades, the core truth emerges as",
    ],
    "lie": [
        "The crucial difference between these two approaches is that",
        "What separates successful strategies from failures is the",
        "The boundary that divides these two categories is clearly",
        "By negating the assumption, we discover that the opposite",
        "The contradiction in this argument reveals a hidden",
    ],
    "yuu": [
        "Even years later, the lasting impact of that event still",
        "The residual effect of the treatment persists as a subtle",
        "What remains after removing all the noise is the underlying",
        "The echo of that decision still influences the current",
        "Despite the passage of time, the underlying pattern persists",
    ],
}

# ============================================================================
# Hook工具
# ============================================================================

def compute_layer_alphas(nl, strength):
    """与v18一致的α计算"""
    alphas = []
    for i in range(nl):
        phase = i % 5
        if phase == 0:
            a = 1.0 - strength  # 认: 收敛
        elif phase == 1:
            a = 1.0 + strength * PHI  # 遇: 放大(φ加权)
        elif phase == 2:
            a = 1.0 - strength * 0.5  # 落: 轻收敛
        elif phase == 3:
            a = 1.0 + strength  # 裂: 放大
        else:
            a = 1.0 + strength * PHI**(-3)  # 余: 微残留
        alphas.append(a)
    return alphas

def layer_motion_assignment(nl):
    """层→动的映射"""
    return [i % 5 for i in range(nl)]

def make_attn_scale(alpha):
    def hook(m, inp, out):
        if isinstance(out, tuple):
            scaled = alpha * out[0]
            return (scaled,) + out[1:]
        return alpha * out
    return hook

def make_mlp_scale(alpha):
    def hook(m, inp, out):
        return alpha * out
    return hook

def apply_fixed_hooks(model, alphas):
    hooks = []
    for i, layer in enumerate(model.model.layers):
        a = alphas[i]
        hooks.append(layer.self_attn.register_forward_hook(make_attn_scale(a)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_scale(a)))
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

def make_layer_capture_hook(captures, layer_idx):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captures[layer_idx] = hs.detach().cpu().float()
    return hook

def apply_capture_hooks(model, captures):
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(
            make_layer_capture_hook(captures, i)
        ))
    return hooks

# ============================================================================
# 核心分析
# ============================================================================

def pca_analysis(vectors_5xh):
    """5个motion的hidden state做PCA, 返回PC1-PC2坐标和五边形得分

    五边形得分 = 5点在PC1-PC2平面上到正五边形的最佳拟合误差
    """
    # vectors_5xh: numpy array [5, hidden_size]
    mean = vectors_5xh.mean(axis=0, keepdims=True)
    centered = vectors_5xh - mean

    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords_2d = U[:, :2] * S[:2]  # [5, 2]

    # 五边形得分: 到正五边形的最小旋转拟合误差
    # 标准五边形顶点
    angles = np.array([2 * np.pi * k / 5 for k in range(5)])

    # 归一化坐标
    center = coords_2d.mean(axis=0)
    centered_2d = coords_2d - center
    max_r = np.max(np.linalg.norm(centered_2d, axis=1))
    if max_r < 1e-10:
        return coords_2d, 0.0, S[:5]
    normalized = centered_2d / max_r

    # 尝试所有5种排列×旋转, 找最小误差
    from itertools import permutations
    best_error = float('inf')

    # 只尝试循环排列(5种)而非全排列(120种), C5对称性
    for shift in range(5):
        perm = [(i + shift) % 5 for i in range(5)]
        # 对每个旋转角度找最佳缩放
        for theta_0 in np.linspace(0, 2*np.pi, 36, endpoint=False):
            target = np.array([[np.cos(2*np.pi*k/5 + theta_0),
                                np.sin(2*np.pi*k/5 + theta_0)]
                               for k in perm])
            error = np.mean((normalized - target)**2)
            if error < best_error:
                best_error = error

    pentagon_score = 1.0 / (1.0 + best_error * 100)  # 0-1, 1=完美五边形

    return coords_2d, pentagon_score, S[:5]


def circular_structure_test(vectors_5xh):
    """检测5个向量是否形成C5循环结构

    返回:
    - adj_sim: 相邻动的平均余弦相似度
    - nonadj_sim: 非相邻动的平均余弦相似度
    - circular_ratio: adj/nonadj (>1 = 有循环结构)
    - phase_order: 相位排序得分
    """
    # 归一化
    norms = np.linalg.norm(vectors_5xh, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vectors_5xh / norms

    # 余弦相似度矩阵 [5, 5]
    sim_matrix = normalized @ normalized.T

    # 相邻平均
    adj_sims = [sim_matrix[i, j] for i, j in C5_ADJACENT]
    adj_mean = np.mean(adj_sims)

    # 非相邻平均
    nonadj_sims = [sim_matrix[i, j] for i, j in C5_NONADJ]
    nonadj_mean = np.mean(nonadj_sims)

    # 循环比
    circular_ratio = adj_mean / max(nonadj_mean, 1e-10)

    # 相位排序: 如果5个向量按C5顺序排列, 相邻应该最近
    # 对每个向量, 最近邻是否是C5邻居?
    nearest_correct = 0
    for i in range(5):
        sims = sim_matrix[i].copy()
        sims[i] = -999  # 排除自身
        nearest = np.argmax(sims)
        if nearest in [(i+1)%5, (i-1)%5]:  # C5邻居
            nearest_correct += 1

    return {
        'adj_sim': float(adj_mean),
        'nonadj_sim': float(nonadj_mean),
        'circular_ratio': float(circular_ratio),
        'nearest_c5_count': nearest_correct,  # 0-5
        'sim_matrix': sim_matrix.tolist(),
    }


def dft_cycle_test(vectors_5xh):
    """DFT分析: 5个向量的序列是否有5-cycle频率

    把5个向量的每个维度看作长度5的信号, 做5点DFT
    如果k=1分量(5-cycle)主导, 说明存在C5旋转结构
    """
    # vectors_5xh: [5, hidden]
    # 对每个hidden维度做5点DFT
    n = 5
    # DFT矩阵
    W = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)

    # 投影: [5, hidden] → DFT系数 [5, hidden]
    dft_coeffs = W @ vectors_5xh  # [5, hidden] complex

    # 每个频率的能量
    freq_energy = np.zeros(n)
    for k in range(n):
        freq_energy[k] = np.mean(np.abs(dft_coeffs[k])**2)

    total_energy = freq_energy.sum()
    if total_energy < 1e-20:
        return {'k1_ratio': 0.0, 'freq_energy': freq_energy.tolist()}

    # k=1是5-cycle主频 (k=0是DC/均值)
    # k=1和k=4是共轭对(C5的DFT性质), 合并
    k1_ratio = (freq_energy[1] + freq_energy[4]) / total_energy

    return {
        'k1_ratio': float(k1_ratio),  # 5-cycle能量占比, >0.4为显著
        'freq_energy': freq_energy.tolist(),
    }


# ============================================================================
# 主实验
# ============================================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("Five-Motion Phase Structure Experiment")
    print("验证: 同一层内五动是否形成C5旋转相位")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"strength={STRENGTH}")
    print()

    # ===== 加载模型 =====
    print("[0] 加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ 模型路径不存在: {MODEL_PATH}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size

    print(f"  配置: layers={num_layers}, heads={num_heads}, hidden={hidden_size}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    layer_alphas = compute_layer_alphas(num_layers, STRENGTH)
    motion_assign = layer_motion_assignment(num_layers)

    # ======================================================================
    # Phase 1: 采集每个motion在每层的hidden state
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 1: 采集5动×5prompt在每层的hidden state")
    print("=" * 70)

    # motion_hs_std[motion_idx][layer_idx] = list of last-token hidden vectors
    # motion_hs_phi[motion_idx][layer_idx] = list of last-token hidden vectors
    motion_hs_std = [[[] for _ in range(num_layers)] for _ in range(5)]
    motion_hs_phi = [[[] for _ in range(num_layers)] for _ in range(5)]

    total_prompts = sum(len(v) for v in PROMPTS.values())
    count = 0

    for mi, mk in enumerate(MOTION_KEYS):
        for pi, prompt in enumerate(PROMPTS[mk]):
            count += 1
            print(f"  [{count}/{total_prompts}] {MOTION_LABELS[mi]} prompt#{pi+1}")

            inputs = tokenizer(prompt, return_tensors="pt")

            # 标准模型
            captures = {}
            hooks = apply_capture_hooks(model, captures)
            with torch.no_grad():
                _ = model(**inputs)
            remove_hooks(hooks)

            for li in range(num_layers):
                if li in captures:
                    hs = captures[li]
                    if hs.dim() == 2:
                        hs = hs.unsqueeze(0)
                    # 取最后一个token
                    vec = hs[0, -1, :].numpy()
                    motion_hs_std[mi][li].append(vec)
            del captures

            # φ-Residual模型
            captures = {}
            hooks_both = []
            # 先注册capture
            for i, layer in enumerate(model.model.layers):
                hooks_both.append(layer.register_forward_hook(
                    make_layer_capture_hook(captures, i)
                ))
            # 再注册缩放
            for i, layer in enumerate(model.model.layers):
                a = layer_alphas[i]
                hooks_both.append(layer.self_attn.register_forward_hook(make_attn_scale(a)))
                hooks_both.append(layer.mlp.register_forward_hook(make_mlp_scale(a)))

            with torch.no_grad():
                _ = model(**inputs)
            remove_hooks(hooks_both)

            for li in range(num_layers):
                if li in captures:
                    hs = captures[li]
                    if hs.dim() == 2:
                        hs = hs.unsqueeze(0)
                    vec = hs[0, -1, :].numpy()
                    motion_hs_phi[mi][li].append(vec)
            del captures

    # 平均: 每个motion每层一个向量 [5, num_layers, hidden]
    motion_avg_std = np.zeros((5, num_layers, hidden_size))
    motion_avg_phi = np.zeros((5, num_layers, hidden_size))
    for mi in range(5):
        for li in range(num_layers):
            if motion_hs_std[mi][li]:
                motion_avg_std[mi, li] = np.mean(motion_hs_std[mi][li], axis=0)
            if motion_hs_phi[mi][li]:
                motion_avg_phi[mi, li] = np.mean(motion_hs_phi[mi][li], axis=0)

    print(f"\n  Phase 1完成! 采集 {total_prompts}×2 次forward")

    # ======================================================================
    # Phase 2: 层内C5相位分析
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 2: 层内C5相位分析")
    print("=" * 70)

    results = {'std': {}, 'phi': {}, 'layers': num_layers}

    for tag, motion_avg in [("std", motion_avg_std), ("phi", motion_avg_phi)]:
        print(f"\n  --- {'标准模型' if tag == 'std' else 'φ-Residual模型'} ---")

        layer_results = {}

        # 全局统计
        pentagon_scores = []
        circular_ratios = []
        k1_ratios = []
        nearest_c5_counts = []

        # 重点分析层(最后1/3, 以及首尾)
        focus_layers = list(range(num_layers))
        # 但也打印关键层
        key_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1,
                      num_layers-2, num_layers-3, num_layers-4, num_layers-5]

        for li in focus_layers:
            # 5个motion在该层的hidden state: [5, hidden]
            vecs = motion_avg[:, li, :]

            # 去掉零向量(没抓到的层)
            if np.max(np.abs(vecs)) < 1e-10:
                continue

            # PCA分析
            coords_2d, pentagon_score, singular_vals = pca_analysis(vecs)

            # 循环结构检测
            circ = circular_structure_test(vecs)

            # DFT分析
            dft = dft_cycle_test(vecs)

            layer_results[li] = {
                'pentagon_score': pentagon_score,
                'circular_ratio': circ['circular_ratio'],
                'adj_sim': circ['adj_sim'],
                'nonadj_sim': circ['nonadj_sim'],
                'nearest_c5': circ['nearest_c5_count'],
                'k1_ratio': dft['k1_ratio'],
                'var_ratio': float(singular_vals[0]**2 / max(sum(singular_vals**2), 1e-10)),
            }

            pentagon_scores.append(pentagon_score)
            circular_ratios.append(circ['circular_ratio'])
            k1_ratios.append(dft['k1_ratio'])
            nearest_c5_counts.append(circ['nearest_c5_count'])

            # 打印关键层
            if li in key_layers:
                motion_label = MOTION_LABELS[motion_assign[li]]
                print(f"\n  Layer {li} [{motion_label}]:")
                print(f"    五边形得分: {pentagon_score:.4f}")
                print(f"    循环比(adj/nonadj): {circ['circular_ratio']:.3f} "
                      f"(adj={circ['adj_sim']:.4f}, nonadj={circ['nonadj_sim']:.4f})")
                print(f"    最近邻C5正确: {circ['nearest_c5_count']}/5")
                print(f"    DFT k=1占比: {dft['k1_ratio']:.4f}")
                print(f"    PC1方差占比: {singular_vals[0]**2 / max(sum(singular_vals**2), 1e-10):.4f}")

                # 打印相似度矩阵
                sim = np.array(circ['sim_matrix'])
                print(f"    相似度矩阵:")
                for row_i in range(5):
                    print(f"      {MOTION_LABELS[row_i]}: "
                          f"{' '.join(f'{sim[row_i,j]:.3f}' for j in range(5))}")

        results[tag] = layer_results

        # 汇总
        if pentagon_scores:
            print(f"\n  === 汇总 ({'标准' if tag=='std' else 'φ-Residual'}模型) ===")
            print(f"  五边形得分: mean={np.mean(pentagon_scores):.4f}, "
                  f"max={np.max(pentagon_scores):.4f}, "
                  f"layers>0.5: {sum(1 for s in pentagon_scores if s > 0.5)}/{len(pentagon_scores)}")
            print(f"  循环比(adj/nonadj): mean={np.mean(circular_ratios):.3f}, "
                  f">1.0层数: {sum(1 for r in circular_ratios if r > 1.0)}/{len(circular_ratios)}")
            print(f"  DFT k=1占比: mean={np.mean(k1_ratios):.4f}, "
                  f">0.4层数: {sum(1 for r in k1_ratios if r > 0.4)}/{len(k1_ratios)}")
            print(f"  最近邻C5正确: mean={np.mean(nearest_c5_counts):.1f}/5, "
                  f"满分层数: {sum(1 for c in nearest_c5_counts if c >= 4)}/{len(nearest_c5_counts)}")

    # ======================================================================
    # Phase 3: 对比分析 (std vs phi)
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 3: 标准模型 vs φ-Residual 对比")
    print("=" * 70)

    std_r = results.get('std', {})
    phi_r = results.get('phi', {})

    common_layers = sorted(set(std_r.keys()) & set(phi_r.keys()))

    if common_layers:
        # 哪个模型的C5结构更强?
        phi_better_pentagon = 0
        phi_better_circular = 0
        phi_better_k1 = 0
        phi_better_nearest = 0

        for li in common_layers:
            s = std_r[li]
            p = phi_r[li]
            if p['pentagon_score'] > s['pentagon_score']:
                phi_better_pentagon += 1
            if p['circular_ratio'] > s['circular_ratio']:
                phi_better_circular += 1
            if p['k1_ratio'] > s['k1_ratio']:
                phi_better_k1 += 1
            if p['nearest_c5'] > s['nearest_c5']:
                phi_better_nearest += 1

        n = len(common_layers)
        print(f"\n  φ-Residual更强(共{n}层):")
        print(f"    五边形得分: {phi_better_pentagon}/{n} ({100*phi_better_pentagon/n:.0f}%)")
        print(f"    循环比:     {phi_better_circular}/{n} ({100*phi_better_circular/n:.0f}%)")
        print(f"    DFT k=1:    {phi_better_k1}/{n} ({100*phi_better_k1/n:.0f}%)")
        print(f"    最近邻C5:   {phi_better_nearest}/{n} ({100*phi_better_nearest/n:.0f}%)")

        # 找C5结构最强的层
        print(f"\n  C5结构最强层 (φ-Residual):")
        for metric in ['pentagon_score', 'circular_ratio', 'k1_ratio']:
            best_li = max(common_layers, key=lambda li: phi_r[li][metric])
            print(f"    {metric}: Layer {best_li} = {phi_r[best_li][metric]:.4f}")

    # ======================================================================
    # Phase 4: 全局C5结构验证
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 4: 全局C5结构验证")
    print("=" * 70)

    for tag, motion_avg in [("std", motion_avg_std), ("phi", motion_avg_phi)]:
        # 所有层拼接: [5, num_layers * hidden_size]
        all_vecs = motion_avg.reshape(5, -1)

        circ = circular_structure_test(all_vecs)
        dft = dft_cycle_test(all_vecs)
        coords_2d, pentagon_score, sv = pca_analysis(all_vecs)

        label = "标准模型" if tag == "std" else "φ-Residual模型"
        print(f"\n  {label} (全层拼接):")
        print(f"    五边形得分: {pentagon_score:.4f}")
        print(f"    循环比: {circ['circular_ratio']:.3f}")
        print(f"    最近邻C5: {circ['nearest_c5_count']}/5")
        print(f"    DFT k=1占比: {dft['k1_ratio']:.4f}")
        print(f"    相似度矩阵:")
        sim = np.array(circ['sim_matrix'])
        for row_i in range(5):
            print(f"      {MOTION_LABELS[row_i]}: "
                  f"{' '.join(f'{sim[row_i,j]:.3f}' for j in range(5))}")

    # ======================================================================
    # 结论
    # ======================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    # 统计关键指标
    phi_layer_data = results.get('phi', {})
    if phi_layer_data:
        layers_with_circular = sum(1 for li, d in phi_layer_data.items() if d['circular_ratio'] > 1.0)
        layers_with_k1 = sum(1 for li, d in phi_layer_data.items() if d['k1_ratio'] > 0.4)
        layers_with_pentagon = sum(1 for li, d in phi_layer_data.items() if d['pentagon_score'] > 0.5)
        total = len(phi_layer_data)

        print(f"\n  φ-Residual模型关键指标:")
        print(f"    循环比>1.0: {layers_with_circular}/{total} 层")
        print(f"    DFT k=1>0.4: {layers_with_k1}/{total} 层")
        print(f"    五边形>0.5: {layers_with_pentagon}/{total} 层")

        if layers_with_circular > total * 0.5:
            print(f"\n  ✅ 多数层存在C5循环结构: 相邻动比非相邻更相似")
        elif layers_with_circular > total * 0.2:
            print(f"\n  ⚠️ 部分层存在弱C5结构, 需进一步验证")
        else:
            print(f"\n  ❌ 未检测到显著C5循环结构")

    elapsed = time.time() - start_time
    print(f"\n  总耗时: {elapsed/60:.1f}分钟")

    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(output_dir, "five_motion_phase_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, default=convert, indent=2, ensure_ascii=False)

    print(f"\n  结果已保存到: {os.path.join(output_dir, 'five_motion_phase_results.json')}")


if __name__ == "__main__":
    main()
