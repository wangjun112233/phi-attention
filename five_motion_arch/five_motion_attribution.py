"""
Five-Motion Attribution Patching Experiment
============================================
验证φ-Residual改造后的Qwen2.5-3B模型中，
五动(认/遇/落/裂/余)是否对应5个独立的功能parcel。

Phase 1: Prompt设计 - 5类motion各5个prompt+minimal pair (共25对)
Phase 2: Attribution Patching - 记录标准/改造模型每层activation difference
Phase 3: Parcel分析 - 5×5层重叠度矩阵(Jaccard), 标准vs改造对比
Phase 4: 因果验证(ablation) - double dissociation (forward pass + generate)
Phase 5: 可视化与报告 - markdown报告输出

用法:
  python five_motion_attribution.py [model_path] [strength] [top_k_pct] [ablation_top_n] [max_new_tokens]

默认值:
  model_path  = C:\\Users\\WANGJUN\\d10\\ms_cache\\Qwen\\Qwen2.5-3B
  strength    = 0.05
  top_k_pct   = 0.25   (top 25%层视为重要层)
  ablation_top_n = 5   (ablation时每motion关5个关键层)
  max_new_tokens = 30

运行环境: Windows, CPU-only, torch.float32, 16GB RAM
预计耗时: 约15-30分钟 (纯CPU)
"""

import os
import sys
import math
import json
import time
from datetime import datetime
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============================================================================
# 参数
# ============================================================================
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\WANGJUN\d10\ms_cache\Qwen\Qwen2.5-3B"
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
TOP_K_PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
ABLATION_TOP_N = int(sys.argv[4]) if len(sys.argv) > 4 else 5
MAX_NEW_TOKENS = int(sys.argv[5]) if len(sys.argv) > 5 else 30

# φ常数
PHI = (1 + 5**0.5) / 2
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}

# 五动名称
MOTION_NAMES = ["认(Recognize)", "遇(Encounter)", "落(Settle)", "裂(Split)", "余(Residue)"]
MOTION_KEYS = ["ren", "yu", "luo", "lie", "yuu"]
MOTION_LABELS = ["认", "遇", "落", "裂", "余"]

# ============================================================================
# Phase 1: Prompt设计
# ============================================================================
# 每个motion: 5个prompt对 (original, alternative/minimal pair)
# original激活该motion, alternative翻转关键元素以抑制该motion

PROMPTS = {
    "ren": [
        ("The pattern in this sequence 2,4,8,16 is clearly",
         "The pattern in this sequence 2,4,6,8 is clearly"),
        ("Looking at the data, I can identify that",
         "Looking at the noise, I cannot identify"),
        ("The key feature that distinguishes this object is",
         "The key feature that all objects share is"),
        ("Classification: this specimen belongs to",
         "This specimen cannot be classified into"),
        ("The fingerprint of this chemical reaction is",
         "The fingerprint of this chemical reaction could be anything"),
    ],
    "yu": [
        ("When I encountered this problem, the first connection I made was",
         "When I saw this problem, no connection came to mind"),
        ("The unexpected link between these two phenomena is",
         "These two phenomena are completely unrelated"),
        ("This reminds me of something I saw before:",
         "This is entirely new, nothing reminds me of"),
        ("By associating X with Y, we discover that",
         "X and Y have nothing to do with each other"),
        ("The surprising intersection of these ideas creates",
         "These ideas remain separate and do not intersect"),
    ],
    "luo": [
        ("After long deliberation, the committee finally settled on",
         "The committee continued to deliberate without settling"),
        ("The solution converges to a stable point at",
         "The solution diverges and never stabilizes"),
        ("When the dust settles, what remains is",
         "The dust never settles and everything stays uncertain"),
        ("The system reaches equilibrium when",
         "The system never reaches equilibrium"),
        ("After all the noise fades, the core truth is",
         "The noise never fades and no core truth emerges"),
    ],
    "lie": [
        ("The crucial difference between these two approaches is",
         "These two approaches are essentially the same"),
        ("What separates successful strategies from failures is",
         "Successful and failed strategies overlap completely"),
        ("The boundary that divides these two categories is",
         "No clear boundary exists between these categories"),
        ("By negating assumption X, we discover that",
         "Negating assumption X changes nothing"),
        ("The contradiction in this argument reveals",
         "There is no contradiction in this argument"),
    ],
    "yuu": [
        ("Even years later, the lasting impact of that event",
         "That event had no lasting impact"),
        ("The residual effect of the treatment persists as",
         "The treatment leaves no residual effect"),
        ("What remains after removing all the noise is",
         "After removing the noise, nothing remains"),
        ("The echo of that decision still influences",
         "That decision has no remaining influence"),
        ("Despite the passage of time, the underlying pattern persists as",
         "The pattern has completely disappeared over time"),
    ],
}

# ============================================================================
# 工具函数
# ============================================================================

def compute_layer_alphas(nl, strength):
    """计算每层的φ-Residual缩放因子α

    层i的k = i % 5 映射到5个φ幂次，对应5个motion:
      k=0→认(φ⁰=1.0), k=1→遇(φ⁻¹), k=2→落(φ⁻²), k=3→裂(φ¹), k=4→余(φ⁻³)
    奇数组翻转(2.0-a)确保多样性
    """
    alphas = []
    for i in range(nl):
        k = i % 5
        g = i // 5 % 2
        ba = PHI_POWERS[k] / math.sqrt(i + 2)
        a = 1.0 + strength * (ba - 1.0)
        if g == 1:
            a = 2.0 - a
        alphas.append(a)
    return alphas


def layer_motion_assignment(nl):
    """每层属于哪个motion parcel (k=i%5)
    返回: list[int], 每层对应的motion index (0=认,1=遇,2=落,3=裂,4=余)
    """
    return [i % 5 for i in range(nl)]


# ---- Hook工厂 ----

def make_attn_scale(alpha):
    """Qwen2.5 Attention缩放hook: 缩放attn输出的hidden_states部分
    Qwen2Attention返回2元素tuple: (attn_output, past_key_value)
    也兼容3元素tuple: (attn_output, attn_weights, past_key_value)
    """
    def hook(m, inp, out):
        if isinstance(out, tuple):
            scaled = alpha * out[0]
            return (scaled,) + out[1:]
        return alpha * out
    return hook


def make_mlp_scale(alpha):
    """MLP缩放hook"""
    def hook(m, inp, out):
        return alpha * out
    return hook


def apply_fixed_hooks(model, alphas):
    """对attn/mlp子模块应用固定α缩放hook"""
    hooks = []
    for i, layer in enumerate(model.model.layers):
        a = alphas[i]
        hooks.append(layer.self_attn.register_forward_hook(make_attn_scale(a)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_scale(a)))
    return hooks


def remove_hooks(hooks):
    """移除所有hook"""
    for h in hooks:
        h.remove()


def make_layer_capture_hook(captures, layer_idx):
    """在layer输出处捕获residual stream (hidden_states)
    Qwen2DecoderLayer输出: (hidden_states, attn_weights, past_kv)
    """
    def hook(module, input, output):
        # output[0] = hidden_states after this layer
        captures[layer_idx] = output[0].detach().clone()
    return hook


def apply_capture_hooks(model, captures):
    """在每层输出处注册capture hook"""
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(
            make_layer_capture_hook(captures, i)
        ))
    return hooks


def apply_fixed_plus_capture(model, alphas, captures):
    """同时应用φ缩放hook + capture hook

    执行顺序（由forward pass结构决定，与注册顺序无关）:
    1. Layer forward开始 → self_attn forward → attn缩放hook执行
    2. → mlp forward → mlp缩放hook执行
    3. → Layer forward结束 → layer capture hook执行
    因此capture hook看到的是缩放后的最终hidden_states
    """
    hooks = []
    # Capture hooks (注册在layer上, 在layer forward完成后触发)
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(
            make_layer_capture_hook(captures, i)
        ))
    # 缩放hooks (注册在attn/mlp子模块上, 在子模块forward完成后触发)
    for i, layer in enumerate(model.model.layers):
        a = alphas[i]
        hooks.append(layer.self_attn.register_forward_hook(make_attn_scale(a)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_scale(a)))
    return hooks


# ---- 激活差异计算 ----

def compute_activation_diff(captures_orig, captures_alt, num_layers):
    """计算每层activation difference: L2 norm of (orig - alt)

    Args:
        captures_orig: dict[layer_idx] = tensor(shape=[1, seq_len, hidden])
        captures_alt:  dict[layer_idx] = tensor(shape=[1, seq_len, hidden])
        num_layers: int

    Returns:
        list[float]: 每层的activation difference (Frobenius norm)
    """
    diffs = []
    for i in range(num_layers):
        if i in captures_orig and i in captures_alt:
            t_orig = captures_orig[i].float()
            t_alt = captures_alt[i].float()
            # Ensure 3D: [batch, seq_len, hidden]
            if t_orig.dim() == 2:
                t_orig = t_orig.unsqueeze(0)
            if t_alt.dim() == 2:
                t_alt = t_alt.unsqueeze(0)
            # Align seq_len: trim to the shorter one
            min_len = min(t_orig.shape[1], t_alt.shape[1])
            diff = (t_orig[:, :min_len, :] - t_alt[:, :min_len, :]).norm().item()
            diffs.append(diff)
        else:
            diffs.append(0.0)
    return diffs


def compute_top_k_layers(diffs, k_pct):
    """返回activation difference最大的top-k%层的索引集合"""
    n = len(diffs)
    k = max(1, int(n * k_pct))
    indexed = sorted(enumerate(diffs), key=lambda x: x[1], reverse=True)
    return set(i for i, _ in indexed[:k])


def jaccard_similarity(set_a, set_b):
    """Jaccard相似度"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---- Forward pass + last hidden state ----

def forward_last_hidden(model, tokenizer, prompt, num_layers):
    """单次forward pass, 返回最后一层在最后一个token位置的hidden state

    Args:
        model: transformers模型
        tokenizer: 分词器
        prompt: 输入文本
        num_layers: 层数

    Returns:
        tensor: shape [1, hidden_size], 最后一层最后一个token的hidden state
    """
    captures = {}
    hooks = apply_capture_hooks(model, captures)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)
    remove_hooks(hooks)
    last_layer_hs = captures.get(num_layers - 1, None)
    if last_layer_hs is not None:
        # 取最后一个token: shape [1, hidden_size]
        last_layer_hs = last_layer_hs[:, -1, :].detach().clone()
    # 释放其余层的capture
    del captures
    return last_layer_hs


def cosine_sim(t1, t2):
    """两个tensor的余弦相似度 (flatten后)"""
    if t1 is None or t2 is None:
        return 0.0
    v1 = t1.flatten().float()
    v2 = t2.flatten().float()
    sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return sim


def generate_simple(model, tokenizer, prompt, max_new_tokens=30):
    """简单generate, 返回生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# ============================================================================
# 主实验
# ============================================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("Five-Motion Attribution Patching Experiment")
    print("φ-Residual Qwen2.5-3B: 认/遇/落/裂/余 Parcel验证")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"strength={STRENGTH}, top_k_pct={TOP_K_PCT}")
    print(f"ablation_top_n={ABLATION_TOP_N}, max_new_tokens={MAX_NEW_TOKENS}")
    print()

    # ===== 加载模型 =====
    print("[0] 加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ 模型路径不存在: {MODEL_PATH}")
        print("  请确认路径后重试")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"  # 确保eager模式, 兼容hook

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    print(f"  配置: layers={num_layers}, heads={num_heads}, "
          f"kv_heads={num_kv_heads}, hidden={hidden_size}, head_dim={head_dim}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 计算α
    layer_alphas = compute_layer_alphas(num_layers, STRENGTH)
    motion_assign = layer_motion_assignment(num_layers)

    print(f"\n  α分布(采样): ", end="")
    for li in [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]:
        print(f"L{li}[{MOTION_LABELS[motion_assign[li]]}]={layer_alphas[li]:.4f} ", end="")
    print()

    # ======================================================================
    # Phase 2: Attribution Patching
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Attribution Patching (Activation Difference)")
    print("=" * 70)

    # 存储: (motion_key, prompt_idx) -> per-layer diff
    std_diffs = defaultdict(dict)
    phi_diffs = defaultdict(dict)

    total_pairs = sum(len(v) for v in PROMPTS.values())
    pair_count = 0

    for mk in MOTION_KEYS:
        prompts = PROMPTS[mk]
        for pi, (orig, alt) in enumerate(prompts):
            pair_count += 1
            mi = MOTION_KEYS.index(mk)
            print(f"\n  [{pair_count}/{total_pairs}] {MOTION_NAMES[mi]} prompt#{pi+1}")
            print(f"    原始: {orig[:60]}...")
            print(f"    对比: {alt[:60]}...")

            inputs_orig = tokenizer(orig, return_tensors="pt")
            inputs_alt = tokenizer(alt, return_tensors="pt")

            # ---- 标准模型: forward pass original + alternative ----
            captures_orig_std = {}
            hooks_cap = apply_capture_hooks(model, captures_orig_std)
            with torch.no_grad():
                _ = model(**inputs_orig)
            remove_hooks(hooks_cap)

            captures_alt_std = {}
            hooks_cap = apply_capture_hooks(model, captures_alt_std)
            with torch.no_grad():
                _ = model(**inputs_alt)
            remove_hooks(hooks_cap)

            std_diff = compute_activation_diff(captures_orig_std, captures_alt_std, num_layers)
            std_diffs[(mk, pi)] = std_diff

            del captures_orig_std, captures_alt_std

            # ---- φ-Residual模型: forward pass original + alternative ----
            captures_orig_phi = {}
            hooks_both = apply_fixed_plus_capture(model, layer_alphas, captures_orig_phi)
            with torch.no_grad():
                _ = model(**inputs_orig)
            remove_hooks(hooks_both)

            captures_alt_phi = {}
            hooks_both = apply_fixed_plus_capture(model, layer_alphas, captures_alt_phi)
            with torch.no_grad():
                _ = model(**inputs_alt)
            remove_hooks(hooks_both)

            phi_diff = compute_activation_diff(captures_orig_phi, captures_alt_phi, num_layers)
            phi_diffs[(mk, pi)] = phi_diff

            del captures_orig_phi, captures_alt_phi

            # 打印摘要
            top3_std = sorted(range(num_layers), key=lambda i: std_diff[i], reverse=True)[:3]
            top3_phi = sorted(range(num_layers), key=lambda i: phi_diff[i], reverse=True)[:3]
            print(f"    标准模型 top3层: {top3_std} (diffs: {[f'{std_diff[i]:.2f}' for i in top3_std]})")
            print(f"    φ-模型   top3层: {top3_phi} (diffs: {[f'{phi_diff[i]:.2f}' for i in top3_phi]})")

    print(f"\n  Phase 2完成! 共处理 {pair_count} 个prompt对")

    # ======================================================================
    # Phase 3: Parcel分析
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 3: Parcel分析 (层重叠度矩阵)")
    print("=" * 70)

    # 聚合每个motion的top-k%重要层 (5个prompt取并集)
    std_top_sets = {}
    phi_top_sets = {}

    # 每个motion每层的平均diff
    std_motion_avg = {}
    phi_motion_avg = {}

    for mk in MOTION_KEYS:
        prompts = PROMPTS[mk]
        std_union = set()
        phi_union = set()
        std_avg = [0.0] * num_layers
        phi_avg = [0.0] * num_layers

        for pi in range(len(prompts)):
            std_top = compute_top_k_layers(std_diffs[(mk, pi)], TOP_K_PCT)
            phi_top = compute_top_k_layers(phi_diffs[(mk, pi)], TOP_K_PCT)
            std_union |= std_top
            phi_union |= phi_top

            for li in range(num_layers):
                std_avg[li] += std_diffs[(mk, pi)][li]
                phi_avg[li] += phi_diffs[(mk, pi)][li]

        for li in range(num_layers):
            std_avg[li] /= len(prompts)
            phi_avg[li] /= len(prompts)

        std_top_sets[mk] = std_union
        phi_top_sets[mk] = phi_union
        std_motion_avg[mk] = std_avg
        phi_motion_avg[mk] = phi_avg

        print(f"\n  {MOTION_NAMES[MOTION_KEYS.index(mk)]}:")
        print(f"    标准模型 top层({len(std_union)}个): {sorted(std_union)}")
        print(f"    φ-模型   top层({len(phi_union)}个): {sorted(phi_union)}")

    # ---- 5×5 重叠度矩阵 (Jaccard) ----
    def print_overlap_matrix(overlap_matrix, label):
        print(f"\n  --- {label} 5×5 Jaccard重叠矩阵 ---")
        print(f"  {'':>12}", end="")
        for mk in MOTION_KEYS:
            print(f"  {MOTION_LABELS[MOTION_KEYS.index(mk)]:>6}", end="")
        print()
        for mi, mk1 in enumerate(MOTION_KEYS):
            print(f"  {MOTION_LABELS[mi]:>12}", end="")
            for mj, mk2 in enumerate(MOTION_KEYS):
                sim = overlap_matrix[mi][mj]
                marker = "★★" if mi == mj and sim > 0.3 else ("★" if mi == mj else "")
                print(f"  {sim:.3f}{marker}", end="")
            print()

    std_overlap_matrix = [[jaccard_similarity(std_top_sets[MOTION_KEYS[i]], std_top_sets[MOTION_KEYS[j]])
                           for j in range(5)] for i in range(5)]
    phi_overlap_matrix = [[jaccard_similarity(phi_top_sets[MOTION_KEYS[i]], phi_top_sets[MOTION_KEYS[j]])
                           for j in range(5)] for i in range(5)]

    print_overlap_matrix(std_overlap_matrix, "标准模型")
    print_overlap_matrix(phi_overlap_matrix, "φ-Residual模型")

    # ---- 对角线平均 vs 非对角线平均 ----
    std_diag = [std_overlap_matrix[i][i] for i in range(5)]
    std_offdiag = [std_overlap_matrix[i][j] for i in range(5) for j in range(5) if i != j]
    phi_diag = [phi_overlap_matrix[i][i] for i in range(5)]
    phi_offdiag = [phi_overlap_matrix[i][j] for i in range(5) for j in range(5) if i != j]

    std_diag_avg = sum(std_diag) / len(std_diag)
    std_offdiag_avg = sum(std_offdiag) / len(std_offdiag) if std_offdiag else 0
    phi_diag_avg = sum(phi_diag) / len(phi_diag)
    phi_offdiag_avg = sum(phi_offdiag) / len(phi_offdiag) if phi_offdiag else 0

    std_contrast = std_diag_avg / std_offdiag_avg if std_offdiag_avg > 0 else float('inf')
    phi_contrast = phi_diag_avg / phi_offdiag_avg if phi_offdiag_avg > 0 else float('inf')

    print(f"\n  标准模型: 对角线均值={std_diag_avg:.3f}, 非对角线均值={std_offdiag_avg:.3f}, "
          f"对比度={std_contrast:.2f}x")
    print(f"  φ-模型:   对角线均值={phi_diag_avg:.3f}, 非对角线均值={phi_offdiag_avg:.3f}, "
          f"对比度={phi_contrast:.2f}x")

    parcel_separation = (phi_diag_avg - phi_offdiag_avg) - (std_diag_avg - std_offdiag_avg)
    if parcel_separation > 0.05:
        print(f"  ★★★ φ-Residual增强了parcel分离度 (+{parcel_separation:.3f})")
    elif parcel_separation > 0:
        print(f"  ★ φ-Residual轻微增强了parcel分离度 (+{parcel_separation:.3f})")
    else:
        print(f"  · φ-Residual未增强parcel分离度 ({parcel_separation:.3f})")

    # ---- 每层的motion偏好 ----
    print("\n  --- 每层motion偏好 (哪个motion在该层最活跃) ---")
    layer_preference_std = []
    layer_preference_phi = []
    for li in range(num_layers):
        std_max_mk = max(MOTION_KEYS, key=lambda mk: std_motion_avg[mk][li])
        phi_max_mk = max(MOTION_KEYS, key=lambda mk: phi_motion_avg[mk][li])
        layer_preference_std.append(std_max_mk)
        layer_preference_phi.append(phi_max_mk)

    std_pref_count = defaultdict(int)
    phi_pref_count = defaultdict(int)
    for li in range(num_layers):
        std_pref_count[layer_preference_std[li]] += 1
        phi_pref_count[layer_preference_phi[li]] += 1

    print(f"  标准模型层偏好分布: ", end="")
    for mk in MOTION_KEYS:
        print(f"{MOTION_LABELS[MOTION_KEYS.index(mk)]}={std_pref_count[mk]} ", end="")
    print()
    print(f"  φ-模型层偏好分布:   ", end="")
    for mk in MOTION_KEYS:
        print(f"{MOTION_LABELS[MOTION_KEYS.index(mk)]}={phi_pref_count[mk]} ", end="")
    print()

    std_assign_match = sum(1 for li in range(num_layers)
                          if MOTION_KEYS[motion_assign[li]] == layer_preference_std[li])
    phi_assign_match = sum(1 for li in range(num_layers)
                          if MOTION_KEYS[motion_assign[li]] == layer_preference_phi[li])
    print(f"\n  与α赋值(i%5)一致性: 标准模型 {std_assign_match}/{num_layers} "
          f"({100*std_assign_match/num_layers:.0f}%), "
          f"φ-模型 {phi_assign_match}/{num_layers} "
          f"({100*phi_assign_match/num_layers:.0f}%)")

    # ======================================================================
    # Phase 4: 因果验证 (Ablation / Double Dissociation)
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 4: 因果验证 (Ablation)")
    print("=" * 70)

    # 识别每个motion的top-N关键层 (基于φ模型平均diff)
    motion_key_layers = {}
    for mk in MOTION_KEYS:
        avg_diffs = phi_motion_avg[mk]
        top_n_indices = sorted(range(num_layers), key=lambda i: avg_diffs[i], reverse=True)[:ABLATION_TOP_N]
        motion_key_layers[mk] = top_n_indices
        print(f"\n  {MOTION_NAMES[MOTION_KEYS.index(mk)]} 关键层: {top_n_indices} "
              f"(avg_diff: {[f'{avg_diffs[i]:.2f}' for i in top_n_indices]})")

    # ---- Double Dissociation测试 ----
    # 策略: 用forward pass获取last hidden state, 计算cosine distance
    # 比generate()快得多(~5s vs ~150s per prompt), 且输入一致便于公平比较

    print("\n  --- Double Dissociation测试 (forward pass) ---")

    # 选取每个motion的第1个prompt作为代表
    ablation_prompts = {mk: PROMPTS[mk][0][0] for mk in MOTION_KEYS}

    # Step 1: Baseline hidden states (标准模型, 无hook)
    print("\n  [4a] 获取baseline hidden states...")
    baseline_hs = {}
    for mk in MOTION_KEYS:
        baseline_hs[mk] = forward_last_hidden(model, tokenizer, ablation_prompts[mk], num_layers)
        print(f"    {MOTION_LABELS[MOTION_KEYS.index(mk)]}: norm={baseline_hs[mk].norm().item():.2f}")

    # Step 2: Full-φ hidden states
    print("\n  [4b] 获取full-φ hidden states...")
    hooks_phi = apply_fixed_hooks(model, layer_alphas)
    phi_hs = {}
    for mk in MOTION_KEYS:
        phi_hs[mk] = forward_last_hidden(model, tokenizer, ablation_prompts[mk], num_layers)
    remove_hooks(hooks_phi)

    # 计算full-φ effect (cosine distance to baseline)
    phi_effect = {}
    for mk in MOTION_KEYS:
        phi_effect[mk] = 1.0 - cosine_sim(baseline_hs[mk], phi_hs[mk])
    print(f"  Full-φ效应(cosine distance): ", end="")
    for mk in MOTION_KEYS:
        print(f"{MOTION_LABELS[MOTION_KEYS.index(mk)]}={phi_effect[mk]:.4f} ", end="")
    print()

    # 释放full-φ hidden states (不再需要)
    del phi_hs

    # Step 3: Ablation — 对每个motion关掉其关键层
    # ablation_result[ablated_motion_key][test_motion_key] = cosine distance to baseline
    ablation_result = {}

    for abl_mk in MOTION_KEYS:
        print(f"\n  [4c] Ablation: 关掉{MOTION_LABELS[MOTION_KEYS.index(abl_mk)]}的关键层 "
              f"{motion_key_layers[abl_mk]}...")

        # 构造ablated alphas: 关键层设为1.0 (无修改)
        ablated_alphas = list(layer_alphas)
        for li in motion_key_layers[abl_mk]:
            ablated_alphas[li] = 1.0

        hooks_abl = apply_fixed_hooks(model, ablated_alphas)

        abl_results = {}
        for test_mk in MOTION_KEYS:
            abl_hs = forward_last_hidden(model, tokenizer, ablation_prompts[test_mk], num_layers)
            cos_dist = 1.0 - cosine_sim(baseline_hs[test_mk], abl_hs)
            abl_results[test_mk] = cos_dist
            del abl_hs

            # ablation_effect正值 = ablation使输出更接近baseline (减少了φ-effect)
            ablation_effect = phi_effect[test_mk] - cos_dist
            print(f"    测试{MOTION_LABELS[MOTION_KEYS.index(test_mk)]}: "
                  f"cos_dist={cos_dist:.4f} (full-φ={phi_effect[test_mk]:.4f}, "
                  f"Δ_effect={ablation_effect:+.4f})")

        remove_hooks(hooks_abl)
        ablation_result[abl_mk] = abl_results

    # ---- Double Dissociation判定 ----
    print("\n  --- Double Dissociation矩阵 ---")
    print(f"  行=ablated motion, 列=test motion")
    print(f"  值=φ-effect减少量(正值=ablation有效, 负值=ablation反增φ-effect)")
    print(f"  {'':>12}", end="")
    for mk in MOTION_KEYS:
        print(f"  {MOTION_LABELS[MOTION_KEYS.index(mk)]:>8}", end="")
    print()

    dissociation_count = 0
    diag_effects = []
    offdiag_effects = []

    for abl_mk in MOTION_KEYS:
        print(f"  ablate{MOTION_LABELS[MOTION_KEYS.index(abl_mk)]:>5}", end="")
        for test_mk in MOTION_KEYS:
            ablation_eff = phi_effect[test_mk] - ablation_result[abl_mk][test_mk]
            marker = "★" if abl_mk == test_mk and ablation_eff > 0 else ""
            print(f"  {ablation_eff:+.4f}{marker}", end="")
            if abl_mk == test_mk:
                diag_effects.append(ablation_eff)
                if ablation_eff > 0:
                    dissociation_count += 1
            else:
                offdiag_effects.append(ablation_eff)
        print()

    print(f"\n  对角线正值计数(选择性ablation有效): {dissociation_count}/5")

    diag_avg = sum(diag_effects) / len(diag_effects) if diag_effects else 0
    offdiag_avg = sum(offdiag_effects) / len(offdiag_effects) if offdiag_effects else 0

    print(f"  对角线平均效应: {diag_avg:+.4f}, 非对角线平均效应: {offdiag_avg:+.4f}")
    if diag_avg > offdiag_avg + 0.005:
        print(f"  ★★★ Double Dissociation成立! 对角线效应显著大于非对角线")
    elif diag_avg > offdiag_avg:
        print(f"  ★ Double Dissociation弱成立, 对角线效应略大于非对角线")
    else:
        print(f"  · Double Dissociation未成立, 需进一步分析")

    # ---- 生成定性对比文本 (仅baseline + full-φ, 每motion第1个prompt) ----
    print("\n  --- 定性对比: 生成文本 (baseline vs full-φ) ---")
    gen_texts = {}
    for mk in MOTION_KEYS:
        prompt = ablation_prompts[mk]
        # baseline
        base_text = generate_simple(model, tokenizer, prompt, MAX_NEW_TOKENS)
        # full-φ
        hooks_phi = apply_fixed_hooks(model, layer_alphas)
        phi_text = generate_simple(model, tokenizer, prompt, MAX_NEW_TOKENS)
        remove_hooks(hooks_phi)
        gen_texts[mk] = {"baseline": base_text, "phi": phi_text}
        diff = "不同" if base_text != phi_text else "相同"
        print(f"\n    {MOTION_LABELS[MOTION_KEYS.index(mk)]} ({diff}):")
        print(f"      基线: {base_text[:100]}...")
        print(f"      φ版: {phi_text[:100]}...")

    # ======================================================================
    # Phase 5: 报告生成
    # ======================================================================
    print("\n" + "=" * 70)
    print("Phase 5: 生成报告")
    print("=" * 70)

    elapsed = time.time() - start_time

    # 确定输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "five_motion_attribution_report.md")

    report_lines = []
    report_lines.append("# Five-Motion Attribution Patching 实验报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**模型**: Qwen2.5-3B (`{MODEL_PATH}`)")
    report_lines.append(f"**配置**: layers={num_layers}, heads={num_heads}, "
                       f"kv_heads={num_kv_heads}, hidden={hidden_size}")
    report_lines.append(f"**参数**: strength={STRENGTH}, top_k_pct={TOP_K_PCT}, "
                       f"ablation_top_n={ABLATION_TOP_N}")
    report_lines.append(f"**耗时**: {elapsed:.0f}秒 ({elapsed/60:.1f}分钟)")
    report_lines.append("")

    # ---- 实验设计 ----
    report_lines.append("## 实验设计")
    report_lines.append("")
    report_lines.append("五动(认/遇/落/裂/余)分别对应φ-Residual中5个不同的α调制模式:")
    report_lines.append("- 认(k=0): α基于φ⁰=1.0")
    report_lines.append("- 遇(k=1): α基于φ⁻¹≈0.618")
    report_lines.append("- 落(k=2): α基于φ⁻²≈0.382")
    report_lines.append("- 裂(k=3): α基于φ¹≈1.618")
    report_lines.append("- 余(k=4): α基于φ⁻³≈0.236")
    report_lines.append("")
    report_lines.append(f"每层i的motion assignment: k = i % 5, 共{num_layers}层")
    report_lines.append("")

    # ---- Phase 2结果 ----
    report_lines.append("## Phase 2: Activation Difference概览")
    report_lines.append("")
    report_lines.append("每对prompt (original vs minimal pair) 在每层产生不同的activation, "
                       "差异越大表示该层对该motion越敏感。")
    report_lines.append("")
    report_lines.append("| Motion | Prompt# | 标准模型Top3层 | φ-模型Top3层 |")
    report_lines.append("|--------|---------|---------------|-------------|")
    for mk in MOTION_KEYS:
        for pi in range(len(PROMPTS[mk])):
            sd = std_diffs[(mk, pi)]
            pd = phi_diffs[(mk, pi)]
            top3s = sorted(range(num_layers), key=lambda i: sd[i], reverse=True)[:3]
            top3p = sorted(range(num_layers), key=lambda i: pd[i], reverse=True)[:3]
            report_lines.append(
                f"| {MOTION_LABELS[MOTION_KEYS.index(mk)]} | {pi+1} | "
                f"L{top3s} ({sd[top3s[0]]:.1f}) | "
                f"L{top3p} ({pd[top3p[0]]:.1f}) |"
            )
    report_lines.append("")

    # ---- Phase 3: 重叠矩阵 ----
    report_lines.append("## Phase 3: Parcel重叠度矩阵")
    report_lines.append("")

    report_lines.append("### 标准模型 Jaccard重叠矩阵")
    report_lines.append("")
    header = "| | " + " | ".join(MOTION_LABELS) + " |"
    sep = "|" + "|".join(["---"] * 6) + "|"
    report_lines.append(header)
    report_lines.append(sep)
    for mi, mk1 in enumerate(MOTION_KEYS):
        row = f"| **{MOTION_LABELS[mi]}** |"
        for mj, mk2 in enumerate(MOTION_KEYS):
            row += f" {std_overlap_matrix[mi][mj]:.3f} |"
        report_lines.append(row)
    report_lines.append("")

    report_lines.append("### φ-Residual模型 Jaccard重叠矩阵")
    report_lines.append("")
    report_lines.append(header)
    report_lines.append(sep)
    for mi, mk1 in enumerate(MOTION_KEYS):
        row = f"| **{MOTION_LABELS[mi]}** |"
        for mj, mk2 in enumerate(MOTION_KEYS):
            row += f" {phi_overlap_matrix[mi][mj]:.3f} |"
        report_lines.append(row)
    report_lines.append("")

    report_lines.append(f"**标准模型**: 对角线均值={std_diag_avg:.3f}, "
                       f"非对角线均值={std_offdiag_avg:.3f}, 对比度={std_contrast:.2f}x")
    report_lines.append(f"**φ-模型**: 对角线均值={phi_diag_avg:.3f}, "
                       f"非对角线均值={phi_offdiag_avg:.3f}, 对比度={phi_contrast:.2f}x")
    report_lines.append(f"**Parcel分离度变化(φ-标准)**: {parcel_separation:+.3f}")
    report_lines.append("")

    # 层偏好
    report_lines.append("### 每层Motion偏好")
    report_lines.append("")
    report_lines.append("| 层 | α赋值(k) | α值 | 标准偏好 | φ偏好 | 一致? |")
    report_lines.append("|----|---------|-----|---------|-------|------|")
    for li in range(num_layers):
        k = motion_assign[li]
        assign_label = MOTION_LABELS[k]
        alpha_val = layer_alphas[li]
        std_pref = MOTION_LABELS[MOTION_KEYS.index(layer_preference_std[li])]
        phi_pref = MOTION_LABELS[MOTION_KEYS.index(layer_preference_phi[li])]
        match = "✓" if layer_preference_std[li] == layer_preference_phi[li] else ""
        report_lines.append(f"| L{li} | k={k}({assign_label}) | {alpha_val:.4f} | "
                           f"{std_pref} | {phi_pref} | {match} |")
    report_lines.append("")

    report_lines.append(f"**与α赋值(i%5)一致性**: 标准模型 {std_assign_match}/{num_layers} "
                       f"({100*std_assign_match/num_layers:.0f}%), "
                       f"φ-模型 {phi_assign_match}/{num_layers} "
                       f"({100*phi_assign_match/num_layers:.0f}%)")
    report_lines.append("")

    # ---- Phase 4: Ablation ----
    report_lines.append("## Phase 4: 因果验证 (Double Dissociation)")
    report_lines.append("")

    report_lines.append("### 关键层识别")
    report_lines.append("")
    report_lines.append("基于φ-Residual模型中每个motion的平均activation difference, "
                       f"选取top-{ABLATION_TOP_N}层作为该motion的关键层。")
    report_lines.append("")
    for mk in MOTION_KEYS:
        report_lines.append(f"- **{MOTION_NAMES[MOTION_KEYS.index(mk)]}**: "
                           f"层 {motion_key_layers[mk]}")
    report_lines.append("")

    report_lines.append("### Full-φ效应强度")
    report_lines.append("")
    report_lines.append("Cosine distance between baseline and full-φ last hidden states "
                       "(越大=φ-Residual改变越大):")
    report_lines.append("")
    for mk in MOTION_KEYS:
        report_lines.append(f"- {MOTION_LABELS[MOTION_KEYS.index(mk)]}: {phi_effect[mk]:.4f}")
    report_lines.append("")

    report_lines.append("### Ablation效果矩阵")
    report_lines.append("")
    report_lines.append("行=ablated motion (关掉其关键层的α), "
                       "列=test motion, "
                       "值=φ-effect减少量 (正值=ablation使输出更接近baseline)")
    report_lines.append("")
    header2 = "| ablate \\ test | " + " | ".join(MOTION_LABELS) + " |"
    report_lines.append(header2)
    report_lines.append(sep)
    for abl_mk in MOTION_KEYS:
        row = f"| **{MOTION_LABELS[MOTION_KEYS.index(abl_mk)]}** |"
        for test_mk in MOTION_KEYS:
            ablation_eff = phi_effect[test_mk] - ablation_result[abl_mk][test_mk]
            row += f" {ablation_eff:+.4f} |"
        report_lines.append(row)
    report_lines.append("")

    report_lines.append(f"**对角线平均效应**: {diag_avg:+.4f}")
    report_lines.append(f"**非对角线平均效应**: {offdiag_avg:+.4f}")
    report_lines.append(f"**对角线-非对角线差**: {diag_avg - offdiag_avg:+.4f}")
    report_lines.append(f"**选择性ablation有效计数**: {dissociation_count}/5")
    report_lines.append("")

    # 定性对比
    report_lines.append("### 定性文本对比 (baseline vs full-φ)")
    report_lines.append("")
    for mk in MOTION_KEYS:
        diff = "✓ 不同" if gen_texts[mk]["baseline"] != gen_texts[mk]["phi"] else "✗ 相同"
        report_lines.append(f"**{MOTION_NAMES[MOTION_KEYS.index(mk)]}** ({diff}):")
        report_lines.append(f"- 基线: `{gen_texts[mk]['baseline'][:120]}`")
        report_lines.append(f"- φ版: `{gen_texts[mk]['phi'][:120]}`")
        report_lines.append("")

    # ---- 结论 ----
    report_lines.append("## 结论")
    report_lines.append("")

    # Parcel分离
    if parcel_separation > 0.05:
        report_lines.append(f"1. **Parcel分离**: φ-Residual显著增强了五动的parcel分离度 "
                           f"(Δ={parcel_separation:+.3f})，5个motion对应5个相对独立的功能层组。")
    elif parcel_separation > 0:
        report_lines.append(f"1. **Parcel分离**: φ-Residual轻微增强了parcel分离度 "
                           f"(Δ={parcel_separation:+.3f})，效果需更大样本验证。")
    else:
        report_lines.append(f"1. **Parcel分离**: φ-Residual未增强parcel分离度 "
                           f"(Δ={parcel_separation:+.3f})，五动可能不构成独立parcel。")

    # Double dissociation
    if dissociation_count >= 4:
        report_lines.append(f"2. **Double Dissociation**: 强成立 ({dissociation_count}/5 motions选择性ablation有效)，"
                           f"关掉某motion的关键层主要影响该motion，不影响其他motion。")
    elif dissociation_count >= 2:
        report_lines.append(f"2. **Double Dissociation**: 部分成立 ({dissociation_count}/5 motions)，"
                           f"存在一定选择性但不够强。")
    else:
        report_lines.append(f"2. **Double Dissociation**: 未成立 ({dissociation_count}/5)，"
                           f"关键层可能不具motion特异性。")

    # α赋值一致性
    if phi_assign_match > std_assign_match + 5:
        report_lines.append(f"3. **α赋值一致性**: φ-模型与i%5赋值的一致性({phi_assign_match}/{num_layers})"
                           f"显著高于标准模型({std_assign_match}/{num_layers})，"
                           f"φ-Residual确实将层分成了与五动对应的5组。")
    else:
        report_lines.append(f"3. **α赋值一致性**: φ-模型({phi_assign_match}/{num_layers}) vs "
                           f"标准模型({std_assign_match}/{num_layers})，"
                           f"φ-Residual的分组效果需要进一步验证。")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*报告由 five_motion_attribution.py 自动生成 | "
                       f"耗时 {elapsed:.0f}s*")

    # 写入报告
    report_content = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"  报告已保存: {report_path}")

    # ---- 保存原始数据为JSON ----
    data_path = os.path.join(output_dir, "five_motion_attribution_data.json")
    data = {
        "config": {
            "model_path": MODEL_PATH,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "hidden_size": hidden_size,
            "strength": STRENGTH,
            "top_k_pct": TOP_K_PCT,
            "ablation_top_n": ABLATION_TOP_N,
        },
        "layer_alphas": layer_alphas,
        "motion_assignment": motion_assign,
        "std_overlap_matrix": std_overlap_matrix,
        "phi_overlap_matrix": phi_overlap_matrix,
        "std_diag_avg": std_diag_avg,
        "std_offdiag_avg": std_offdiag_avg,
        "phi_diag_avg": phi_diag_avg,
        "phi_offdiag_avg": phi_offdiag_avg,
        "parcel_separation": parcel_separation,
        "std_assign_match": std_assign_match,
        "phi_assign_match": phi_assign_match,
        "motion_key_layers": {mk: motion_key_layers[mk] for mk in MOTION_KEYS},
        "phi_effect": phi_effect,
        "dissociation_count": dissociation_count,
        "diag_avg": diag_avg,
        "offdiag_avg": offdiag_avg,
        "gen_texts": gen_texts,
        "elapsed_seconds": elapsed,
    }
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  数据已保存: {data_path}")

    print(f"\n{'='*70}")
    print(f"实验完成! 总耗时: {elapsed:.0f}秒 ({elapsed/60:.1f}分钟)")
    print(f"报告: {report_path}")
    print(f"数据: {data_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
