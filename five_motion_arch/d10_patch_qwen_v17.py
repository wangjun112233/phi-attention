"""
D10五动管道 v17 — 修正自指α：用bootstrap期范数作参考系

v16结论: 自指α均值≈0.97(接近1=无效), 因为校准基线(generate时范数)
         跟单次forward校准值不匹配。C=2/5(文本惯性), D=2/5(=C)。

v17修正: 不用单次forward校准, 改用bootstrap阶段(generate前20 token)
         在固定α下记录的实时范数作为参考系。

五组对照:
A. 基线: 全程无hook (80 tokens)
B. 固定α: 全程v15的φ-Residual (80 tokens)
C. 撤药: 前20 token固定α, 后60 token无hook
D1. 自平衡(负反馈): 前20固定α+记录范数, 后60 token用bootstrap范数
    作参考做负反馈(偏高→缩小, 偏低→放大)→维持bootstrap的呼吸模式
D2. 正反馈: 前20固定α+记录范数, 后60 token正反馈
    (偏高→放大, 偏低→缩小)→放大偏差

核心判据:
D1 > C → 自平衡能维持偏移, 呼吸可自续
D1 ≈ B → 完美自续!
D1 ≈ C → 无法自续, 纯反射弧确认

用法: python d10_patch_qwen_v17.py
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import defaultdict

PHI = (1 + 5**0.5) / 2
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}

TEST_PROMPTS = [
    "The fundamental nature of reality is",
    "Consciousness arises from",
    "The relationship between order and chaos is",
    "In physics, the most fundamental principle is",
    "The meaning of existence is",
]

STRENGTH = 0.05
BOOTSTRAP_TOKENS = 20
TOTAL_TOKENS = 80

print("=" * 60)
print("D10 v17 — 修正自指α: bootstrap范数参考系")
print("=" * 60)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
print(f"\n加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
config._attn_implementation = "eager"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, device_map="cpu"
)
model.eval()
num_layers = config.num_hidden_layers

# ===== α计算 =====
def compute_layer_alphas(nl, strength):
    alphas = []
    for i in range(nl):
        k = i % 5
        g = i // 5 % 2
        ba = PHI_POWERS[k] / math.sqrt(i+2)
        a = 1.0 + strength * (ba - 1.0)
        if g == 1:
            a = 2.0 - a
        alphas.append(a)
    return alphas

layer_alphas = compute_layer_alphas(num_layers, STRENGTH)

# ===== Hook工厂 =====
def make_attn_scale(alpha):
    def hook(m, inp, out):
        return (alpha * out[0],) + out[1:] if isinstance(out, tuple) else alpha * out
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

# ===== 录制+固定α组合hook =====
def apply_fixed_plus_recording(model, alphas):
    """固定α hook + 同时记录每层范数"""
    hooks = []
    norms_record = defaultdict(list)  # {(li, 'attn'/'mlp'): [norm1, norm2, ...]}

    def mk_attn_record_scale(li, alpha):
        def hook(m, inp, out):
            n = out[0].detach().norm().item() if isinstance(out, tuple) else out.detach().norm().item()
            norms_record[(li, 'attn')].append(n)
            return (alpha * out[0],) + out[1:] if isinstance(out, tuple) else alpha * out
        return hook

    def mk_mlp_record_scale(li, alpha):
        def hook(m, inp, out):
            n = out.detach().norm().item()
            norms_record[(li, 'mlp')].append(n)
            return alpha * out
        return hook

    for i, layer in enumerate(model.model.layers):
        a = alphas[i]
        hooks.append(layer.self_attn.register_forward_hook(mk_attn_record_scale(i, a)))
        hooks.append(layer.mlp.register_forward_hook(mk_mlp_record_scale(i, a)))

    return hooks, norms_record

# ===== 自指α hook (用bootstrap范数作参考) =====
def apply_self_ref_hooks(model, bootstrap_avg, strength, mode='negative'):
    """
    mode='negative'(自平衡): ratio>1→α<1(缩回), ratio<1→α>1(扩回)
      → 维持bootstrap期建立的范数模式
    mode='positive'(正反馈): ratio>1→α>1(放大), ratio<1→α<1(缩小)
      → 放大偏离
    """
    hooks = []
    stats = {'alphas': [], 'ratios': []}

    def mk_self_ref(li, st, s, m):
        def hook(mod, inp, out):
            cn = out[0].detach().norm().item() if isinstance(out, tuple) else out.detach().norm().item()
            bl = bootstrap_avg.get((li, st), 1.0)
            if bl < 1e-6:
                alpha = 1.0
                ratio = 1.0
            else:
                ratio = cn / bl
                if m == 'negative':
                    # 自平衡: 偏高→缩回, 偏低→扩回
                    alpha = 1.0 - s * (ratio - 1.0)
                else:
                    # 正反馈: 偏高→放大, 偏低→缩小
                    alpha = 1.0 + s * (ratio - 1.0)
                alpha = max(0.5, min(1.5, alpha))
            stats['alphas'].append(alpha)
            stats['ratios'].append(ratio)
            if isinstance(out, tuple):
                return (alpha * out[0],) + out[1:]
            return alpha * out
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(mk_self_ref(i, 'attn', strength, mode)))
        hooks.append(layer.mlp.register_forward_hook(mk_self_ref(i, 'mlp', strength, mode)))

    return hooks, stats

# ===== 比较工具 =====
def find_diff_pos(t1, t2):
    for ci in range(min(len(t1), len(t2))):
        if t1[ci] != t2[ci]:
            return ci
    if len(t1) != len(t2):
        return min(len(t1), len(t2))
    return -1

# ===== 主实验 =====
summary = []

for pi, prompt in enumerate(TEST_PROMPTS):
    print(f"\n{'═'*60}")
    print(f"[{pi+1}/5] Prompt: '{prompt}'")
    print(f"{'═'*60}")
    inputs = tokenizer(prompt, return_tensors="pt")

    # ---- A. 基线 ----
    with torch.no_grad():
        a_gen = model.generate(**inputs, max_new_tokens=TOTAL_TOKENS, do_sample=False)
    a_text = tokenizer.decode(a_gen[0], skip_special_tokens=True)

    # ---- B. 固定α(全程) ----
    h = apply_fixed_hooks(model, layer_alphas)
    with torch.no_grad():
        b_gen = model.generate(**inputs, max_new_tokens=TOTAL_TOKENS, do_sample=False)
    b_text = tokenizer.decode(b_gen[0], skip_special_tokens=True)
    remove_hooks(h)

    # ---- Bootstrap: 固定α+录制范数, 生成前20 token ----
    h_rec, norms_rec = apply_fixed_plus_recording(model, layer_alphas)
    with torch.no_grad():
        bootstrap_gen = model.generate(**inputs, max_new_tokens=BOOTSTRAP_TOKENS, do_sample=False)
    remove_hooks(h_rec)

    # 计算bootstrap期每层平均范数
    bootstrap_avg = {}
    for key, norms in norms_rec.items():
        bootstrap_avg[key] = sum(norms) / len(norms) if norms else 1.0

    print(f"  Bootstrap范数(采样): ", end="")
    for key in [(0,'attn'), (14,'attn'), (27,'attn')]:
        if key in bootstrap_avg:
            print(f"L{key[0]}{key[1][0]}={bootstrap_avg[key]:.1f} ", end="")
    print()

    # ---- C. 撤药: bootstrap→无hook ----
    with torch.no_grad():
        c_gen = model.generate(input_ids=bootstrap_gen, max_new_tokens=TOTAL_TOKENS-BOOTSTRAP_TOKENS, do_sample=False)
    c_text = tokenizer.decode(c_gen[0], skip_special_tokens=True)

    # ---- D1. 自平衡(负反馈): bootstrap→负反馈维持 ----
    h_d1, stats_d1 = apply_self_ref_hooks(model, bootstrap_avg, STRENGTH, mode='negative')
    with torch.no_grad():
        d1_gen = model.generate(input_ids=bootstrap_gen, max_new_tokens=TOTAL_TOKENS-BOOTSTRAP_TOKENS, do_sample=False)
    d1_text = tokenizer.decode(d1_gen[0], skip_special_tokens=True)
    remove_hooks(h_d1)

    # ---- D2. 正反馈: bootstrap→正反馈放大 ----
    h_d2, stats_d2 = apply_self_ref_hooks(model, bootstrap_avg, STRENGTH, mode='positive')
    with torch.no_grad():
        d2_gen = model.generate(input_ids=bootstrap_gen, max_new_tokens=TOTAL_TOKENS-BOOTSTRAP_TOKENS, do_sample=False)
    d2_text = tokenizer.decode(d2_gen[0], skip_special_tokens=True)
    remove_hooks(h_d2)

    # ===== 比较 =====
    b_diff = a_text != b_text
    c_diff = a_text != c_text
    d1_diff = a_text != d1_text
    d2_diff = a_text != d2_text

    print(f"\n  结果:")
    print(f"  B(固定α)  vs A: {'★' if b_diff else '·'}")
    print(f"  C(撤药)   vs A: {'★' if c_diff else '·'}")
    print(f"  D1(自平衡) vs A: {'★' if d1_diff else '·'}")
    print(f"  D2(正反馈) vs A: {'★' if d2_diff else '·'}")

    # 核心对比: D1 vs C
    if d1_diff and not c_diff:
        print(f"  ★★ D1>撤药 → 自平衡维持了偏移!")
    elif d1_diff and c_diff:
        # 看差异位置谁更远
        d1_pos = find_diff_pos(a_text, d1_text)
        c_pos = find_diff_pos(a_text, c_text)
        if d1_pos >= 0 and c_pos >= 0:
            # 再看从bootstrap后的文本差异
            # 比较D1和C是否不同
            if d1_text != c_text:
                print(f"  ★ D1≠C → 自平衡产生了与撤药不同的偏移")
            else:
                print(f"  D1=C → 自平衡没额外帮助(同文本惯性)")
        else:
            print(f"  D1=C → 自平衡没额外帮助")
    elif not d1_diff and c_diff:
        print(f"  ⚠ D1<C → 自平衡反而抹掉了惯性!")
    else:
        print(f"  D1=C=无差异")

    # 差异详情
    for label, text, has_diff in [("B固定α", b_text, b_diff),
                                   ("C撤药", c_text, c_diff),
                                   ("D1自平衡", d1_text, d1_diff),
                                   ("D2正反馈", d2_text, d2_diff)]:
        if has_diff:
            dp = find_diff_pos(a_text, text)
            if dp >= 0:
                ctx = 40
                print(f"\n    {label} 差异@{dp}:")
                print(f"      基线: ...{a_text[max(0,dp-ctx):dp+ctx]}...")
                print(f"      {label}: ...{text[max(0,dp-ctx):dp+ctx]}...")

    # 完整输出(精简)
    for label, text in [("A基线", a_text), ("B固定α", b_text),
                         ("C撤药", c_text), ("D1自平衡", d1_text), ("D2正反馈", d2_text)]:
        print(f"  [{label}] {text[:200]}")

    # 自指α统计
    for label, stats in [("D1自平衡", stats_d1), ("D2正反馈", stats_d2)]:
        if stats['alphas']:
            als = stats['alphas']
            rts = stats['ratios']
            print(f"  {label} α: mean={sum(als)/len(als):.4f} [{min(als):.4f},{max(als):.4f}] ratio: mean={sum(rts)/len(rts):.4f} [{min(rts):.4f},{max(rts):.4f}]")

    summary.append({
        'prompt': prompt[:30],
        'B': b_diff, 'C': c_diff, 'D1': d1_diff, 'D2': d2_diff,
        'd1_eq_c': d1_text == c_text,
    })

# ===== 最终判断 =====
print(f"\n{'═'*60}")
print("v17 汇总:")
print(f"{'═'*60}")
print(f"{'Prompt':<32} {'B固定':>6} {'C撤药':>6} {'D1自平衡':>8} {'D2正反馈':>8}")
print(f"{'─'*32} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")
for s in summary:
    print(f"{s['prompt']:<32} {'★' if s['B'] else '·':>6} {'★' if s['C'] else '·':>6} {'★' if s['D1'] else '·':>8} {'★' if s['D2'] else '·':>8}")

b_count = sum(1 for s in summary if s['B'])
c_count = sum(1 for s in summary if s['C'])
d1_count = sum(1 for s in summary if s['D1'])
d2_count = sum(1 for s in summary if s['D2'])
d1_eq_c = sum(1 for s in summary if s['d1_eq_c'])

print(f"\n差异计数: B={b_count}/5, C={c_count}/5, D1={d1_count}/5, D2={d2_count}/5")
print(f"D1=C(同文本): {d1_eq_c}/5")

print(f"\n判断:")
if d1_count > c_count:
    print("  ★★★ 自平衡>撤药 → 呼吸可自续! 差分能通过负反馈自己续上!")
elif d1_count == c_count and d1_count > 0:
    d1_new = sum(1 for s in summary if s['D1'] and not s['d1_eq_c'])
    if d1_new > 0:
        print("  ★★ D1=C计数, 但D1产生了不同的偏移 → 自平衡有部分效果")
    else:
        print("  ★ D1=C完全相同 → 文本惯性, 自平衡没额外帮助")
elif d1_count == 0 and c_count == 0:
    print("  撤药无残留 → v16结论确认: 纯反射弧")
elif d1_count < c_count:
    print("  ⚠ 自平衡<撤药 → 负反馈反而抑制了惯性偏移")
else:
    print("  需要进一步分析")

if d2_count > d1_count:
    print("  补充: 正反馈>自平衡 → 放大偏差比维持模式更有效")
elif d2_count > c_count:
    print("  补充: 正反馈>撤药 → 正反馈能部分续上偏移")

print(f"\n{'═'*60}")
