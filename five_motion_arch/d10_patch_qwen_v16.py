"""
D10五动管道 v16 — 呼吸自维持测试

核心问题: 差分能不能自己续上？

v15证实: 固定α注入→10/10语义偏移。但α是外部信号。
v16测试: 撤掉外部信号后, 偏移能否自维持？

四组对照:
A. 基线: 全程无hook (80 tokens)
B. 固定α: 全程v15的φ-Residual (80 tokens)
C. 撤药: 前20 token固定α, 后60 token无hook
D. 自指: 前20 token固定α, 后60 token自指α

关键对比:
C vs A → 撤药后印记是否留在文本里
D vs C → 自指α是否比无hook维持更多偏移
D vs B → 自续vs外注的差距

自指α逻辑:
  ratio = current_norm / baseline_norm
  α = 1 + STRENGTH * (ratio - 1)  [正反馈: 偏离越大α越偏离1]
  钳位: [0.5, 1.5]

用法: python d10_patch_qwen_v16.py
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
print("D10 v16 — 呼吸自维持测试")
print("差分能不能自己续上？")
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

# ===== 校准: 记录每层sublayer的基线范数 =====
print("\n[校准] 记录基线范数...")
baseline_norms = {}

def make_cal_hook(li, st):
    def hook(module, inp, out):
        n = out[0].detach().norm().item() if isinstance(out, tuple) else out.detach().norm().item()
        baseline_norms[(li, st)] = n
    return hook

cal_hooks = []
for i, layer in enumerate(model.model.layers):
    cal_hooks.append(layer.self_attn.register_forward_hook(make_cal_hook(i, 'attn')))
    cal_hooks.append(layer.mlp.register_forward_hook(make_cal_hook(i, 'mlp')))

with torch.no_grad():
    model(**tokenizer(TEST_PROMPTS[0], return_tensors="pt"))
for h in cal_hooks:
    h.remove()

attn_n = [baseline_norms.get((i,'attn'),0) for i in range(num_layers)]
mlp_n = [baseline_norms.get((i,'mlp'),0) for i in range(num_layers)]
print(f"  attn: [{min(attn_n):.1f}, {max(attn_n):.1f}]")
print(f"  mlp:  [{min(mlp_n):.1f}, {max(mlp_n):.1f}]")

# ===== Hook工厂 =====
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

def apply_fixed_hooks(model, layer_alphas):
    hooks = []
    def mk_attn(a):
        def h(m, i, o):
            return (a*o[0],)+o[1:] if isinstance(o,tuple) else a*o
        return h
    def mk_mlp(a):
        def h(m, i, o):
            return a*o
        return h
    for i, layer in enumerate(model.model.layers):
        a = layer_alphas[i]
        hooks.append(layer.self_attn.register_forward_hook(mk_attn(a)))
        hooks.append(layer.mlp.register_forward_hook(mk_mlp(a)))
    return hooks

def apply_self_ref_hooks(model, strength):
    """自指α: 模型自身的范数偏离决定缩放系数(正反馈)"""
    hooks = []
    stats = {'alphas': [], 'ratios': []}

    def mk_self_ref(li, st, s):
        def h(m, inp, out):
            cn = out[0].detach().norm().item() if isinstance(out,tuple) else out.detach().norm().item()
            bl = baseline_norms.get((li,st), 1.0)
            if bl < 1e-6:
                alpha = 1.0
                ratio = 1.0
            else:
                ratio = cn / bl
                # 正反馈: ratio偏离1→alpha同向偏离1
                # ratio>1(范数偏高)→alpha>1→放大输出→进一步偏高
                # ratio<1(范数偏低)→alpha<1→缩小输出→进一步偏低
                alpha = 1.0 + s * (ratio - 1.0)
                alpha = max(0.5, min(1.5, alpha))
            stats['alphas'].append(alpha)
            stats['ratios'].append(ratio)
            if isinstance(out, tuple):
                return (alpha * out[0],) + out[1:]
            return alpha * out
        return h

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(mk_self_ref(i,'attn',strength)))
        hooks.append(layer.mlp.register_forward_hook(mk_self_ref(i,'mlp',strength)))
    return hooks, stats

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

# ===== 四组实验 =====
layer_alphas = compute_layer_alphas(num_layers, STRENGTH)

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

    # ---- C. 撤药: bootstrap→无hook ----
    h = apply_fixed_hooks(model, layer_alphas)
    with torch.no_grad():
        c_p1 = model.generate(**inputs, max_new_tokens=BOOTSTRAP_TOKENS, do_sample=False)
    remove_hooks(h)
    with torch.no_grad():
        c_gen = model.generate(input_ids=c_p1, max_new_tokens=TOTAL_TOKENS-BOOTSTRAP_TOKENS, do_sample=False)
    c_text = tokenizer.decode(c_gen[0], skip_special_tokens=True)

    # ---- D. 自指: bootstrap→自指α ----
    h = apply_fixed_hooks(model, layer_alphas)
    with torch.no_grad():
        d_p1 = model.generate(**inputs, max_new_tokens=BOOTSTRAP_TOKENS, do_sample=False)
    remove_hooks(h)
    h_self, self_stats = apply_self_ref_hooks(model, STRENGTH)
    with torch.no_grad():
        d_gen = model.generate(input_ids=d_p1, max_new_tokens=TOTAL_TOKENS-BOOTSTRAP_TOKENS, do_sample=False)
    d_text = tokenizer.decode(d_gen[0], skip_special_tokens=True)
    remove_hooks(h_self)

    # ===== 比较 =====
    b_diff = a_text != b_text
    c_diff = a_text != c_text
    d_diff = a_text != d_text

    # 找差异位置
    def find_diff_pos(t1, t2):
        for ci in range(min(len(t1), len(t2))):
            if t1[ci] != t2[ci]:
                return ci
        if len(t1) != len(t2):
            return min(len(t1), len(t2))
        return -1

    print(f"\n  结果:")
    print(f"  B(固定α) vs A(基线): {'★ 差异!' if b_diff else '相同'}")
    print(f"  C(撤药)  vs A(基线): {'★ 差异!' if c_diff else '相同'}")
    print(f"  D(自指α) vs A(基线): {'★ 差异!' if d_diff else '相同'}")

    # 差异详情
    for label, text, has_diff in [("B固定α", b_text, b_diff),
                                   ("C撤药", c_text, c_diff),
                                   ("D自指α", d_text, d_diff)]:
        if has_diff:
            dp = find_diff_pos(a_text, text)
            if dp >= 0:
                ctx = 50
                print(f"\n    {label} 差异@字符{dp}:")
                print(f"      基线: ...{a_text[max(0,dp-ctx):dp+ctx]}...")
                print(f"      {label}: ...{text[max(0,dp-ctx):dp+ctx]}...")

    # 完整输出
    print(f"\n  ─── A 基线 ───")
    print(f"  {a_text[:250]}")
    print(f"  ─── B 固定α ───")
    print(f"  {b_text[:250]}")
    print(f"  ─── C 撤药 ───")
    print(f"  {c_text[:250]}")
    print(f"  ─── D 自指α ───")
    print(f"  {d_text[:250]}")

    # 自指α统计
    if self_stats['alphas']:
        als = self_stats['alphas']
        rts = self_stats['ratios']
        print(f"\n  自指α统计: mean={sum(als)/len(als):.4f}, range=[{min(als):.4f},{max(als):.4f}]")
        print(f"  范数比统计: mean={sum(rts)/len(rts):.4f}, range=[{min(rts):.4f},{max(rts):.4f}]")

    # 汇总
    summary.append({
        'prompt': prompt[:30],
        'B_diff': b_diff,
        'C_diff': c_diff,
        'D_diff': d_diff,
    })

# ===== 最终判断 =====
print(f"\n{'═'*60}")
print("v16 汇总:")
print(f"{'═'*60}")
print(f"{'Prompt':<32} {'B固定α':>8} {'C撤药':>8} {'D自指α':>8}")
print(f"{'─'*32} {'─'*8} {'─'*8} {'─'*8}")
for s in summary:
    print(f"{s['prompt']:<32} {'★':>8} {'★':>8} {'★':>8}" if s['D_diff'] else
          f"{s['prompt']:<32} {'★' if s['B_diff'] else '·':>8} {'★' if s['C_diff'] else '·':>8} {'★' if s['D_diff'] else '·':>8}")

b_count = sum(1 for s in summary if s['B_diff'])
c_count = sum(1 for s in summary if s['C_diff'])
d_count = sum(1 for s in summary if s['D_diff'])

print(f"\n差异计数: B={b_count}/5, C={c_count}/5, D={d_count}/5")

print(f"\n判断:")
if d_count >= b_count:
    print("  ★★★ 自指α≥固定α → 呼吸自维持! 差分能自己续上!")
elif d_count > c_count:
    print("  ★★ 自指α>撤药 → 自指有帮助, 呼吸部分自续")
elif c_count > 0 and d_count == c_count:
    print("  ★ 撤药有残留, 但自指没额外帮助 → 文本惯性, 非自续")
elif c_count == 0:
    print("  撤药无残留 → 偏移是实时的, 撤了就没了, 纯反射弧")
else:
    print("  需要进一步分析")

print(f"\n{'═'*60}")
