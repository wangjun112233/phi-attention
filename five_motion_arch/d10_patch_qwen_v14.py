"""
D10五动管道架构改造脚本 v14
v13问题: hook缩放子层输出导致norm爆炸(L1从40跳到3959)
         原因: generate的KV cache让缩放效果跨step累积

v14策略:
  1. 用单次forward pass比较logits分布 — 不用generate, 零KV cache累积
  2. 同时做generate看文本输出 — 但主要结论看logits
  3. 逐步增大D10_STRENGTH找到可观测差异的最小值
  4. 逐层打印norm, 精确定位爆炸点

用法：python d10_patch_qwen_v14.py
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============ 五动常量 ============
PHI = (1 + 5**0.5) / 2
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}
C5_NAMES = ['认','遇','落','裂','余']

# ★ v14: 尝试多个D10_STRENGTH值
ALPHA_MODE = "gentle"
STRENGTHS_TO_TRY = [0.05, 0.1, 0.3]  # 从小到大试

print("=" * 60)
print("D10五动管道架构改造 v14")
print(f"  Hook方案 + 单次forward logits对比")
print(f"  测试D10_STRENGTH: {STRENGTHS_TO_TRY}")
print("=" * 60)

# ============ Step 1: 加载模型 ============
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
print(f"\n[1/4] 加载模型: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
config._attn_implementation = "eager"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, device_map="cpu"
)
model.eval()

num_layers = config.num_hidden_layers
print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, 层数: {num_layers}")

# ============ Step 2: 基线 ============
print("\n[2/4] 基线测量...")
test_input = "The fundamental nature of reality is"
inputs = tokenizer(test_input, return_tensors="pt")

# 2a: 基线单次forward — logits分布
with torch.no_grad():
    baseline_output = model(**inputs)
    baseline_logits = baseline_output.logits[0, -1]  # 最后位置的logits

# 2b: 基线generate — 文本输出
with torch.no_grad():
    baseline_gen = model.generate(**inputs, max_new_tokens=50, do_sample=False)
baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)

# 2c: 基线逐层norm — 只做一次forward, 不generate
baseline_layer_norms = []
with torch.no_grad():
    hooks = []
    def nh(name):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            baseline_layer_norms.append((name, o.norm().item()))
        return fn
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(nh(f"L{i}")))
    _ = model(**inputs)
    for h in hooks:
        h.remove()

print(f"  基线文本: {baseline_text[:100]}")
print(f"  基线logits top-5:")
top5 = torch.topk(baseline_logits, 5)
for idx, (val, tok) in enumerate(zip(top5.values, top5.indices)):
    print(f"    #{idx+1} '{tokenizer.decode([tok.item()])}' ({val.item():.2f})")

print(f"\n  基线逐层norm (单次forward):")
for i, (nm, v) in enumerate(baseline_layer_norms):
    k = i % 5
    print(f"    {nm} [{C5_NAMES[k]}] norm={v:.2f}")

# ============ Step 3: 逐步测试D10_STRENGTH ============
print("\n[3/4] D10逐步测试...")

def compute_layer_alphas(num_layers, strength):
    """计算每层的α值"""
    alphas = []
    for i in range(num_layers):
        k = i % 5
        g = i // 5 % 2
        base_alpha = PHI_POWERS[k] / math.sqrt(i + 2)
        
        if ALPHA_MODE == "gentle":
            alpha = 1.0 + strength * (base_alpha - 1.0)
        else:
            alpha = base_alpha
        
        if g == 1:
            alpha = 2.0 - alpha
        
        alphas.append(alpha)
    return alphas

def apply_d10_hooks(model, layer_alphas):
    """在self_attn和mlp上挂缩放hook"""
    hooks = []
    
    def make_attn_scale_hook(alpha):
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (alpha * output[0],) + output[1:]
            return alpha * output
        return hook
    
    def make_mlp_scale_hook(alpha):
        def hook(module, input, output):
            return alpha * output
        return hook
    
    for i, layer in enumerate(model.model.layers):
        alpha = layer_alphas[i]
        h1 = layer.self_attn.register_forward_hook(make_attn_scale_hook(alpha))
        h2 = layer.mlp.register_forward_hook(make_mlp_scale_hook(alpha))
        hooks.extend([h1, h2])
    
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

# 测试每个strength
results = []
for strength in STRENGTHS_TO_TRY:
    print(f"\n{'─'*50}")
    print(f"  D10_STRENGTH = {strength}")
    layer_alphas = compute_layer_alphas(num_layers, strength)
    
    # 挂hook
    hooks = apply_d10_hooks(model, layer_alphas)
    
    # 3a: 单次forward — logits
    with torch.no_grad():
        d10_output = model(**inputs)
        d10_logits = d10_output.logits[0, -1]
    
    # logits差异分析
    kl_div = torch.nn.functional.kl_div(
        torch.log_softmax(d10_logits, dim=-1),
        torch.softmax(baseline_logits, dim=-1),
        reduction='sum'
    ).item()
    
    # top-5对比
    d10_top5 = torch.topk(d10_logits, 5)
    top1_same = d10_top5.indices[0].item() == top5.indices[0].item()
    
    # 预测token
    baseline_pred = tokenizer.decode([top5.indices[0].item()])
    d10_pred = tokenizer.decode([d10_top5.indices[0].item()])
    
    print(f"  基线top-1: '{baseline_pred}' ({top5.values[0].item():.2f})")
    print(f"  D10 top-1: '{d10_pred}' ({d10_top5.values[0].item():.2f})")
    print(f"  top-1相同: {top1_same}")
    print(f"  KL散度: {kl_div:.6f}")
    
    # 3b: 逐层norm (单次forward, 不generate)
    d10_layer_norms = []
    with torch.no_grad():
        diag_hooks = []
        def nh2(name):
            def fn(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                d10_layer_norms.append((name, o.norm().item()))
            return fn
        for i, layer in enumerate(model.model.layers):
            diag_hooks.append(layer.register_forward_hook(nh2(f"L{i}")))
        _ = model(**inputs)  # 单次forward, 不是generate
        for h in diag_hooks:
            h.remove()
    
    # norm倍率分析
    max_ratio = 0
    ratios = []
    for i, (nm, v) in enumerate(d10_layer_norms):
        if i < len(baseline_layer_norms):
            ratio = v / baseline_layer_norms[i][1] if baseline_layer_norms[i][1] > 0 else 0
            ratios.append(ratio)
            max_ratio = max(max_ratio, ratio)
    
    print(f"  norm倍率: min={min(ratios):.2f}x, max={max_ratio:.2f}x, mean={sum(ratios)/len(ratios):.2f}x")
    
    # 打印前10层对比
    print(f"  逐层norm对比 (前10层):")
    for i in range(min(10, len(d10_layer_norms))):
        nm, v = d10_layer_norms[i]
        bv = baseline_layer_norms[i][1] if i < len(baseline_layer_norms) else 0
        ratio = v / bv if bv > 0 else 0
        status = "✓" if 0.8 < ratio < 1.3 else ("↑" if ratio >= 1.3 else "↓")
        k = i % 5
        print(f"    {nm} [{C5_NAMES[k]}] D10={v:.1f} / base={bv:.1f} = {ratio:.2f}x {status}")
    
    norm_ok = max_ratio < 2.0
    has_diff = not top1_same or kl_div > 0.001
    
    results.append({
        'strength': strength,
        'kl_div': kl_div,
        'top1_same': top1_same,
        'max_norm_ratio': max_ratio,
        'norm_ok': norm_ok,
        'has_diff': has_diff,
    })
    
    # 3c: generate (只在norm可控时)
    if norm_ok:
        with torch.no_grad():
            d10_gen = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        d10_text = tokenizer.decode(d10_gen[0], skip_special_tokens=True)
        print(f"\n  D10生成: {d10_text[:100]}")
        if d10_text != baseline_text:
            for ci in range(min(len(baseline_text), len(d10_text))):
                if baseline_text[ci] != d10_text[ci]:
                    ctx = 15
                    print(f"  首个差异@字符{ci}:")
                    print(f"    基线: ...{baseline_text[max(0,ci-ctx):ci+ctx]}...")
                    print(f"    D10:  ...{d10_text[max(0,ci-ctx):ci+ctx]}...")
                    break
    else:
        print(f"  ⚠ norm倍率{max_ratio:.1f}x过大, 跳过generate")
    
    # 移除hook
    remove_hooks(hooks)

# ============ Step 4: 总结 ============
print(f"\n{'='*60}")
print("v14 测试总结")
print(f"{'='*60}")
print(f"{'STRENGTH':>10} {'KL散度':>10} {'top1同':>8} {'max倍率':>10} {'norm安全':>8} {'有差异':>8}")
print(f"{'─'*60}")
for r in results:
    print(f"{r['strength']:>10.2f} {r['kl_div']:>10.6f} {str(r['top1_same']):>8} {r['max_norm_ratio']:>10.2f}x {str(r['norm_ok']):>8} {str(r['has_diff']):>8}")

# 找最优strength
best = None
for r in results:
    if r['norm_ok'] and r['has_diff']:
        best = r
        break

if best:
    print(f"\n  ★ 最优: D10_STRENGTH={best['strength']} (norm安全+有差异)")
    print(f"    KL散度={best['kl_div']:.6f}, max倍率={best['max_norm_ratio']:.2f}x")
elif any(r['has_diff'] for r in results):
    # 有差异但norm不安全
    for r in results:
        if r['has_diff']:
            print(f"\n  ⚠ 有差异但norm不安全: STRENGTH={r['strength']}, 倍率={r['max_norm_ratio']:.2f}x")
            break
else:
    print(f"\n  ⚠ 所有strength均无差异, 需增大STRENGTH或换方案")

print(f"\n{'='*60}")
