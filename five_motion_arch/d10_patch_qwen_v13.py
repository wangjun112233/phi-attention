"""
D10五动管道架构改造脚本 v13
v12 bug: 手拆DecoderLayer.forward时返回tuple格式与transformers 5.13.1内部不兼容
        导致下一层收到tuple当hidden_states, 报 'tuple' object has no attribute 'dtype'

v13: 用register_forward_hook缩放子层输出 — 不碰DecoderLayer.forward!
     原理: residual + sublayer_output 中, sublayer_output就是self_attn/mlp的返回值
     在这两个模块上挂hook, 把output[0]乘α, 等价于 residual + α*sublayer_output
     这就是φ-Residual, 但实现方式100%兼容原始forward

     优势:
       1. DecoderLayer.forward完全不动 — 零兼容性风险
       2. KV cache / RoPE / attention_mask / position_embeddings 全部由原始代码处理
       3. 只改self_attn和mlp的输出缩放 — 最小侵入

用法：python d10_patch_qwen_v13.py
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============ 五动常量 ============
PHI = (1 + 5**0.5) / 2

# φ-Residual: 每个五动相位的alpha
# 认(0): 1.0, 遇(1): 1/φ≈0.618, 落(2): 1/φ²≈0.382, 裂(3): φ≈1.618, 余(4): 1/φ³≈0.236
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}

C5_NAMES = ['认','遇','落','裂','余']
C5_TEMPS = [0.8, 1.0, 1.2, PHI, 1.0]

# ★ v13: α缩放模式
ALPHA_MODE = "gentle"    # "gentle" = 1 + strength*(base-1), "full" = 直接用base
D10_STRENGTH = 0.3       # gentle模式下的偏移强度

print("=" * 60)
print("D10五动管道架构改造 v13")
print(f"  Hook方案: forward_hook缩放子层输出(不拆DecoderLayer.forward)")
print(f"  ALPHA_MODE = {ALPHA_MODE}")
print(f"  D10_STRENGTH = {D10_STRENGTH}")
print("=" * 60)

# ============ Step 1: 加载模型 ============
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
print(f"\n[1/5] 加载模型: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
config._attn_implementation = "eager"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, device_map="cpu"
)
model.eval()

num_layers = config.num_hidden_layers
d_model = config.hidden_size

print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"  层数: {num_layers}")

# ============ Step 2: 基线输出 ============
print("\n[2/5] 生成基线输出...")
test_input = "The fundamental nature of reality is"
inputs = tokenizer(test_input, return_tensors="pt")
with torch.no_grad():
    baseline_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
print(f"  基线: {baseline_text[:120]}")

# ============ Step 3: D10改造 ============
print("\n[3/5] 执行D10架构改造...")

# ★ 计算每层的φ-Residual alpha
layer_alphas = []
for i in range(num_layers):
    k = i % 5
    g = i // 5 % 2  # Z₂翻转
    base_alpha = PHI_POWERS[k] / math.sqrt(i + 2)
    
    if ALPHA_MODE == "gentle":
        alpha = 1.0 + D10_STRENGTH * (base_alpha - 1.0)
    else:
        alpha = base_alpha
    
    # Z₂翻转: 奇数组residual方向反转
    if g == 1:
        alpha = 2.0 - alpha
    
    layer_alphas.append(alpha)

print(f"  φ-Residual alpha分布:")
for i in range(min(10, num_layers)):
    k = i % 5
    g = i // 5 % 2
    z2 = '⇄' if g == 1 else '→'
    print(f"    L{i} [{C5_NAMES[k]}] {z2} α={layer_alphas[i]:.4f}")
print(f"    ...")
for i in range(max(10, num_layers-3), num_layers):
    k = i % 5
    g = i // 5 % 2
    z2 = '⇄' if g == 1 else '→'
    print(f"    L{i} [{C5_NAMES[k]}] {z2} α={layer_alphas[i]:.4f}")

# ★★★ 核心: 用forward_hook缩放子层输出 ★★★
# 原理: DecoderLayer.forward 内部:
#   residual = hidden_states
#   hidden_states = self_attn(input_layernorm(hidden_states), ...)
#   hidden_states = residual + hidden_states          ← attn的输出直接加到residual
#   residual = hidden_states
#   hidden_states = mlp(post_attention_layernorm(hidden_states))
#   hidden_states = residual + hidden_states          ← mlp的输出直接加到residual
#
# 所以: self_attn返回(x, ...) → residual + x
#       如果hook把x变成α*x → residual + α*x = φ-Residual!
#
# 关键: 不碰DecoderLayer.forward! 不碰任何其他逻辑!

hooks = []

def make_attn_scale_hook(alpha):
    """缩放self_attn输出的hook — attn_output是tuple的第一个元素"""
    def hook(module, input, output):
        if isinstance(output, tuple):
            # Qwen2Attention.forward返回 (attn_output, attn_weights, past_key_value)
            # 只缩放第一个元素(attn_output), 其余不变
            return (alpha * output[0],) + output[1:]
        return alpha * output
    return hook

def make_mlp_scale_hook(alpha):
    """缩放mlp输出的hook — mlp直接返回tensor"""
    def hook(module, input, output):
        return alpha * output
    return hook

# 给每层挂hook
for i, layer in enumerate(model.model.layers):
    alpha = layer_alphas[i]
    
    # self_attn hook — 缩放attention输出
    h1 = layer.self_attn.register_forward_hook(make_attn_scale_hook(alpha))
    hooks.append(h1)
    
    # mlp hook — 缩放FFN输出
    h2 = layer.mlp.register_forward_hook(make_mlp_scale_hook(alpha))
    hooks.append(h2)

print(f"\n  [1] φ-Residual: forward_hook缩放子层输出 ✓")
print(f"  ★ DecoderLayer.forward完全不动!")
print(f"  ★ RoPE/KV cache/attention_mask完全不动!")
print(f"  ★ 挂载了 {len(hooks)} 个hook ({num_layers}层 × 2)")
print(f"\n  ✅ D10架构改造完成!")

# ============ Step 4: D10推理 ============
print("\n[4/5] D10改造后推理...")
try:
    with torch.no_grad():
        d10_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    d10_text = tokenizer.decode(d10_out[0], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print(f"输入: {test_input}")
    print(f"{'='*60}")
    print(f"基线: {baseline_text[:200]}")
    print(f"{'─'*60}")
    print(f"D10:  {d10_text[:200]}")
    print(f"{'='*60}")

    d10_tokens = tokenizer.encode(d10_text, add_special_tokens=False)
    baseline_tokens = tokenizer.encode(baseline_text, add_special_tokens=False)
    print(f"\n  基线token数: {len(baseline_tokens)}")
    print(f"  D10 token数: {len(d10_tokens)}")
    
    # 差异分析
    if d10_text != baseline_text:
        for ci in range(min(len(baseline_text), len(d10_text))):
            if baseline_text[ci] != d10_text[ci]:
                ctx = 20
                print(f"  首个差异位置: 字符{ci}")
                print(f"    基线: ...{baseline_text[max(0,ci-ctx):ci+ctx]}...")
                print(f"    D10:  ...{d10_text[max(0,ci-ctx):ci+ctx]}...")
                break
        else:
            print(f"  前缀相同, 长度不同 (基线{len(baseline_text)} vs D10{len(d10_text)})")
    else:
        print(f"  ★ 输出完全相同! α偏移未产生可观测差异")
        print(f"  → 可考虑增大D10_STRENGTH")
except Exception as e:
    print(f"  ⚠ D10推理失败: {e}")
    import traceback
    traceback.print_exc()
    d10_text = None

# ============ Step 5: 五动诊断 ============
print("\n[5/5] 五动层诊断...")
diag_tokens = tokenizer("The universe is", return_tensors="pt")
if 'position_ids' not in diag_tokens:
    diag_tokens['position_ids'] = torch.arange(
        diag_tokens['input_ids'].shape[-1], 
        dtype=torch.long
    ).unsqueeze(0)

layer_norms = []

try:
    with torch.no_grad():
        diag_hooks = []
        def nh(name):
            def fn(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                layer_norms.append((name, o.norm().item()))
            return fn
        for i, layer in enumerate(model.model.layers):
            diag_hooks.append(layer.register_forward_hook(nh(f"L{i}")))
        _ = model.generate(**diag_tokens, max_new_tokens=10, do_sample=False)
        for h in diag_hooks:
            h.remove()

    print("\n  层级激活范数 (五动周期):")
    for i, (nm, v) in enumerate(layer_norms):
        k = i % 5
        g = i // 5 % 2
        z2 = '⇄' if g == 1 else '→'
        alpha_str = f"α={layer_alphas[i]:.3f}"
        print(f"  {nm} [{C5_NAMES[k]}] {z2} {alpha_str} norm={v:.4f}")
    
    # 和基线norm对比
    print(f"\n  norm增长分析 (vs v8基线):")
    v8_baseline = [50.9, 84.6, 91.2, 142.7, 143.8, 158.2, 160.5, 161.6, 174.5, 174.9,
                   177.7, 179.5, 179.3, 184.8, 185.0, 190.3, 192.2, 192.5, 195.2, 195.7,
                   199.0, 202.1, 202.5, 213.5, 214.5, 216.7, 219.8, 220.9]
    max_ratio = 0
    for i, (nm, v) in enumerate(layer_norms):
        if i < len(v8_baseline):
            ratio = v / v8_baseline[i]
            max_ratio = max(max_ratio, ratio)
            status = "✓" if 0.8 < ratio < 1.3 else ("↑" if ratio >= 1.3 else "↓")
            print(f"  {nm}: {v:.1f} / baseline~{v8_baseline[i]:.1f} = {ratio:.2f}x {status}")
    print(f"\n  最大norm倍率: {max_ratio:.2f}x")
    if max_ratio < 1.5:
        print(f"  ★★★ norm在安全范围内! D10微扰可控! ★★★")
    elif max_ratio < 3.0:
        print(f"  norm有偏移但可接受, 可继续调参")
    else:
        print(f"  ⚠ norm偏移过大, 需降低D10_STRENGTH")
except Exception as e:
    print(f"  诊断失败: {e}")
    import traceback
    traceback.print_exc()

# 清理hooks
for h in hooks:
    h.remove()

print(f"\n{'='*60}")
print("D10五动管道架构改造 v13 完成!")
print(f"  Hook方案: forward_hook缩放子层输出")
print(f"  ALPHA_MODE = {ALPHA_MODE}, D10_STRENGTH = {D10_STRENGTH}")
print(f"{'='*60}")

if d10_text is not None:
    save_path = "./qwen2.5-1.5b-d10-v13"
    print(f"\n保存到 {save_path}...")
    try:
        # 保存前移除所有hook
        for h in hooks:
            h.remove()
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ 保存完成!")
    except Exception as e:
        print(f"  ⚠ 保存失败: {e}")
