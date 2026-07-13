"""
D10五动管道架构改造脚本 v12
核心洞察: v11证明了保持RoPE时D10输出通顺且有语义偏移(好!)
但hook里算完整C5-attention导致norm爆炸(坏!)

v12: 极简方案 — 只改residual连接缩放(φ-Residual)
  不算额外forward, 不碰attention/FFN内部, 只缩skip connection
  这是φ-Residual的原始定义: 每层的residual贡献按五动α缩放
  
  DecoderLayer: hidden = residual + α * sublayer_output
  原版: α=1 (所有层等权)
  D10:  α=PHI_POWERS[k]/√(i+2) (五动节奏)
  
  零额外计算, 零norm风险, 纯结构性改动
用法：python d10_patch_qwen_v12.py
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

# ★ v12: α缩放模式
# "full" = 直接用PHI_POWERS[k]/√(i+2) (可能变化太大)
# "gentle" = 1 + strength * (PHI_POWERS[k]/√(i+2) - 1) (温和偏移)
ALPHA_MODE = "gentle"
D10_STRENGTH = 0.3  # gentle模式下的偏移强度

print("=" * 60)
print("D10五动管道架构改造 v12")
print(f"  极简方案: 只改residual连接缩放(φ-Residual)")
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
        # 温和偏移: alpha = 1 + strength * (base_alpha - 1)
        # 当base_alpha > 1时alpha > 1 (裂层放大)
        # 当base_alpha < 1时alpha < 1 (其他层缩小)
        alpha = 1.0 + D10_STRENGTH * (base_alpha - 1.0)
    else:
        alpha = base_alpha
    
    # Z₂翻转: 奇数组residual方向反转
    if g == 1:
        alpha = 2.0 - alpha  # 关于1对称翻转
    
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

# ★★★ 核心: 替换DecoderLayer.forward ★★★
# Qwen2DecoderLayer.forward结构:
#   residual = hidden_states
#   hidden_states = input_layernorm(hidden_states)
#   hidden_states = self_attn(hidden_states, ...)
#   hidden_states = residual + hidden_states    ← 这里改!
#   residual = hidden_states
#   hidden_states = post_attention_layernorm(hidden_states)
#   hidden_states = mlp(hidden_states)
#   hidden_states = residual + hidden_states    ← 这里也改!
#
# 我们把 residual + hidden_states 改成 residual + α * hidden_states
# 这等价于缩放子层输出, 不改子层内部

import types

# 保存原始forward
_orig_layer_fwd = {}
for i, layer in enumerate(model.model.layers):
    _orig_layer_fwd[i] = layer.forward

def make_d10_layer_forward(layer_idx, orig_fwd, alpha_attn, alpha_mlp):
    """创建D10修改版DecoderLayer forward"""
    def d10_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs
    ):
        # ★ 调用原始forward拿到输出
        # 但我们不能直接调原始forward然后改输出, 因为residual在内部
        # 必须拆开decoder layer的forward
        
        layer = model.model.layers[layer_idx]
        
        # --- 复制Qwen2DecoderLayer.forward逻辑, 只改residual缩放 ---
        
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        
        # Self Attention
        attn_outputs = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        attn_output = attn_outputs[0]
        
        # ★ D10: φ-Residual — 缩放attention输出
        hidden_states = residual + alpha_attn * attn_output
        
        # Post-attention residual
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        
        # MLP
        mlp_output = layer.mlp(hidden_states)
        
        # ★ D10: φ-Residual — 缩放MLP输出
        hidden_states = residual + alpha_mlp * mlp_output
        
        # 返回格式跟原始一样
        present_key_value = None
        if use_cache:
            present_key_value = attn_outputs[1] if len(attn_outputs) > 1 else None
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[2] if len(attn_outputs) > 2 else None,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs
    
    return d10_forward

# Patch每层
for i, layer in enumerate(model.model.layers):
    k = i % 5
    # attn和mlp用同一个alpha (都是这个层的五动相位)
    alpha = layer_alphas[i]
    layer.forward = make_d10_layer_forward(i, _orig_layer_fwd[i], alpha, alpha)

print(f"  [1] φ-Residual: residual + α*sublayer_output ✓")
print(f"  ★ attention/mlp forward完全不动!")
print(f"  ★ RoPE/KV cache/attention_mask完全不动!")
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
        hooks = []
        def nh(name):
            def fn(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                layer_norms.append((name, o.norm().item()))
            return fn
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(nh(f"L{i}")))
        _ = model.generate(**diag_tokens, max_new_tokens=10, do_sample=False)
        for h in hooks:
            h.remove()

    print("\n  层级激活范数 (五动周期):")
    for i, (nm, v) in enumerate(layer_norms):
        k = i % 5
        g = i // 5 % 2
        z2 = '⇄' if g == 1 else '→'
        alpha_str = f"α={layer_alphas[i]:.3f}"
        print(f"  {nm} [{C5_NAMES[k]}] {z2} {alpha_str} norm={v:.4f}")
    
    # 和基线norm对比
    print(f"\n  norm增长分析 (vs 基线):")
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

print(f"\n{'='*60}")
print("D10五动管道架构改造 v12 完成!")
print(f"  极简方案: 只改residual连接缩放")
print(f"  ALPHA_MODE = {ALPHA_MODE}, D10_STRENGTH = {D10_STRENGTH}")
print(f"{'='*60}")

if d10_text is not None:
    save_path = "./qwen2.5-1.5b-d10-v12"
    print(f"\n保存到 {save_path}...")
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ 保存完成!")
    except Exception as e:
        print(f"  ⚠ 保存失败: {e}")
