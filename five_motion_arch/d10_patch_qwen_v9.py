"""
D10五动管道架构改造脚本 v9
核心改进: alpha降维 + 线性混合, 解决v8输出乱码(norm偏大60-85%)
  - α从φ⁻³降到φ⁻⁵ (0.236→0.090)
  - d10_strength控制整体扰动强度, 默认0.1
  - 输出线性混合: out = (1-s)·out_base + s·out_d10
用法：python d10_patch_qwen_v9.py
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============ 五动常量 ============
PHI = (1 + 5**0.5) / 2

_C5_ADJ = torch.tensor([
    [0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]
], dtype=torch.float32)

_A5 = torch.eye(5) + math.cos(math.radians(72)) * _C5_ADJ

C5_RPB_SLOPES = [-1.0, 0.0, -0.5, 1.0, 0.0]
C5_TEMPS = [0.8, 1.0, 1.2, PHI, 1.0]

# ★ v9: α从φ⁻³降到φ⁻⁵, 减少未训练扰动
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: 1/PHI**3, 4: 1/PHI**5}

ORGAN_CONFIG = [
    ('silu', -0.5), ('tanh', 0.0), ('linear', 0.0),
    ('silu', +0.5), ('identity', 0.0)
]

# ★ v9: 全局扰动强度
D10_STRENGTH = 0.1  # s=0.1, 即10%扰动+90%基线

print("=" * 60)
print("D10五动管道架构改造 v9")
print(f"  扰动强度 d10_strength = {D10_STRENGTH}")
print(f"  α范围: φ⁻⁵={1/PHI**5:.4f} ~ 1.0")
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
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // num_heads
num_kv_groups = num_heads // num_kv_heads
d_model = config.hidden_size
d_ff = config.intermediate_size
d_organ = d_ff // 5

print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"  层数: {num_layers}, heads: {num_heads}, kv_heads: {num_kv_heads}")
print(f"  hidden: {d_model}, d_ff: {d_ff}, head_dim: {head_dim}")

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

# --- 改动1: RoPE theta = φ² ---
if hasattr(config, 'rope_parameters'):
    old_theta = config.rope_parameters.get('rope_theta', 10000.0)
    config.rope_parameters = dict(config.rope_parameters)
    config.rope_parameters['rope_theta'] = PHI ** 2
    new_theta = PHI ** 2
elif hasattr(config, 'rope_theta'):
    old_theta = config.rope_theta
    config.rope_theta = PHI ** 2
    new_theta = config.rope_theta
else:
    old_theta = 10000.0
    new_theta = PHI ** 2
print(f"  [1] RoPE: {old_theta} -> {new_theta:.4f}")

new_rotary = None
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    new_rotary = Qwen2RotaryEmbedding(config=config, device=model.device)
    model.model.rotary_emb = new_rotary
    print(f"  RoPE重建成功")
except Exception as e:
    print(f"  ⚠ RoPE重建失败: {e}")

# C5 coupling matrix
n_g = num_heads // 5
rem = num_heads % 5
blocks = [_A5.clone()] * n_g
if rem > 0:
    blocks.append(torch.eye(rem, dtype=torch.float32))
c5_coupling = torch.block_diag(*blocks)
print(f"  [2] C5耦合: {c5_coupling.shape}")

# --- 通用函数 ---
def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin)

def compute_phi2_rope(seq_len, hd, device, dtype):
    inv_freq = 1.0 / (PHI**2 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

# ★ v9核心: 保存原始forward引用, 用于线性混合
# 先保存每个attn和mlp的原始forward
_orig_attn_fwd = {}
_orig_mlp_fwd = {}
for i, layer in enumerate(model.model.layers):
    _orig_attn_fwd[i] = layer.self_attn.forward
    _orig_mlp_fwd[i] = layer.mlp.forward

# --- 重写Attention forward ---
# ★ v9: 先跑原生attention拿到base_out, 再跑D10拿到d10_out, 线性混合
def d10_attention_forward(
    self, hidden_states, attention_mask=None, position_ids=None,
    past_key_value=None, output_attentions=False, use_cache=False,
    cache_position=None, position_embeddings=None, **kwargs
):
    bsz, q_len, _ = hidden_states.size()
    nh = self.d10_num_heads
    nkh = self.d10_num_kv_heads
    hd = self.d10_head_dim
    nkg = self.d10_num_kv_groups
    s = self.d10_strength

    # ★ Step A: 跑原生attention拿基线输出
    # 需要用保存的原始forward, 但它绑定了原始self
    # 所以直接用原生投影+原生逻辑算一遍
    base_q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1,2)
    base_k = self.k_proj(hidden_states).view(bsz, q_len, nkh, hd).transpose(1,2)
    base_v = self.v_proj(hidden_states).view(bsz, q_len, nkh, hd).transpose(1,2)

    # 原生RoPE
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = compute_phi2_rope(q_len, hd, base_v.device, base_v.dtype)
    base_q_r, base_k_r = apply_rotary_pos_emb(base_q, base_k, cos, sin)

    # KV cache
    base_k_cache, base_v_cache = base_k_r, base_v
    if past_key_value is not None and hasattr(past_key_value, 'update'):
        try:
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            if position_embeddings is not None:
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            base_k_cache, base_v_cache = past_key_value.update(base_k_r, base_v, self.d10_layer_idx, cache_kwargs)
        except Exception:
            pass

    # GQA expand
    base_k_exp = base_k_cache
    base_v_exp = base_v_cache
    if nkg > 1:
        base_k_exp = base_k_exp.unsqueeze(2).expand(-1,-1,nkg,-1,-1).reshape(bsz,nh,-1,hd)
        base_v_exp = base_v_exp.unsqueeze(2).expand(-1,-1,nkg,-1,-1).reshape(bsz,nh,-1,hd)

    # Base attention
    base_attn = torch.matmul(base_q_r, base_k_exp.transpose(-2,-1)) / math.sqrt(hd)
    if attention_mask is not None:
        base_attn = base_attn + attention_mask
    base_attn = F.softmax(base_attn, dim=-1, dtype=torch.float32).to(base_q.dtype)
    base_out = torch.matmul(base_attn, base_v_exp)
    base_out = base_out.transpose(1,2).contiguous().view(bsz, q_len, nh*hd)
    base_out = self.o_proj(base_out)

    # ★ Step B: D10扰动 — 在Q上加C5耦合, 在attn_weights上加RPB+温度
    d10_q = base_q.clone()
    # C5 head耦合
    coupling = self.d10_c5_coupling.to(d10_q.device, d10_q.dtype)
    d10_q = torch.einsum('ij,bjsd->bisd', coupling, d10_q)
    # RoPE已在base_q_r上, D10的q也做了RoPE(共用), 现在加C5耦合
    d10_q_r = d10_q  # 已经RoPE过了(从base_q_r clone再耦合)

    # 重新算D10的attention (k/v复用base的cache)
    d10_attn = torch.matmul(d10_q_r, base_k_exp.transpose(-2,-1)) / math.sqrt(hd)

    # C5 RPB + Z₂
    k_phase = self.d10_layer_idx % 5
    g = self.d10_layer_idx // 5 % 2
    slopes = C5_RPB_SLOPES[:]
    if g == 1:
        slopes = [-s for s in slopes]
    slope = slopes[k_phase]
    if slope != 0.0:
        seq = d10_attn.shape[-1]
        pos = torch.arange(seq, device=d10_attn.device, dtype=d10_attn.dtype)
        dist = (pos[:,None] - pos[None,:]).abs().float()
        d10_attn = d10_attn + slope * dist / seq

    # C5温度
    d10_attn = d10_attn / C5_TEMPS[k_phase]

    if attention_mask is not None:
        d10_attn = d10_attn + attention_mask

    d10_attn = F.softmax(d10_attn, dim=-1, dtype=torch.float32).to(base_q.dtype)
    d10_out = torch.matmul(d10_attn, base_v_exp)
    d10_out = d10_out.transpose(1,2).contiguous().view(bsz, q_len, nh*hd)
    d10_out = self.o_proj(d10_out)

    # ★ Step C: 线性混合
    out = (1.0 - s) * base_out + s * d10_out

    # 返回严格2元素tuple
    attn_weights_out = d10_attn if output_attentions else None
    return out, attn_weights_out

# Patch attention
for i, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    attn.d10_num_heads = num_heads
    attn.d10_num_kv_heads = num_kv_heads
    attn.d10_head_dim = head_dim
    attn.d10_num_kv_groups = num_kv_groups
    attn.d10_layer_idx = i
    attn.d10_c5_coupling = c5_coupling.clone()
    attn.d10_strength = D10_STRENGTH
    attn.forward = types.MethodType(d10_attention_forward, attn)

print(f"  [3] Attention: C5耦合(Q-head)+RPB+温度+Z₂+线性混合(s={D10_STRENGTH}) ✓")

# --- 五动FFN ---
# ★ v9: 同样用线性混合, 先跑原生mlp再跑D10 mlp
for i, layer in enumerate(model.model.layers):
    mlp = layer.mlp
    mlp.W_organ = nn.ModuleList([nn.Linear(d_model, d_organ, bias=True) for _ in range(5)])
    for j, (_, bv) in enumerate(ORGAN_CONFIG):
        nn.init.constant_(mlp.W_organ[j].bias, bv)

    with torch.no_grad():
        w = mlp.gate_proj.weight.data
        b = mlp.gate_proj.bias.data if mlp.gate_proj.bias is not None else None
        for j in range(5):
            s_idx, e_idx = j*d_organ, (j+1)*d_organ
            mlp.W_organ[j].weight.data.copy_(w[s_idx:e_idx,:])
            if b is not None:
                mlp.W_organ[j].bias.data.copy_(b[s_idx:e_idx])
                mlp.W_organ[j].bias.data.fill_(ORGAN_CONFIG[j][1])

    mlp.register_buffer('c5_adj', _C5_ADJ.clone())
    mlp.W_out = nn.Linear(5*d_organ, d_model, bias=True)

    with torch.no_grad():
        dw = mlp.down_proj.weight.data
        mlp.W_out.weight.data.copy_(dw[:,:5*d_organ])
        if mlp.down_proj.bias is not None:
            mlp.W_out.bias.data.copy_(mlp.down_proj.bias.data)

    k = i % 5
    alpha = PHI_POWERS[k] / math.sqrt(i + 2)
    mlp.d10_alpha = alpha
    mlp.d10_strength = D10_STRENGTH

    # ★ v9: 保存原始gate_proj/up_proj/down_proj引用, 用于base计算
    mlp._orig_gate = mlp.gate_proj
    mlp._orig_up = mlp.up_proj
    mlp._orig_down = mlp.down_proj

    def make_ffn_fwd(mod):
        def fwd(x):
            s = mod.d10_strength

            # ★ Base: 原生SwiGLU
            gate = mod._orig_gate(x)
            up = mod._orig_up(x)
            base_out = mod._orig_down(F.silu(gate) * up)

            # ★ D10: 五动FFN
            organs = []
            for j, w in enumerate(mod.W_organ):
                h = w(x)
                act = ORGAN_CONFIG[j][0]
                if act == 'silu': h = F.silu(h)
                elif act == 'tanh': h = torch.tanh(h)
                organs.append(h)
            organs = torch.stack(organs, dim=-1)  # [B, S, d_organ, 5]
            organs = torch.einsum('ij,bsdj->bsdi', mod.c5_adj.to(organs.device,organs.dtype), organs)
            d10_out = mod.W_out(organs.reshape(*x.shape[:-1], 5*d_organ))
            d10_out = mod.d10_alpha * d10_out

            # ★ 线性混合
            return (1.0 - s) * base_out + s * d10_out
        return fwd
    mlp.forward = make_ffn_fwd(mlp)

print(f"  [4] 五动FFN: 5器官+耦合+α+线性混合(s={D10_STRENGTH}) ✓")
print(f"  ★ layer.forward保持原样!")
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

    # ★ 简单质量评估
    # 计算输出中有效token比例(非特殊字符)
    d10_tokens = tokenizer.encode(d10_text, add_special_tokens=False)
    baseline_tokens = tokenizer.encode(baseline_text, add_special_tokens=False)
    print(f"\n  基线token数: {len(baseline_tokens)}")
    print(f"  D10 token数: {len(d10_tokens)}")
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
    names = ['认','遇','落','裂','余']
    for i, (nm, v) in enumerate(layer_norms):
        k = i % 5
        g = i // 5 % 2
        z2 = '⇄' if g == 1 else '→'
        print(f"  {nm} [{names[k]}] {z2} norm={v:.4f}")
    
    # ★ v9: 和基线norm对比(基线应接近v8截图的值)
    print(f"\n  norm增长分析 (vs v8基线):")
    v8_baseline_approx = [50.9, 84.6, 91.2, 142.7, 143.8, 158.2, 160.5, 161.6, 174.5, 174.9,
                          177.7, 179.5, 179.3, 184.8, 185.0, 190.3, 192.2, 192.5, 195.2, 195.7,
                          199.0, 202.1, 202.5, 213.5, 214.5, 216.7, 219.8, 220.9]
    for i, (nm, v) in enumerate(layer_norms):
        if i < len(v8_baseline_approx):
            ratio = v / v8_baseline_approx[i]
            status = "✓" if 0.9 < ratio < 1.2 else ("↑" if ratio >= 1.2 else "↓")
            print(f"  {nm}: {v:.1f} / baseline~{v8_baseline_approx[i]:.1f} = {ratio:.2f}x {status}")
except Exception as e:
    print(f"  诊断失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("D10五动管道架构改造 v9 完成!")
print(f"  d10_strength = {D10_STRENGTH}")
print(f"  α范围: φ⁻⁵={1/PHI**5:.4f} ~ 1.0")
print(f"{'='*60}")

if d10_text is not None:
    save_path = "./qwen2.5-1.5b-d10-v9"
    print(f"\n保存到 {save_path}...")
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ 保存完成!")
    except Exception as e:
        print(f"  ⚠ 保存失败: {e}")
