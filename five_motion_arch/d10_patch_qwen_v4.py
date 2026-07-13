"""
D10五动管道架构改造脚本 v4
修复: 1) 闭包变量捕获bug 2) layer forward返回格式 3) transformers 5.13.1兼容
用法：python d10_patch_qwen_v4.py
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

_C5_ADJ = torch.tensor([
    [0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]
], dtype=torch.float32)

_A5 = torch.eye(5) + math.cos(math.radians(72)) * _C5_ADJ

C5_RPB_SLOPES = [-1.0, 0.0, -0.5, 1.0, 0.0]
C5_TEMPS = [0.8, 1.0, 1.2, PHI, 1.0]
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}
ORGAN_CONFIG = [
    ('silu', -0.5), ('tanh', 0.0), ('linear', 0.0),
    ('silu', +0.5), ('identity', 0.0)
]

print("=" * 60)
print("D10五动管道架构改造 v4")
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

print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"  层数: {config.num_hidden_layers}, heads: {config.num_attention_heads}")
print(f"  hidden: {config.hidden_size}, d_ff: {config.intermediate_size}")

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

# 重建RoPE
new_rotary = None
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    new_rotary = Qwen2RotaryEmbedding(config=config, device=model.device)
    print(f"  RoPE重建成功")
except Exception as e:
    print(f"  ⚠ RoPE重建失败: {e}, 将用手动计算")

# C5 coupling
num_heads = config.num_attention_heads
n_g = num_heads // 5
rem = num_heads % 5
blocks = [_A5.clone()] * n_g
if rem > 0:
    blocks.append(torch.eye(rem, dtype=torch.float32))
c5_coupling = torch.block_diag(*blocks)
print(f"  [2] C5耦合: {c5_coupling.shape}")

d_ff = config.intermediate_size
d_organ = d_ff // 5
d_model = config.hidden_size

# --- 通用函数 ---
def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin)

# --- 手动φ²-RoPE计算 ---
def compute_phi2_rope(seq_len, head_dim, device, dtype):
    inv_freq = 1.0 / (PHI**2 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

# --- 重写Attention forward ---
def d10_attention_forward(
    self, hidden_states, attention_mask=None, position_ids=None,
    past_key_value=None, output_attentions=False, use_cache=False,
    cache_position=None, position_embeddings=None, **kwargs
):
    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

    # RoPE - 优先用position_embeddings(5.x), fallback手动
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = compute_phi2_rope(q_len, self.head_dim, v.device, v.dtype)
    
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if self.num_key_value_groups > 1:
        k = k.unsqueeze(2).expand(-1,-1,self.num_key_value_groups,-1,-1).reshape(bsz,self.num_heads,-1,self.head_dim)
        v = v.unsqueeze(2).expand(-1,-1,self.num_key_value_groups,-1,-1).reshape(bsz,self.num_heads,-1,self.head_dim)

    attn_weights = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)

    # C5 head耦合
    coupling = self.c5_coupling.to(attn_weights.device, attn_weights.dtype)
    attn_weights = torch.matmul(coupling, attn_weights)

    # C5 RPB + Z₂
    lidx = self.layer_idx
    k_phase = lidx % 5
    g = lidx // 5 % 2
    slopes = C5_RPB_SLOPES[:]
    if g == 1:
        slopes = [-s for s in slopes]
    slope = slopes[k_phase]
    if slope != 0.0:
        seq = attn_weights.shape[-1]
        pos = torch.arange(seq, device=attn_weights.device, dtype=attn_weights.dtype)
        dist = (pos[:,None] - pos[None,:]).abs().float()
        attn_weights = attn_weights + slope * dist / seq

    # C5温度
    attn_weights = attn_weights / C5_TEMPS[k_phase]

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, v)
    out = out.transpose(1,2).contiguous().view(bsz, q_len, self.num_heads*self.head_dim)
    out = self.o_proj(out)

    return (out, attn_weights) if output_attentions else (out,)

# Patch attention - 直接替换forward，不用__get__
for i, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    attn.layer_idx = i
    attn.c5_coupling = c5_coupling.clone()
    import types
    attn.forward = types.MethodType(d10_attention_forward, attn)

print(f"  [3] Attention: C5耦合+RPB+温度+Z₂ ✓")

# --- 五动FFN ---
for i, layer in enumerate(model.model.layers):
    mlp = layer.mlp
    mlp.W_organ = nn.ModuleList([nn.Linear(d_model, d_organ, bias=True) for _ in range(5)])
    for j, (_, bv) in enumerate(ORGAN_CONFIG):
        nn.init.constant_(mlp.W_organ[j].bias, bv)

    with torch.no_grad():
        w = mlp.gate_proj.weight.data
        b = mlp.gate_proj.bias.data if mlp.gate_proj.bias is not None else None
        for j in range(5):
            s, e = j*d_organ, (j+1)*d_organ
            mlp.W_organ[j].weight.data.copy_(w[s:e,:])
            if b is not None:
                mlp.W_organ[j].bias.data.copy_(b[s:e])
                mlp.W_organ[j].bias.data.fill_(ORGAN_CONFIG[j][1])

    mlp.register_buffer('c5_adj', _C5_ADJ.clone())
    mlp.W_out = nn.Linear(5*d_organ, d_model, bias=True)

    with torch.no_grad():
        dw = mlp.down_proj.weight.data
        mlp.W_out.weight.data.copy_(dw[:,:5*d_organ])
        if mlp.down_proj.bias is not None:
            mlp.W_out.bias.data.copy_(mlp.down_proj.bias.data)

    def make_ffn_fwd(mod):
        def fwd(x):
            organs = []
            for j, w in enumerate(mod.W_organ):
                h = w(x)
                act = ORGAN_CONFIG[j][0]
                if act == 'silu': h = F.silu(h)
                elif act == 'tanh': h = torch.tanh(h)
                organs.append(h)
            organs = torch.stack(organs, dim=-1)
            organs = torch.einsum('ij,bsdj->bsdi', mod.c5_adj.to(organs.device,organs.dtype), organs)
            return mod.W_out(organs.reshape(*x.shape[:-1], 5*d_organ))
        return fwd
    mlp.forward = make_ffn_fwd(mlp)

print(f"  [4] 五动FFN: 5器官+耦合 ✓")

# --- φ-Residual + Z₂ 层级patch ---
# ★修复闭包bug: 正确捕获layer引用
for i, layer in enumerate(model.model.layers):
    k = i % 5
    alpha = PHI_POWERS[k] / math.sqrt(i + 2)
    g = i // 5 % 2

    def make_layer_fwd(lyr, a, lidx):
        def fwd(hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
            residual = hidden_states
            hidden_states = lyr.input_layernorm(hidden_states)
            attn_out = lyr.self_attn(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )
            hidden_states = residual + a * attn_out[0]

            residual = hidden_states
            hidden_states = lyr.post_attention_layernorm(hidden_states)
            hidden_states = lyr.mlp(hidden_states)
            hidden_states = residual + a * hidden_states

            # 返回格式与Qwen2DecoderLayer一致
            if use_cache:
                return (hidden_states, past_key_value)
            return (hidden_states,)
        return fwd
    layer.forward = types.MethodType(make_layer_fwd(layer, alpha, i), layer)

print(f"  [5] φ-Residual: 呼吸节奏 ✓")
print(f"\n  ✅ D10架构改造完成!")
print(f"  - φ²-RoPE θ={new_theta:.4f}")
print(f"  - C5-Attention: 耦合+RPB+温度+Z₂")
print(f"  - 五动FFN: 5器官+耦合")
print(f"  - φ-Residual: φ呼吸")

# ============ Step 4: D10推理 ============
print("\n[4/5] D10改造后推理...")
try:
    with torch.no_grad():
        d10_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    d10_text = tokenizer.decode(d10_out[0], skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print(f"输入: {test_input}")
    print(f"{'='*60}")
    print(f"基线: {baseline_text[:150]}")
    print(f"{'─'*60}")
    print(f"D10:  {d10_text[:150]}")
    print(f"{'='*60}")
except Exception as e:
    print(f"  ⚠ D10推理失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 降级测试：不用generate，直接forward
    print("\n  降级测试：直接forward单步...")
    try:
        with torch.no_grad():
            out = model.model(**inputs)
            logits = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
            next_token = torch.argmax(logits[0, -1], dim=-1)
            next_word = tokenizer.decode(next_token)
            print(f"  D10单步预测下一个token: {next_word}")
    except Exception as e2:
        print(f"  降级也失败: {e2}")
        import traceback
        traceback.print_exc()
    d10_text = "(推理失败)"

# ============ Step 5: 五动诊断 ============
print("\n[5/5] 五动层诊断...")
diag_in = tokenizer("The universe is", return_tensors="pt")
layer_norms = []
hooks = []

def nh(name):
    def fn(mod, inp, out):
        if isinstance(out, tuple):
            layer_norms.append((name, out[0].norm().item()))
        else:
            layer_norms.append((name, out.norm().item()))
    return fn

try:
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(nh(f"L{i}")))
        _ = model.generate(**diag_in, max_new_tokens=10, do_sample=False)
        for h in hooks:
            h.remove()

    print("\n  层级激活范数 (五动周期):")
    names = ['认','遇','落','裂','余']
    for i, (nm, v) in enumerate(layer_norms):
        k = i % 5
        g = i // 5 % 2
        z2 = '⇄' if g == 1 else '→'
        print(f"  {nm} [{names[k]}] {z2} norm={v:.4f}")
except Exception as e:
    print(f"  诊断失败: {e}")
    # 降级：直接forward收集
    print("  降级诊断...")
    try:
        layer_norms = []
        with torch.no_grad():
            x = model.model.embed_tokens(diag_in.input_ids)
            for i, layer in enumerate(model.model.layers):
                x = layer(x, position_ids=diag_in.position_ids)[0]
                layer_norms.append((f"L{i}", x.norm().item()))
        print("\n  层级激活范数 (降级模式):")
        names = ['认','遇','落','裂','余']
        for i, (nm, v) in enumerate(layer_norms):
            k = i % 5
            g = i // 5 % 2
            z2 = '⇄' if g == 1 else '→'
            print(f"  {nm} [{names[k]}] {z2} norm={v:.4f}")
    except Exception as e2:
        print(f"  降级诊断也失败: {e2}")

print(f"\n{'='*60}")
print("D10五动管道架构改造完成!")
print(f"{'='*60}")

if d10_text != "(推理失败)":
    save_path = "./qwen2.5-1.5b-d10"
    print(f"\n保存到 {save_path}...")
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ 保存完成!")
    except Exception as e:
        print(f"  ⚠ 保存失败: {e}")
else:
    print("\n推理失败，跳过保存")
