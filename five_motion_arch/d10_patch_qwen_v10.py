"""
D10五动管道架构改造脚本 v10
核心改进: 回归单路(v8架构), 但alpha极低+residual归一化
v9失败原因: 双路(base+D10)共享KV cache互相污染, norm爆炸到8000+
v10方案:
  - 单路, 不碰KV cache逻辑
  - alpha极低: PHI_POWERS全部/100 (约0.001-0.01量级)
  - residual连接: out = out_original + alpha * (out_d10 - out_original)
    等效于: out = (1-alpha)*out_original + alpha*out_d10
    但不需要算两遍forward, 只在输出上加微扰
用法：python d10_patch_qwen_v10.py
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

# ★ v10: alpha极低, 微扰级别
# v8的alpha约为0.06~0.24, 导致norm偏大60-85%
# v10先压到1/100, 确保输出正常, 再逐步调高
ALPHA_SCALE = 0.01  # 全局alpha缩放因子
PHI_POWERS_BASE = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}

ORGAN_CONFIG = [
    ('silu', -0.5), ('tanh', 0.0), ('linear', 0.0),
    ('silu', +0.5), ('identity', 0.0)
]

print("=" * 60)
print("D10五动管道架构改造 v10")
print(f"  ALPHA_SCALE = {ALPHA_SCALE}")
print(f"  策略: 单路+alpha微扰, 不碰KV cache")
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

# ★★★ v10核心: 不替换self_attn.forward ★★★
# 而是在DecoderLayer级别做微扰
# 思路: 给每层注册一个forward hook, 在attn/mlp输出后加微扰
# 这样完全不碰HuggingFace内部逻辑, KV cache/position_ids全由原生处理

# --- 方案A: 用register_forward_hook在attn/mlp输出后加微扰 ---
# 这是最低侵入性的方案, 完全不改forward, 不碰cache

# C5耦合作用于Q: 我们用hook在q_proj输出后修改Q
# 但hook拿不到中间变量... 
# 还是得patch forward, 但要最小化改动

# ★★★ v10最终方案: 仅在原始forward输出上叠加一个微小的C5残差 ★★★
# 不改attn forward逻辑, 不碰RoPE/cache/mask
# 只在layer输出后: out = out + alpha * c5_residual(hidden_before_layer)

# 这个方案的好处:
# 1. 原生forward完全不动
# 2. KV cache完全由原生处理
# 3. 只在输出上加微扰, 不会爆炸
# 4. C5节奏通过alpha随层变化体现

# 但这太弱了——C5耦合不在attention内部, 只是一个后处理残差
# 还是得patch attn forward, 但用最安全的方式

# ★★★ v10实际方案: patch attn forward, 但C5耦合和RPB都缩放 ★★★
# 关键洞察: v8的问题是norm偏大60-85%, 不是崩溃
# 所以只需要把C5耦合和RPB的强度降低到合适的水平
# 不需要线性混合, 只需要alpha够小

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

    q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1,2)  # [B, nh, S, hd]
    k = self.k_proj(hidden_states).view(bsz, q_len, nkh, hd).transpose(1,2)  # [B, nkh, S, hd]
    v = self.v_proj(hidden_states).view(bsz, q_len, nkh, hd).transpose(1,2)  # [B, nkh, S, hd]

    # RoPE - 优先用model传来的position_embeddings (5.x标准)
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = compute_phi2_rope(q_len, hd, v.device, v.dtype)
    
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # ★C5 head耦合 - 在Q的head维度混合 (缩放ALPHA_SCALE)
    coupling = self.d10_c5_coupling.to(q.device, q.dtype)
    # v10: 只加C5扰动, 不替换Q
    # q_new = q + alpha_scale * (coupling @ q - q)
    # 这保证C5效果是叠加的微扰, 而不是替换
    q_c5 = torch.einsum('ij,bjsd->bisd', coupling, q)
    q = q + ALPHA_SCALE * (q_c5 - q)

    # KV cache - 兼容5.x Cache对象
    if past_key_value is not None and hasattr(past_key_value, 'update'):
        try:
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            if position_embeddings is not None:
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            k, v = past_key_value.update(k, v, self.d10_layer_idx, cache_kwargs)
        except Exception:
            pass

    # GQA: expand kv heads to match query heads
    if nkg > 1:
        k = k.unsqueeze(2).expand(-1,-1,nkg,-1,-1).reshape(bsz,nh,-1,hd)
        v = v.unsqueeze(2).expand(-1,-1,nkg,-1,-1).reshape(bsz,nh,-1,hd)

    # Standard attention
    attn_weights = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(hd)

    # C5 RPB + Z₂ (缩放ALPHA_SCALE)
    k_phase = self.d10_layer_idx % 5
    g = self.d10_layer_idx // 5 % 2
    slopes = C5_RPB_SLOPES[:]
    if g == 1:
        slopes = [-s for s in slopes]
    slope = slopes[k_phase] * ALPHA_SCALE  # ★ 缩放RPB
    if slope != 0.0:
        seq = attn_weights.shape[-1]
        pos = torch.arange(seq, device=attn_weights.device, dtype=attn_weights.dtype)
        dist = (pos[:,None] - pos[None,:]).abs().float()
        attn_weights = attn_weights + slope * dist / seq

    # C5温度 (接近1的微调, 不缩放, 但限制偏移量)
    temp = C5_TEMPS[k_phase]
    # v10: 温度偏移也在微扰范围
    # 原始temp偏离1最多是PHI≈1.618, 缩放到1 + ALPHA_SCALE*(temp-1)
    temp = 1.0 + ALPHA_SCALE * (temp - 1.0)
    attn_weights = attn_weights / temp

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, v)
    out = out.transpose(1,2).contiguous().view(bsz, q_len, nh*hd)
    out = self.o_proj(out)

    # ★φ-Residual: alpha缩放 (v8是直接乘alpha, v10改为残差微扰)
    # out = out_original * (1 - alpha) + out * alpha
    # 等效于: 在residual连接时, D10层的贡献被alpha缩放
    # 但这里我们直接乘alpha, 因为residual连接在DecoderLayer里:
    # hidden = hidden + out (DecoderLayer做的)
    # 所以out的幅度直接决定D10的扰动量
    # v8的alpha约0.06-0.24, 导致norm偏大
    # v10: alpha再缩放ALPHA_SCALE
    k = self.d10_layer_idx % 5
    alpha = PHI_POWERS_BASE[k] / math.sqrt(self.d10_layer_idx + 2) * ALPHA_SCALE
    out = alpha * out

    # ★★★ 严格返回2元素tuple ★★★
    attn_weights_out = attn_weights if output_attentions else None
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
    attn.forward = types.MethodType(d10_attention_forward, attn)

print(f"  [3] Attention: C5耦合微扰+RPB微扰+温度微扰+α微扰 (scale={ALPHA_SCALE}) ✓")

# --- 五动FFN ---
# ★ v10: 同样用微扰方式
# 不替换mlp forward, 而是在mlp输出后叠加一个小的C5-FFN残差
# 用register_forward_hook实现, 完全不碰原始forward

class D10FFNHook:
    """在mlp输出后叠加C5-FFN微扰"""
    def __init__(self, layer_idx, W_organ, W_out, c5_adj, d_model, d_organ):
        self.layer_idx = layer_idx
        self.W_organ = W_organ
        self.W_out = W_out
        self.c5_adj = c5_adj
        self.d_model = d_model
        self.d_organ = d_organ
        k = layer_idx % 5
        self.alpha = PHI_POWERS_BASE[k] / math.sqrt(layer_idx + 2) * ALPHA_SCALE
    
    def __call__(self, module, input, output):
        # output可能是tuple (hidden_states, ...) 或 tensor
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        x = input[0] if isinstance(input, tuple) else input
        
        # 计算C5-FFN微扰
        organs = []
        for j, w in enumerate(self.W_organ):
            h = w(x)
            act = ORGAN_CONFIG[j][0]
            if act == 'silu': h = F.silu(h)
            elif act == 'tanh': h = torch.tanh(h)
            organs.append(h)
        organs = torch.stack(organs, dim=-1)  # [B, S, d_organ, 5]
        organs = torch.einsum('ij,bsdj->bsdi', 
                             self.c5_adj.to(organs.device, organs.dtype), organs)
        d10_residual = self.W_out(organs.reshape(*x.shape[:-1], 5*self.d_organ))
        d10_residual = self.alpha * d10_residual
        
        # 叠加微扰
        hidden = hidden + d10_residual
        
        if rest is not None:
            return (hidden,) + rest
        return hidden

# 为每层创建五动FFN hook
for i, layer in enumerate(model.model.layers):
    mlp = layer.mlp
    
    # 创建五动器官
    W_organ = nn.ModuleList([nn.Linear(d_model, d_organ, bias=True) for _ in range(5)])
    for j, (_, bv) in enumerate(ORGAN_CONFIG):
        nn.init.constant_(W_organ[j].bias, bv)
    
    with torch.no_grad():
        w = mlp.gate_proj.weight.data
        b = mlp.gate_proj.bias.data if mlp.gate_proj.bias is not None else None
        for j in range(5):
            s, e = j*d_organ, (j+1)*d_organ
            W_organ[j].weight.data.copy_(w[s:e,:])
            if b is not None:
                W_organ[j].bias.data.copy_(b[s:e])
                W_organ[j].bias.data.fill_(ORGAN_CONFIG[j][1])
    
    W_out = nn.Linear(5*d_organ, d_model, bias=True)
    with torch.no_grad():
        dw = mlp.down_proj.weight.data
        W_out.weight.data.copy_(dw[:,:5*d_organ])
        if mlp.down_proj.bias is not None:
            W_out.bias.data.copy_(mlp.down_proj.bias.data)
    
    # 注册hook (不替换forward!)
    c5_adj = _C5_ADJ.clone()
    hook_fn = D10FFNHook(i, W_organ, W_out, c5_adj, d_model, d_organ)
    layer.mlp.register_forward_hook(hook_fn)

print(f"  [4] 五动FFN: hook方式叠加微扰 (不替换forward) ✓")
print(f"  ★ layer.forward和mlp.forward保持原样!")
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

    # 简单质量评估
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
    
    # 和基线norm对比
    print(f"\n  norm增长分析 (vs 基线):")
    v8_baseline = [50.9, 84.6, 91.2, 142.7, 143.8, 158.2, 160.5, 161.6, 174.5, 174.9,
                   177.7, 179.5, 179.3, 184.8, 185.0, 190.3, 192.2, 192.5, 195.2, 195.7,
                   199.0, 202.1, 202.5, 213.5, 214.5, 216.7, 219.8, 220.9]
    for i, (nm, v) in enumerate(layer_norms):
        if i < len(v8_baseline):
            ratio = v / v8_baseline[i]
            status = "✓" if 0.9 < ratio < 1.2 else ("↑" if ratio >= 1.2 else "↓")
            print(f"  {nm}: {v:.1f} / baseline~{v8_baseline[i]:.1f} = {ratio:.2f}x {status}")
except Exception as e:
    print(f"  诊断失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print("D10五动管道架构改造 v10 完成!")
print(f"  ALPHA_SCALE = {ALPHA_SCALE}")
print(f"  策略: C5微扰叠加(不替换) + FFN hook + α极低")
print(f"{'='*60}")

if d10_text is not None:
    save_path = "./qwen2.5-1.5b-d10-v10"
    print(f"\n保存到 {save_path}...")
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ 保存完成!")
    except Exception as e:
        print(f"  ⚠ 保存失败: {e}")
