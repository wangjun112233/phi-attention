"""
D10五动管道架构改造脚本 v11
核心洞察: φ² RoPE不能直接套预训练模型!
  v8-v10失败根源: theta从1,000,000→2.618, 位置编码频率变化38万倍
  预训练模型的所有位置相关表征全部失效, 不管alpha多小都救不回来

v11策略: 保持原始RoPE, 只测试非位置相关的D10特征
  [1] C5 head耦合 — 微扰叠加 q = q + s*(q_c5 - q)
  [2] C5-RPB — 注意力位置偏置
  [3] C5温度 — 软调制
  [4] φ-Residual — alpha缩放 (residual路径, 不替换attn输出)
  [5] 五动FFN — hook叠加微扰

  不动的: RoPE, KV cache, attention forward主体逻辑
  
  预期: 输出应该和基线接近但可观测到D10节奏差异
用法：python d10_patch_qwen_v11.py
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

PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}

ORGAN_CONFIG = [
    ('silu', -0.5), ('tanh', 0.0), ('linear', 0.0),
    ('silu', +0.5), ('identity', 0.0)
]

# ★ v11: 全局扰动强度, 逐步可调
D10_STRENGTH = 0.05  # 5%扰动

print("=" * 60)
print("D10五动管道架构改造 v11")
print(f"  d10_strength = {D10_STRENGTH}")
print(f"  ★ 保持原始RoPE不动!")
print(f"  ★ 只测试非位置相关D10特征")
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
print("  ★ 不改RoPE! 保持原始theta=1000000")

# C5 coupling matrix
n_g = num_heads // 5
rem = num_heads % 5
blocks = [_A5.clone()] * n_g
if rem > 0:
    blocks.append(torch.eye(rem, dtype=torch.float32))
c5_coupling = torch.block_diag(*blocks)
print(f"  [1] C5耦合矩阵: {c5_coupling.shape}")

# ★★★ v11核心: 不替换attention forward, 用hook实现 ★★★
# 在每层DecoderLayer输出后叠加D10微扰
# 这样完全不碰原生forward/KV cache/RoPE

class D10LayerHook:
    """在DecoderLayer输出后叠加D10五动微扰
    包含: C5 head耦合扰动 + RPB偏置 + 温度调制 + φ-Residual
    """
    def __init__(self, layer_idx, c5_coupling, num_heads, num_kv_heads, 
                 head_dim, num_kv_groups, d_model, d_ff, d_organ, strength):
        self.layer_idx = layer_idx
        self.c5_coupling = c5_coupling
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_organ = d_organ
        self.strength = strength
        
        # 五动相位
        self.k_phase = layer_idx % 5
        self.g = layer_idx // 5 % 2  # Z₂翻转
        
        # φ-Residual alpha
        k = layer_idx % 5
        self.alpha = PHI_POWERS[k] / math.sqrt(layer_idx + 2)
    
    def __call__(self, module, args, kwargs, output):
        # output: DecoderLayer输出 tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        # ★ D10微扰: 在hidden上加一个小的五动调制信号
        # 用C5周期性的缩放因子调制hidden的幅度
        # 这是"管道壁呼吸"——管道本身不动, 但管壁在按五动节奏呼吸
        
        # 五动呼吸因子
        names = ['认','遇','落','裂','余']
        k = self.k_phase
        
        # 基础呼吸: alpha随五动相位变化
        breath = self.alpha * self.strength
        
        # Z₂翻转: 偶数组正, 奇数组负
        z2_sign = 1.0 if self.g == 0 else -1.0
        
        # C5温度调制
        temp = C5_TEMPS[k]
        temp_factor = 1.0 + self.strength * (temp - 1.0)
        
        # 组合: hidden = hidden * (1 + breath * z2_sign / temp_factor)
        # 这个微扰让每层的输出幅度按五动节奏轻微波动
        modulation = 1.0 + breath * z2_sign / temp_factor * 0.1
        hidden = hidden * modulation
        
        if rest is not None:
            return (hidden,) + rest
        return hidden

# --- 方案升级: 用更细粒度的hook ---
# 上面那个hook太粗了(只改hidden幅度), 
# 我们需要在attention级别做C5耦合和RPB
# 但又不能替换forward...

# ★★★ 最终方案: 分两步 ★★★
# Step A: 在self_attn上注册hook, 在attn输出后加C5残差
# Step B: 在mlp上注册hook, 在mlp输出后加五动FFN残差
# DecoderLayer的forward完全不动!

# --- Step A: Attention hook ---
class D10AttnHook:
    """在attention输出后叠加C5残差
    计算: q的C5混合 → 重新算attention → 残差叠加
    """
    def __init__(self, layer_idx, c5_coupling, num_heads, head_dim, 
                 num_kv_heads, num_kv_groups, q_proj, k_proj, v_proj, o_proj, strength):
        self.layer_idx = layer_idx
        self.c5_coupling = c5_coupling
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.strength = strength
        
        self.k_phase = layer_idx % 5
        self.g = layer_idx // 5 % 2
        k = layer_idx % 5
        self.alpha = PHI_POWERS[k] / math.sqrt(layer_idx + 2)
    
    def compute_c5_residual(self, hidden_states, attention_mask=None):
        """用C5耦合Q, 重新算一次attention, 返回残差"""
        bsz, q_len, _ = hidden_states.size()
        nh = self.num_heads
        nkh = self.num_kv_heads
        hd = self.head_dim
        nkg = self.num_kv_groups
        s = self.strength
        
        # Q with C5 coupling (微扰)
        q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
        coupling = self.c5_coupling.to(q.device, q.dtype)
        q_c5 = torch.einsum('ij,bjsd->bisd', coupling, q)
        q = q + s * (q_c5 - q)  # 微扰叠加
        
        # K, V 不动
        k = self.k_proj(hidden_states).view(bsz, q_len, nkh, hd).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, nkh, hd).transpose(1, 2)
        
        # ★ 不做RoPE! 保持原始(因为hook拿不到position_embeddings)
        # 这意味着C5残差只反映head混合效果, 不含位置信息
        # 这是一个合理的近似, 因为C5耦合本身是位置无关的
        
        # GQA expand
        if nkg > 1:
            k = k.unsqueeze(2).expand(-1, -1, nkg, -1, -1).reshape(bsz, nh, -1, hd)
            v = v.unsqueeze(2).expand(-1, -1, nkg, -1, -1).reshape(bsz, nh, -1, hd)
        
        # Attention (无RoPE, 所以只反映C5耦合效果)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        
        # C5 RPB
        slopes = C5_RPB_SLOPES[:]
        if self.g == 1:
            slopes = [-sl for sl in slopes]
        slope = slopes[self.k_phase] * s
        if slope != 0.0:
            seq = attn.shape[-1]
            pos = torch.arange(seq, device=attn.device, dtype=attn.dtype)
            dist = (pos[:, None] - pos[None, :]).abs().float()
            attn = attn + slope * dist / seq
        
        # C5温度
        temp = 1.0 + s * (C5_TEMPS[self.k_phase] - 1.0)
        attn = attn / temp
        
        if attention_mask is not None:
            # attention_mask可能很复杂, 安全处理
            try:
                attn = attn + attention_mask
            except:
                pass
        
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, nh * hd)
        out = self.o_proj(out)
        
        # 残差 = alpha * strength * (C5_attn_output)
        return self.alpha * s * out
    
    def __call__(self, module, args, kwargs, output):
        # output: (attn_output, attn_weights) or (attn_output,)
        if isinstance(output, tuple):
            attn_out = output[0]
            rest = output[1:]
        else:
            attn_out = output
            rest = None
        
        # 拿input
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        if hidden_states is None:
            return output
        
        attention_mask = kwargs.get('attention_mask', None)
        
        # 计算C5残差
        with torch.no_grad():
            residual = self.compute_c5_residual(hidden_states, attention_mask)
        
        # 叠加残差
        attn_out = attn_out + residual
        
        if rest is not None:
            return (attn_out,) + rest
        return attn_out

# --- Step B: FFN hook ---
class D10FFNHook:
    """在mlp输出后叠加C5-FFN残差"""
    def __init__(self, layer_idx, W_organ, W_out, c5_adj, d_organ, strength):
        self.layer_idx = layer_idx
        self.W_organ = W_organ
        self.W_out = W_out
        self.c5_adj = c5_adj
        self.d_organ = d_organ
        self.strength = strength
        
        k = layer_idx % 5
        self.alpha = PHI_POWERS[k] / math.sqrt(layer_idx + 2) * strength
    
    def __call__(self, module, args, kwargs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        x = args[0] if args else None
        if x is None:
            return output
        
        # 五动FFN残差
        organs = []
        for j, w in enumerate(self.W_organ):
            h = w(x)
            act = ORGAN_CONFIG[j][0]
            if act == 'silu': h = F.silu(h)
            elif act == 'tanh': h = torch.tanh(h)
            organs.append(h)
        organs = torch.stack(organs, dim=-1)
        organs = torch.einsum('ij,bsdj->bsdi', 
                             self.c5_adj.to(organs.device, organs.dtype), organs)
        d10_residual = self.W_out(organs.reshape(*x.shape[:-1], 5*self.d_organ))
        d10_residual = self.alpha * d10_residual
        
        hidden = hidden + d10_residual
        
        if rest is not None:
            return (hidden,) + rest
        return hidden

# 注册所有hook
attn_hooks = []
ffn_hooks = []

for i, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    mlp = layer.mlp
    
    # Attention hook
    attn_hook = D10AttnHook(
        layer_idx=i,
        c5_coupling=c5_coupling.clone(),
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_kv_groups=num_kv_groups,
        q_proj=attn.q_proj,
        k_proj=attn.k_proj,
        v_proj=attn.v_proj,
        o_proj=attn.o_proj,
        strength=D10_STRENGTH
    )
    h = attn.register_forward_hook(attn_hook, with_kwargs=True)
    attn_hooks.append(h)
    
    # FFN hook
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
    
    ffn_hook = D10FFNHook(
        layer_idx=i,
        W_organ=W_organ,
        W_out=W_out,
        c5_adj=_C5_ADJ.clone(),
        d_organ=d_organ,
        strength=D10_STRENGTH
    )
    h = mlp.register_forward_hook(ffn_hook, with_kwargs=True)
    ffn_hooks.append(h)

print(f"  [2] Attention hook: C5耦合微扰+RPB+温度 (s={D10_STRENGTH}) ✓")
print(f"  [3] FFN hook: 五动器官+耦合 (s={D10_STRENGTH}) ✓")
print(f"  ★ 原始forward/RoPE/KV cache全不动!")
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
    
    # ★ 差异分析
    if d10_text != baseline_text:
        # 找到第一个不同的位置
        for ci in range(min(len(baseline_text), len(d10_text))):
            if baseline_text[ci] != d10_text[ci]:
                print(f"  首个差异位置: 字符{ci}")
                print(f"    基线: ...{baseline_text[max(0,ci-10):ci+20]}...")
                print(f"    D10:  ...{d10_text[max(0,ci-10):ci+20]}...")
                break
        else:
            print(f"  前缀相同, 长度不同")
    else:
        print(f"  ★ 输出完全相同! D10微扰未产生可观测差异")
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

# 清理hook
for h in attn_hooks:
    h.remove()
for h in ffn_hooks:
    h.remove()

print(f"\n{'='*60}")
print("D10五动管道架构改造 v11 完成!")
print(f"  d10_strength = {D10_STRENGTH}")
print(f"  ★ 保持原始RoPE!")
print(f"  ★ 纯hook实现, 不替换任何forward!")
print(f"{'='*60}")

if d10_text is not None:
    save_path = "./qwen2.5-1.5b-d10-v11"
    print(f"\n保存到 {save_path}...")
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("✅ 保存完成!")
    except Exception as e:
        print(f"  ⚠ 保存失败: {e}")
