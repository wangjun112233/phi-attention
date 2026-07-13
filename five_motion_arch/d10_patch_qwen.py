"""
D10五动管道架构改造脚本
基于ARCHITECTURE.md v5，对Qwen2.5-1.5B进行D10架构改造
改造内容：φ²-RoPE + C5-Attention(耦合+RPB) + 五动FFN + φ-Residual + Z₂门控

用法：python d10_patch_qwen.py
首次运行会自动从HuggingFace下载Qwen2.5-1.5B模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============================================================
# 五动常量
# ============================================================
PHI = (1 + 5**0.5) / 2  # 黄金比例 ≈ 1.618

# C5邻接矩阵
_C5_ADJ = torch.tensor([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
], dtype=torch.float32)

# C5耦合矩阵 A = I + cos72° × C5邻接
_COS72 = math.cos(math.radians(72))
_A5 = torch.eye(5) + _COS72 * _C5_ADJ

# 五动RPB斜率：认=-1(近距) 遇=0 落=-0.5 裂=+1(远距) 余=0
C5_RPB_SLOPES = [-1.0, 0.0, -0.5, 1.0, 0.0]

# 五动温度：认=0.8 遇=1.0 落=1.2 裂=φ 余=1.0
C5_TEMPS = [0.8, 1.0, 1.2, PHI, 1.0]

# 五动残差系数：认=1.0 遇=φ⁻¹ 落=φ⁻² 裂=φ 余=φ⁻³
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}

# 五动FFN激活和偏置
ORGAN_CONFIG = [
    ('silu', -0.5),   # 认：SiLU + 关门
    ('tanh', 0.0),    # 遇：tanh + 中性
    ('linear', 0.0),  # 落：linear + 无偏
    ('silu', +0.5),   # 裂：SiLU + 开门
    ('identity', 0.0) # 余：identity + 直过
]

print("=" * 60)
print("D10五动管道架构改造")
print("=" * 60)

# ============================================================
# Step 1: 加载模型
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
print(f"\n[1/4] 加载模型: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # CPU用float32
    device_map="cpu"
)
model.eval()

print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"  层数: {config.num_hidden_layers}")
print(f"  heads: {config.num_attention_heads}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  intermediate_size: {config.intermediate_size}")

# ============================================================
# Step 2: 备份原始输出（对比基线）
# ============================================================
print("\n[2/4] 生成基线输出...")

test_input = "The fundamental nature of reality is"
inputs = tokenizer(test_input, return_tensors="pt")

with torch.no_grad():
    baseline_output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0
    )
baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
print(f"  基线: {baseline_text[:100]}...")

# ============================================================
# Step 3: D10架构改造
# ============================================================
print("\n[3/4] 执行D10架构改造...")

# --- 改动1：rope_theta = φ² ≈ 2.618 ---
old_theta = config.rope_theta
config.rope_theta = PHI ** 2  # ≈ 2.618
print(f"  [改动1] RoPE底数: {old_theta} → {config.rope_theta:.4f}")

# --- 改动2+3：Attention改造 ---
num_heads = config.num_attention_heads
head_dim = config.hidden_size // num_heads

# C5 coupling矩阵（按head数分块）
n_groups = num_heads // 5
remainder = num_heads % 5
blocks = [_A5.clone()] * n_groups
if remainder > 0:
    blocks.append(torch.eye(remainder, dtype=torch.float32))
c5_coupling = torch.block_diag(*blocks)

print(f"  [改动2] C5 head耦合矩阵: {c5_coupling.shape}")

# Patch所有attention层
for layer_idx, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    
    # 存储C5参数
    attn.c5_coupling = c5_coupling.clone()
    attn.c5_rpb_slopes = C5_RPB_SLOPES
    attn.c5_temps = C5_TEMPS
    attn.layer_idx = layer_idx
    
    # D10 Z₂门控：每5层翻转
    g = layer_idx // 5 % 2
    attn.z2_gate = g
    
    # Monkey-patch forward方法
    original_forward = attn.forward
    
    def make_patched_forward(orig_fwd, attn_module, lidx):
        def patched_forward(
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
            # 先调用原始forward获取QKV
            # 我们在QK计算后、softmax前插入C5机制
            # 用hook方式实现
            
            result = orig_fwd(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=True,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )
            return result
        
        return patched_forward
    
    # 实际上直接patch Qwen2Attention.forward更可靠
    # 我们用register_forward_hook来修改attention weights

print(f"  [改动3] C5-RPB斜率: {C5_RPB_SLOPES}")
print(f"  [改动3] C5温度: {[f'{t:.2f}' if t != PHI else f'{t:.4f}(φ)' for t in C5_TEMPS]}")

# --- 改动4：FFN改造 ---
d_ff = config.intermediate_size
d_organ = d_ff // 5
d_model = config.hidden_size

print(f"  [改动4] 五动FFN: d_ff={d_ff} → 5×d_organ={5*d_organ}")

for layer_idx, layer in enumerate(model.model.layers):
    mlp = layer.mlp
    
    # 创建5个器官投影
    mlp.W_organ = nn.ModuleList([
        nn.Linear(d_model, d_organ, bias=True)
        for _ in range(5)
    ])
    
    # 设置器官偏置
    for i, (_, bias_val) in enumerate(ORGAN_CONFIG):
        nn.init.constant_(mlp.W_organ[i].bias, bias_val)
    
    # 用原gate_up_proj的权重初始化（均分）
    with torch.no_grad():
        orig_weight = mlp.gate_proj.weight.data  # [d_ff, d_model]
        orig_bias = mlp.gate_proj.bias.data if mlp.gate_proj.bias is not None else None
        for i in range(5):
            start = i * d_organ
            end = start + d_organ
            mlp.W_organ[i].weight.data.copy_(orig_weight[start:end, :])
            if orig_bias is not None:
                mlp.W_organ[i].bias.data.copy_(orig_bias[start:end])
                # 覆盖为五动偏置
                mlp.W_organ[i].bias.data.fill_(ORGAN_CONFIG[i][1])
    
    # C5邻接buffer
    mlp.register_buffer('c5_adj', _C5_ADJ.clone())
    
    # 输出投影
    mlp.W_out = nn.Linear(5 * d_organ, d_model, bias=True)
    with torch.no_grad():
        # 用原down_proj权重初始化
        orig_down_weight = mlp.down_proj.weight.data  # [d_model, d_ff]
        mlp.W_out.weight.data.copy_(orig_down_weight[:, :5*d_organ])
        if mlp.down_proj.bias is not None:
            mlp.W_out.bias.data.copy_(mlp.down_proj.bias.data)
    
    # 替换forward
    k_phase = layer_idx % 5
    
    def make_ffn_forward(mlp_module, phase_k):
        def ffn_forward(x):
            organs = []
            for i, w in enumerate(mlp_module.W_organ):
                h = w(x)
                act_name = ORGAN_CONFIG[i][0]
                if act_name == 'silu':
                    h = F.silu(h)
                elif act_name == 'tanh':
                    h = torch.tanh(h)
                elif act_name == 'identity':
                    pass  # 直过
                # linear: 什么都不做
                organs.append(h)
            
            # Stack: [batch, seq, d_organ, 5]
            organs = torch.stack(organs, dim=-1)
            # C5耦合: [5,5] × [batch, seq, d_organ, 5] → [batch, seq, d_organ, 5]
            organs = torch.einsum('ij,bsdj->bsdi', mlp_module.c5_adj.to(organs.device, organs.dtype), organs)
            # 拼接回d_ff
            concat = organs.reshape(*x.shape[:-1], 5 * d_organ)
            return mlp_module.W_out(concat)
        return ffn_forward
    
    mlp.forward = make_ffn_forward(mlp, k_phase)

# --- 改动5：残差连接 ---
print(f"  [改动5] φ-Residual: φ呼吸节奏")

for layer_idx, layer in enumerate(model.model.layers):
    k = layer_idx % 5
    alpha = PHI_POWERS[k] / math.sqrt(layer_idx + 2)
    layer._phi_alpha = alpha
    layer._phi_k = k
    
    # Z₂门控翻转
    g = layer_idx // 5 % 2
    layer._z2_gate = g
    
    original_layer_forward = layer.forward
    
    def make_layer_forward(orig_lf, phi_alpha, phi_k, z2_gate, lidx):
        def patched_forward(
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
            residual = hidden_states
            
            # Self Attention
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Attention with C5 modifications
            attn_outputs = layer.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=True,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = attn_outputs[0]
            
            # φ-Residual
            hidden_states = residual + phi_alpha * hidden_states
            
            # FFN
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            
            # φ-Residual
            hidden_states = residual + phi_alpha * hidden_states
            
            outputs = (hidden_states,)
            if use_cache:
                outputs += (attn_outputs[1] if len(attn_outputs) > 1 else None,)
            if output_attentions:
                outputs += (attn_outputs[2] if len(attn_outputs) > 2 else None,)
            
            return outputs
        return patched_forward
    
    layer.forward = make_layer_forward(original_layer_forward, alpha, k, g, layer_idx)

# --- Attention Hook：C5耦合 + RPB + 温度 ---
print(f"  [改动6] Attention Hook: C5耦合+RPB+温度+Z₂")

def create_attn_hook(c5_coupling_mat, layer_idx_val):
    k = layer_idx_val % 5
    g = layer_idx_val // 5 % 2
    
    slopes = C5_RPB_SLOPES.copy()
    if g == 1:  # Z₂翻转
        slopes = [-s for s in slopes]
    
    slope = slopes[k]
    temp = C5_TEMPS[k]
    
    def hook_fn(module, input, output):
        # output是一个tuple，我们需要修改attention weights
        # 但Qwen2用的是sdpa，不直接暴露weights
        # 所以我们改用eager attention模式
        return output
    
    return hook_fn

# 由于Qwen2默认用SDPA（不暴露attention weights），
# 我们需要强制切换到eager模式并直接patch attention计算
# 最可靠的方式：重写Qwen2Attention.forward

def patched_qwen2_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
):
    bsz, q_len, _ = hidden_states.size()
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # RoPE (用新的rope_theta=φ²)
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # GQA repeat
    if self.num_key_value_groups > 1:
        key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
    
    # QK^T / sqrt(d)
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
    
    # === C5五动改造 ===
    layer_idx = self.layer_idx
    k = layer_idx % 5
    g = layer_idx // 5 % 2
    
    # C5 head耦合
    coupling = self.c5_coupling.to(attn_weights.device, attn_weights.dtype)
    attn_weights = torch.matmul(coupling, attn_weights)
    
    # C5 RPB
    slopes = C5_RPB_SLOPES.copy()
    if g == 1:  # Z₂翻转
        slopes = [-s for s in slopes]
    slope = slopes[k]
    
    if slope != 0.0:
        seq_len = attn_weights.shape[-1]
        pos = torch.arange(seq_len, device=attn_weights.device, dtype=attn_weights.dtype)
        dist = (pos[:, None] - pos[None, :]).abs().float()
        rpb = slope * dist / seq_len
        attn_weights = attn_weights + rpb[None, None, :, :]
    
    # C5温度
    temp = C5_TEMPS[k]
    attn_weights = attn_weights / temp
    
    # Attention mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)
    
    outputs = (attn_output,)
    if output_attentions:
        outputs += (attn_weights,)
    
    return outputs


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Patch所有attention层的forward
for layer_idx, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    attn.layer_idx = layer_idx
    attn.forward = patched_qwen2_attention_forward.__get__(attn, type(attn))

# 设置config为eager attention
config._attn_implementation = "eager"
model.config._attn_implementation = "eager"

# 重建RoPE with φ² theta
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
new_rotary = Qwen2RotaryEmbedding(config=config, device=model.device)
for layer in model.model.layers:
    layer.self_attn.rotary_emb = new_rotary

print("\n  ✅ D10架构改造完成！")
print(f"  - φ²-RoPE: θ = {config.rope_theta:.4f}")
print(f"  - C5-Attention: 耦合+RPB+温度")
print(f"  - 五动FFN: 5器官+耦合")
print(f"  - φ-Residual: 呼吸节奏")
print(f"  - Z₂门控: 每5层翻转")

# ============================================================
# Step 4: D10改造后推理测试
# ============================================================
print("\n[4/4] D10改造后推理测试...")

with torch.no_grad():
    d10_output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0
    )
d10_text = tokenizer.decode(d10_output[0], skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"输入: {test_input}")
print(f"{'='*60}")
print(f"基线输出: {baseline_text[:150]}")
print(f"{'─'*60}")
print(f"D10输出:  {d10_text[:150]}")
print(f"{'='*60}")

# ============================================================
# Step 5: 五动诊断 - 检查各层行为差异
# ============================================================
print("\n[5/5] 五动层诊断...")

diag_input = tokenizer("The universe is", return_tensors="pt")

with torch.no_grad():
    # Hook每层的residual norm
    layer_norms = []
    hooks = []
    
    def norm_hook(name):
        def fn(module, input, output):
            layer_norms.append((name, output[0].norm().item()))
        return fn
    
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(norm_hook(f"L{i}"))
        hooks.append(h)
    
    _ = model.generate(**diag_input, max_new_tokens=10, do_sample=False)
    
    for h in hooks:
        h.remove()

print("\n  层级激活范数 (五动周期):")
for i, (name, norm_val) in enumerate(layer_norms):
    k = i % 5
    phase_name = ['认', '遇', '落', '裂', '余'][k]
    g = i // 5 % 2
    z2 = '⇄' if g == 1 else '→'
    print(f"  {name} [{phase_name}] {z2} norm={norm_val:.4f}")

print("\n" + "=" * 60)
print("D10五动管道架构改造完成！")
print("=" * 60)

# 保存改造后的模型
save_path = "./qwen2.5-1.5b-d10"
print(f"\n保存模型到 {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ 保存完成！")
