#!/usr/bin/env python3
"""
C5-RPB Attention — 真实Qwen2.5-1.5B验证
=========================================
在训练过的模型上验证C5-RPB:
- 训练过的模型head已分化 → C5-RPB有东西可调
- 对比: 标准vs C5-RPB vs C5-RPB+Z2
- 测量: 注意力权重的C5循环结构 + Z2塌缩偏移

用法:
  python c5_rpb_qwen_verify.py [--model_path PATH] [--rpb_amp FLOAT] [--device cpu|cuda]

兼容: transformers 5.13.1, Qwen2.5-1.5B/3B
"""

import argparse
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# C5-RPB生成
# ============================================================================

def make_c5_rpb_tensor(n_heads, seq_len, amplitude=1.0, phi_shift=0.0, device='cpu', dtype=torch.float32):
    """生成C5结构的相对位置偏置 (PyTorch tensor)
    
    B[h, i, j] = A * cos(2π(h%5)/5 + φ_shift + π*(i-j)/seq_len)
    """
    B = torch.zeros(n_heads, seq_len, seq_len, device=device, dtype=dtype)
    for h in range(n_heads):
        phase_h = 2 * math.pi * (h % 5) / 5 + phi_shift
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos = (i - j) / max(seq_len, 1)
                phase = phase_h + math.pi * rel_pos
                B[h, i, j] = amplitude * math.cos(phase)
    return B

# ============================================================================
# 5种motion的测试prompt
# ============================================================================

PROMPTS = {
    "认": "Analyze the mathematical structure of prime numbers and explain why they form the foundation of number theory.",
    "遇": "What if consciousness emerges not from complexity but from a specific geometric pattern in neural activity?",
    "落": "When a civilization collapses, what are the last things that disappear and why do they persist?",
    "裂": "The contradiction between determinism and free will: can both be true simultaneously in a quantum framework?",
    "余": "After all the known forces are accounted for, what remains unexplained about the structure of reality?",
}

MOTION_ORDER = ["认", "遇", "落", "裂", "余"]

# ============================================================================
# Attention Hook
# ============================================================================

class C5RPBHook:
    """Hook to inject C5-RPB into attention scores before softmax"""
    
    def __init__(self, rpb_bias=None):
        """
        rpb_bias: [n_heads, max_seq, max_seq] tensor, or None for standard (no injection)
        """
        self.rpb_bias = rpb_bias
        self.captured_attn_weights = {}
        self.layer_idx = 0
    
    def set_layer_idx(self, idx):
        self.layer_idx = idx
    
    def __call__(self, module, args, kwargs):
        """Pre-forward hook: modify attention computation"""
        # We can't easily modify scores with pre-hook since they're computed inside forward
        # Instead, we'll use a post-hook to capture, and monkey-patch forward
        pass

def patched_attention_forward(original_forward, rpb_bias_per_layer, capture_dict):
    """Create a patched forward that adds C5-RPB to attention scores
    
    rpb_bias_per_layer: dict {layer_idx: rpb_tensor} or None
    capture_dict: dict to store captured attention weights
    """
    
    def new_forward(self, *args, **kwargs):
        # Call original forward but with output_attentions=True
        # We need to intercept the attention weights
        
        # Get the layer index from the module
        layer_idx = getattr(self, '_c5_layer_idx', 0)
        rpb = rpb_bias_per_layer.get(layer_idx) if rpb_bias_per_layer else None
        
        # Use the standard eager attention path
        # This is a simplified approach: we'll capture and modify after the fact
        # For a proper implementation, we'd need to modify the attention computation directly
        
        result = original_forward(*args, **kwargs)
        
        return result
    
    return new_forward

# ============================================================================
# 更直接的方式: 替换attention forward
# ============================================================================

def inject_c5_rpb(model, rpb_bias_per_layer, capture_weights=False):
    """Inject C5-RPB by replacing attention forward methods
    
    This modifies the model in-place to add C5-RPB bias to attention scores.
    Also captures attention weights if requested.
    """
    captured = {} if capture_weights else None
    
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        rpb = rpb_bias_per_layer.get(layer_idx) if rpb_bias_per_layer else None
        
        # Store original forward
        if not hasattr(attn, '_original_forward'):
            attn._original_forward = attn.forward
        
        original_forward = attn._original_forward
        _rpb = rpb  # capture for closure
        _layer_idx = layer_idx
        _capture = capture_weights
        
        def make_forward(orig_fwd, rpb_bias, lidx, capt):
            def new_forward(self, *args, **kwargs):
                # Force output_attentions and use_eager
                kwargs['output_attentions'] = True
                
                # Get hidden_states
                hidden_states = args[0] if args else kwargs.get('hidden_states')
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]
                
                # Get Q, K, V projections (matching Qwen2/Llama attention)
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                
                # Reshape to multi-head
                n_heads = self.config.num_attention_heads
                n_kv_heads = self.config.num_key_value_heads
                head_dim = self.hidden_size // n_heads
                
                query_states = query_states.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                key_states = key_states.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                value_states = value_states.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                
                # GQA: repeat K, V if needed
                if n_kv_heads < n_heads:
                    n_rep = n_heads // n_kv_heads
                    key_states = key_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, n_heads, seq_len, head_dim)
                    value_states = value_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, n_heads, seq_len, head_dim)
                
                # Apply RoPE (simplified: use the model's rotary_emb)
                past_key_value = kwargs.get('past_key_value', None)
                position_ids = kwargs.get('position_ids', None)
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                
                # Get rotary embeddings
                rotary_emb = self.rotary_emb
                # transformers 5.13.1: rotary_emb returns (cos, sin) with position_ids
                cos, sin = rotary_emb(value_states, position_ids, seq_len=seq_len)
                
                # Apply rotary to Q and K
                def rotate_half(x):
                    x1 = x[..., :x.shape[-1]//2]
                    x2 = x[..., x.shape[-1]//2:]
                    return torch.cat((-x2, x1), dim=-1)
                
                query_states = query_states * cos + rotate_half(query_states) * sin
                key_states = key_states * cos + rotate_half(key_states) * sin
                
                # Compute attention scores
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                
                # ★★★ INJECT C5-RPB HERE ★★★
                if rpb_bias is not None:
                    # rpb_bias: [n_heads, max_seq, max_seq]
                    # We only use the first seq_len positions
                    rpb_slice = rpb_bias[:, :seq_len, :seq_len]
                    attn_weights = attn_weights + rpb_slice.unsqueeze(0)
                
                # Causal mask
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
                    diagonal=1
                )
                attn_weights = attn_weights + causal_mask
                
                # Softmax
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                # Capture if requested
                if capt is not None:
                    capt[lidx] = attn_weights.detach().cpu().float().numpy()
                
                # Apply attention to values
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
                
                # Output projection
                attn_output = self.o_proj(attn_output)
                
                return attn_output, attn_weights if kwargs.get('output_attentions', False) else (attn_output,)
            
            return new_forward
        
        attn.forward = make_forward(original_forward, _rpb, _layer_idx, captured)
    
    return captured

def restore_attention(model):
    """Restore original attention forward methods"""
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, '_original_forward'):
            attn.forward = attn._original_forward
            delattr(attn, '_original_forward')

# ============================================================================
# C5结构测量
# ============================================================================

C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ = [(0,2),(0,3),(1,3),(1,4),(2,4)]

def measure_attn_c5(attn_weights_dict, motion_names=MOTION_ORDER):
    """测量注意力权重中的C5循环结构
    
    attn_weights_dict: {layer_idx: ndarray [batch, n_heads, seq_len, seq_len]}
    返回: {layer: {k1_ratio, adj_sim, nonadj_sim, nearest_c5, sim_matrix}}
    """
    results = {}
    
    for layer_idx, weights in attn_weights_dict.items():
        # weights: [batch, n_heads, seq_len, seq_len]
        # 提取5种motion在每个head上的注意力模式
        # 取每个prompt在最后一个token位置对前面token的注意力分布
        n_heads = weights.shape[1]
        
        # 5种motion × n_heads 的激活矩阵
        motion_head_act = np.zeros((5, n_heads))
        
        for mi, motion in enumerate(motion_names):
            # 第mi个prompt的注意力
            w = weights[mi]  # [n_heads, seq_len, seq_len]
            # 每个head: 最后一个token对各token的注意力均值
            for h in range(n_heads):
                motion_head_act[mi, h] = w[h, -1, :].mean()
        
        # 测C5结构 (5 motion之间的相似度)
        norms = np.linalg.norm(motion_head_act, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = motion_head_act / norms
        sim = normalized @ normalized.T
        
        adj_sim = np.mean([sim[i,j] for i,j in C5_ADJACENT])
        nonadj_sim = np.mean([sim[i,j] for i,j in C5_NONADJ])
        circular_ratio = adj_sim / max(abs(nonadj_sim), 1e-10)
        
        nearest_c5 = 0
        for i in range(5):
            sims = sim[i].copy()
            sims[i] = -999
            nearest = np.argmax(sims)
            if nearest in [(i+1)%5, (i-1)%5]:
                nearest_c5 += 1
        
        # DFT k=1
        n = 5
        W_dft = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
        dft = W_dft @ motion_head_act
        freq_energy = np.array([np.mean(np.abs(dft[k])**2) for k in range(n)])
        total = freq_energy.sum()
        k1_ratio = (freq_energy[1] + freq_energy[4]) / max(total, 1e-10)
        
        results[layer_idx] = {
            'k1_ratio': float(k1_ratio),
            'adj_sim': float(adj_sim),
            'nonadj_sim': float(nonadj_sim),
            'circular_ratio': float(circular_ratio),
            'nearest_c5': nearest_c5,
            'sim_matrix': sim,
        }
    
    return results

def measure_head_c5_structure(attn_weights, motion_names=MOTION_ORDER):
    """测量5个head之间的C5结构 (head特异化维度)"""
    # attn_weights: [batch, n_heads, seq_len, seq_len]
    # 对每个motion, 看5个head的注意力分布差异
    n_heads = attn_weights.shape[1]
    
    # 选择5个head (0,3,6,9,12 或 0,4,8,12,15, 取决于head数)
    if n_heads >= 5:
        selected_heads = [i * n_heads // 5 for i in range(5)]
    else:
        selected_heads = list(range(n_heads))
    
    # 5 motion × 5 selected heads 激活矩阵
    act = np.zeros((5, len(selected_heads)))
    for mi in range(5):
        w = attn_weights[mi]  # [n_heads, seq, seq]
        for hi, h in enumerate(selected_heads):
            act[mi, hi] = w[h, -1, :].mean()
    
    # C5结构
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = act / norms
    sim = normalized @ normalized.T
    
    adj_sim = np.mean([sim[i,j] for i,j in C5_ADJACENT])
    nonadj_sim = np.mean([sim[i,j] for i,j in C5_NONADJ])
    
    return {'adj_sim': float(adj_sim), 'nonadj_sim': float(nonadj_sim), 'sim_matrix': sim}

# ============================================================================
# 主实验
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        default=r'C:\Users\WANGJUN\d10\ms_cache\models\Qwen-Qwen2.5-1.5B\snapshots\master',
                        help='本地模型路径')
    parser.add_argument('--rpb_amp', type=float, default=2.0,
                        help='C5-RPB幅度')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='生成token数')
    args = parser.parse_args()
    
    print("=" * 70)
    print("C5-RPB Attention — 真实Qwen2.5验证")
    print("=" * 70)
    print(f"\n模型: {args.model_path}")
    print(f"RPB幅度: {args.rpb_amp}")
    print(f"设备: {args.device}")
    
    # ===== 加载模型 =====
    print("\n[1] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",  # 强制eager以获取attention weights
    )
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads
    
    print(f"  层数: {n_layers}, Heads: {n_heads}, KV Heads: {n_kv_heads}")
    print(f"  Hidden: {hidden_size}, Head dim: {head_dim}")
    
    # ===== 准备输入 =====
    print("\n[2] 准备5种motion的输入...")
    encoded = {}
    for motion, prompt in PROMPTS.items():
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        encoded[motion] = inputs
        print(f"  {motion}: {prompt[:50]}... (seq_len={inputs['input_ids'].shape[1]})")
    
    max_seq = max(v['input_ids'].shape[1] for v in encoded.values())
    print(f"  最大序列长度: {max_seq}")
    
    # ===== 实验1: 标准Attention =====
    print("\n[3] 标准Attention (无RPB)...")
    std_attn_all = {}
    std_outputs = {}
    
    with torch.no_grad():
        for motion, inputs in encoded.items():
            output = model(**inputs, output_attentions=True)
            # output.attentions: tuple of (batch, n_heads, seq, seq), one per layer
            std_outputs[motion] = output
            for l, attn in enumerate(output.attentions):
                if l not in std_attn_all:
                    std_attn_all[l] = []
                std_attn_all[l].append(attn[0].cpu().float().numpy())  # [n_heads, seq, seq]
    
    # 合并成 [5_motion, n_heads, seq, seq]
    std_attn_stacked = {}
    for l in std_attn_all:
        std_attn_stacked[l] = np.stack(std_attn_all[l])  # [5, n_heads, seq, seq]
    
    # ===== 实验2: C5-RPB Attention =====
    print(f"\n[4] C5-RPB Attention (amp={args.rpb_amp})...")
    
    # 生成RPB (所有层共用)
    rpb_normal = make_c5_rpb_tensor(n_heads, max_seq + 10, amplitude=args.rpb_amp, device=args.device, dtype=torch.float32)
    rpb_per_layer = {l: rpb_normal for l in range(n_layers)}
    
    # 注入C5-RPB
    captured_c5 = inject_c5_rpb(model, rpb_per_layer, capture_weights=True)
    
    c5_attn_all = {}
    c5_outputs = {}
    
    with torch.no_grad():
        for motion, inputs in encoded.items():
            output = model(**inputs, output_attentions=False)
            c5_outputs[motion] = output
    
    # 从captured获取注意力权重
    c5_attn_stacked = {}
    for l in captured_c5:
        # captured_c5[l]: [batch=1, n_heads, seq, seq] per prompt
        # 我们需要收集5个motion的
        pass
    
    # 重新跑一遍, 这次每个motion单独捕获
    c5_attn_stacked = {}
    for l in range(n_layers):
        c5_attn_stacked[l] = []
    
    with torch.no_grad():
        for motion, inputs in encoded.items():
            captured_single = {}
            # Re-inject with capture for this prompt
            inject_c5_rpb(model, rpb_per_layer, capture_weights=False)
            
            # Use a simpler approach: manually compute with RPB
            hidden = model.model.embed_tokens(inputs['input_ids'])
            
            for l_idx, layer in enumerate(model.model.layers):
                # Forward through each layer manually
                rpb = rpb_per_layer.get(l_idx)
                
                # Self-attention
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                
                # Compute Q, K, V
                attn = layer.self_attn
                query = attn.q_proj(hidden)
                key = attn.k_proj(hidden)
                value = attn.v_proj(hidden)
                
                bsz, seq_len_q, _ = query.shape
                q = query.view(bsz, seq_len_q, n_heads, head_dim).transpose(1, 2)
                k = key.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)
                v = value.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)
                
                # GQA
                if n_kv_heads < n_heads:
                    n_rep = n_heads // n_kv_heads
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)
                    v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)
                
                # RoPE
                position_ids = torch.arange(seq_len_q, device=hidden.device).unsqueeze(0)
                cos_r, sin_r = attn.rotary_emb(v, position_ids, seq_len=seq_len_q)
                
                def rotate_half(x):
                    x1 = x[..., :x.shape[-1]//2]
                    x2 = x[..., x.shape[-1]//2:]
                    return torch.cat((-x2, x1), dim=-1)
                
                q = q * cos_r + rotate_half(q) * sin_r
                k = k * cos_r + rotate_half(k) * sin_r
                
                # Attention scores
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                
                # ★ C5-RPB ★
                if rpb is not None:
                    rpb_slice = rpb[:, :seq_len_q, :seq_len_q]
                    scores = scores + rpb_slice.unsqueeze(0)
                
                # Causal mask
                mask = torch.triu(torch.full((seq_len_q, seq_len_q), float('-inf'), device=hidden.device, dtype=hidden.dtype), diagonal=1)
                scores = scores + mask
                
                # Softmax
                attn_w = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                
                # Capture
                c5_attn_stacked[l_idx].append(attn_w[0].cpu().float().numpy())
                
                # Continue forward
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, seq_len_q, hidden_size)
                attn_out = attn.o_proj(attn_out)
                
                hidden = residual + attn_out
                
                # MLP
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden
            
            # Final norm
            hidden = model.model.norm(hidden)
            # We don't need the logits for this experiment
    
    # Stack attention weights
    for l in c5_attn_stacked:
        c5_attn_stacked[l] = np.stack(c5_attn_stacked[l])  # [5, n_heads, seq, seq]
    
    # Restore model
    restore_attention(model)
    
    # ===== 测量C5结构 =====
    print("\n[5] C5结构测量...")
    
    # 标准Attention
    std_c5 = measure_attn_c5(std_attn_stacked)
    
    # C5-RPB Attention
    c5_c5 = measure_attn_c5(c5_attn_stacked)
    
    # 输出对比
    print(f"\n  {'层':>4} | {'标准k1':>8} {'C5-RPB k1':>10} | {'标准adj':>8} {'C5-RPB adj':>10} | {'Δk1':>8}")
    print(f"  {'-'*65}")
    
    for l in [0, 1, 5, 10, 14, 20, n_layers-1]:
        if l not in std_c5 or l not in c5_c5:
            continue
        s = std_c5[l]
        c = c5_c5[l]
        dk1 = c['k1_ratio'] - s['k1_ratio']
        print(f"  {l:4d} | {s['k1_ratio']:8.4f} {c['k1_ratio']:10.4f} | "
              f"{s['adj_sim']:8.4f} {c['adj_sim']:10.4f} | {dk1:8.4f}")
    
    # ===== Head维度C5 =====
    print("\n[6] Head维度C5结构 (最终层)...")
    
    last_layer = n_layers - 1
    for tag, attn_data in [("标准", std_attn_stacked), ("C5-RPB", c5_attn_stacked)]:
        if last_layer not in attn_data:
            continue
        hc = measure_head_c5_structure(attn_data[last_layer])
        print(f"  {tag}: adj_sim={hc['adj_sim']:.4f}, nonadj={hc['nonadj_sim']:.4f}")
        sim = hc['sim_matrix']
        for i in range(5):
            row = ' '.join(f'{sim[i,j]:7.3f}' for j in range(5))
            print(f"    {MOTION_ORDER[i]}: {row}")
    
    # ===== Z₂塌缩实验 =====
    print(f"\n[7] Z₂塌缩 (层{last_layer//2}: C5-RPB → Z₂-RPB)...")
    
    collapse_layer = last_layer // 2
    rpb_z2 = make_c5_rpb_tensor(n_heads, max_seq + 10, amplitude=args.rpb_amp, 
                                  phi_shift=math.pi, device=args.device, dtype=torch.float32)
    
    # 层0~collapse: C5-RPB; 层collapse+1~end: Z₂-RPB
    rpb_collapse = {}
    for l in range(n_layers):
        if l <= collapse_layer:
            rpb_collapse[l] = rpb_normal
        else:
            rpb_collapse[l] = rpb_z2
    
    # 跑Z₂版本 (同样手动逐层)
    z2_attn_stacked = {}
    for l in range(n_layers):
        z2_attn_stacked[l] = []
    
    with torch.no_grad():
        for motion, inputs in encoded.items():
            hidden = model.model.embed_tokens(inputs['input_ids'])
            
            for l_idx, layer in enumerate(model.model.layers):
                rpb = rpb_collapse.get(l_idx)
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                
                attn = layer.self_attn
                query = attn.q_proj(hidden)
                key = attn.k_proj(hidden)
                value = attn.v_proj(hidden)
                
                bsz, seq_len_q, _ = query.shape
                q = query.view(bsz, seq_len_q, n_heads, head_dim).transpose(1, 2)
                k = key.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)
                v = value.view(bsz, seq_len_q, n_kv_heads, head_dim).transpose(1, 2)
                
                if n_kv_heads < n_heads:
                    n_rep = n_heads // n_kv_heads
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)
                    v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, n_heads, seq_len_q, head_dim)
                
                position_ids = torch.arange(seq_len_q, device=hidden.device).unsqueeze(0)
                cos_r, sin_r = attn.rotary_emb(v, position_ids, seq_len=seq_len_q)
                
                def rotate_half(x):
                    x1 = x[..., :x.shape[-1]//2]
                    x2 = x[..., x.shape[-1]//2:]
                    return torch.cat((-x2, x1), dim=-1)
                
                q = q * cos_r + rotate_half(q) * sin_r
                k = k * cos_r + rotate_half(k) * sin_r
                
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                
                if rpb is not None:
                    rpb_slice = rpb[:, :seq_len_q, :seq_len_q]
                    scores = scores + rpb_slice.unsqueeze(0)
                
                mask = torch.triu(torch.full((seq_len_q, seq_len_q), float('-inf'), device=hidden.device, dtype=hidden.dtype), diagonal=1)
                scores = scores + mask
                
                attn_w = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                
                z2_attn_stacked[l_idx].append(attn_w[0].cpu().float().numpy())
                
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, seq_len_q, hidden_size)
                attn_out = attn.o_proj(attn_out)
                
                hidden = residual + attn_out
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden
            
            hidden = model.model.norm(hidden)
    
    for l in z2_attn_stacked:
        z2_attn_stacked[l] = np.stack(z2_attn_stacked[l])
    
    # Z₂塌缩测量
    z2_c5 = measure_attn_c5(z2_attn_stacked)
    
    print(f"\n  塌缩层: {collapse_layer}")
    print(f"  {'层':>4} | {'正常k1':>8} {'塌缩k1':>8} | {'正常adj':>8} {'塌缩adj':>8} | {'相位偏移':>8}")
    print(f"  {'-'*60}")
    
    for l in [0, collapse_layer-1, collapse_layer, collapse_layer+1, last_layer]:
        if l not in c5_c5 or l not in z2_c5:
            continue
        cn = c5_c5[l]
        cz = z2_c5[l]
        shift = np.mean(np.abs(cn['sim_matrix'] - cz['sim_matrix']))
        tag = ""
        if l == collapse_layer: tag = " ← Z₂翻转"
        elif l == collapse_layer + 1: tag = " ← 翻转后1层"
        print(f"  {l:4d} | {cn['k1_ratio']:8.4f} {cz['k1_ratio']:8.4f} | "
              f"{cn['adj_sim']:8.4f} {cz['adj_sim']:8.4f} | {shift:8.4f}{tag}")
    
    # ===== 输出生成对比 =====
    print("\n[8] 生成输出对比 (第一个prompt)...")
    
    # 标准输出
    with torch.no_grad():
        std_text = tokenizer.decode(std_outputs[MOTION_ORDER[0]].logits.argmax(-1)[0], skip_special_tokens=True)
    
    # C5-RPB输出 (用hook方式生成)
    inject_c5_rpb(model, rpb_per_layer, capture_weights=False)
    with torch.no_grad():
        inputs_0 = encoded[MOTION_ORDER[0]]
        out_c5 = model.generate(**inputs_0, max_new_tokens=args.max_new_tokens, do_sample=False)
        c5_text = tokenizer.decode(out_c5[0], skip_special_tokens=True)
    
    # Z₂输出
    inject_c5_rpb(model, rpb_collapse, capture_weights=False)
    with torch.no_grad():
        out_z2 = model.generate(**inputs_0, max_new_tokens=args.max_new_tokens, do_sample=False)
        z2_text = tokenizer.decode(out_z2[0], skip_special_tokens=True)
    
    restore_attention(model)
    
    print(f"\n  标准输出: {std_text[:200]}")
    print(f"\n  C5-RPB:   {c5_text[:200]}")
    print(f"\n  Z₂塌缩:   {z2_text[:200]}")
    
    # ===== 核心结论 =====
    print("\n" + "=" * 70)
    print("核心结论")
    print("=" * 70)
    
    std_k1_final = std_c5.get(last_layer, {}).get('k1_ratio', 0)
    c5_k1_final = c5_c5.get(last_layer, {}).get('k1_ratio', 0)
    
    print(f"\n  标准Attention k1(终层): {std_k1_final:.4f}")
    print(f"  C5-RPB k1(终层):        {c5_k1_final:.4f}")
    print(f"  Δk1:                    {c5_k1_final - std_k1_final:.4f}")
    
    if c5_k1_final > std_k1_final + 0.05:
        print("  ✅✅ C5-RPB在训练模型上显著增强C5循环结构!")
    elif c5_k1_final > std_k1_final + 0.01:
        print("  ⚠️ C5-RPB有微弱增强效果")
    else:
        print("  ❌ C5-RPB未显著增强C5结构 (但注意力天然保持相位不衰减)")
    
    # Z₂判定
    if collapse_layer in c5_c5 and collapse_layer + 1 in c5_c5:
        shift = np.mean(np.abs(c5_c5[collapse_layer+1]['sim_matrix'] - z2_c5[collapse_layer+1]['sim_matrix']))
        print(f"\n  Z₂塌缩偏移(层{collapse_layer+1}): {shift:.4f}")
        if shift > 0.1:
            print("  ✅✅ Z₂否定在训练模型上产生显著塌缩!")
        elif shift > 0.03:
            print("  ⚠️ Z₂否定有可观测偏移")
        else:
            print("  ❌ Z₂否定偏移仍然太小")

if __name__ == "__main__":
    main()
