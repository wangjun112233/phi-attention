"""
D10五动管道架构改造脚本 v15
基于v14结论: STRENGTH=0.05时norm安全(1.38x)且KL=0.007718
但top-1 token未变 — 微偏移需要更长生成才能累积成分叉

v15: 用STRENGTH=0.05生成长文本(200 tokens), 让微偏移累积
     同时测试多个prompt, 有些prompt的top-1/top-2差距小, 更容易翻
     加一组STRENGTH=0.08作为中间值测试

用法：python d10_patch_qwen_v15.py
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

PHI = (1 + 5**0.5) / 2
PHI_POWERS = {0: 1.0, 1: 1/PHI, 2: 1/PHI**2, 3: PHI, 4: 1/PHI**3}
C5_NAMES = ['认','遇','落','裂','余']

ALPHA_MODE = "gentle"

# 多个测试prompt — 有些top1/top2接近更容易翻
TEST_PROMPTS = [
    "The fundamental nature of reality is",
    "Consciousness arises from",
    "The relationship between order and chaos is",
    "In physics, the most fundamental principle is",
    "The meaning of existence is",
]

print("=" * 60)
print("D10五动管道 v15 — 长文本累积测试")
print("=" * 60)

# 加载模型
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

def compute_layer_alphas(num_layers, strength):
    alphas = []
    for i in range(num_layers):
        k = i % 5
        g = i // 5 % 2
        base_alpha = PHI_POWERS[k] / math.sqrt(i + 2)
        alpha = 1.0 + strength * (base_alpha - 1.0)
        if g == 1:
            alpha = 2.0 - alpha
        alphas.append(alpha)
    return alphas

def apply_d10_hooks(model, layer_alphas):
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

# 对每个STRENGTH值测试
for strength in [0.05, 0.08]:
    print(f"\n{'='*60}")
    print(f"D10_STRENGTH = {strength}")
    print(f"{'='*60}")
    
    layer_alphas = compute_layer_alphas(num_layers, strength)
    
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 基线generate
        with torch.no_grad():
            baseline_gen = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
        
        # D10 generate
        hooks = apply_d10_hooks(model, layer_alphas)
        try:
            with torch.no_grad():
                d10_gen = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            d10_text = tokenizer.decode(d10_gen[0], skip_special_tokens=True)
        except Exception as e:
            print(f"  ⚠ '{prompt[:30]}...' D10生成失败: {e}")
            remove_hooks(hooks)
            continue
        remove_hooks(hooks)
        
        # 比较
        same = baseline_text == d10_text
        if same:
            print(f"\n  [{prompt[:35]}]")
            print(f"  基线=D10 (完全相同)")
        else:
            # 找第一个差异
            diff_pos = -1
            for ci in range(min(len(baseline_text), len(d10_text))):
                if baseline_text[ci] != d10_text[ci]:
                    diff_pos = ci
                    break
            
            print(f"\n  [{prompt[:35]}]")
            print(f"  ★ 差异@字符{diff_pos}!")
            ctx = 30
            if diff_pos >= 0:
                print(f"    基线: ...{baseline_text[max(0,diff_pos-ctx):diff_pos+ctx]}...")
                print(f"    D10:  ...{d10_text[max(0,diff_pos-ctx):diff_pos+ctx]}...")
            
            # 显示完整输出对比
            print(f"    ─── 基线完整 ───")
            print(f"    {baseline_text[:200]}")
            print(f"    ─── D10完整 ───")
            print(f"    {d10_text[:200]}")
        
        # 单步forward logits对比
        with torch.no_grad():
            base_out = model(**inputs)
            base_logits = base_out.logits[0, -1]
        
        hooks = apply_d10_hooks(model, layer_alphas)
        with torch.no_grad():
            d10_out = model(**inputs)
            d10_logits = d10_out.logits[0, -1]
        remove_hooks(hooks)
        
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(d10_logits, dim=-1),
            torch.softmax(base_logits, dim=-1),
            reduction='sum'
        ).item()
        
        # top-3对比
        base_top3 = torch.topk(base_logits, 3)
        d10_top3 = torch.topk(d10_logits, 3)
        
        print(f"    KL={kl:.6f}")
        for j in range(3):
            bt = tokenizer.decode([base_top3.indices[j].item()]).strip() or f"tok{base_top3.indices[j].item()}"
            dt = tokenizer.decode([d10_top3.indices[j].item()]).strip() or f"tok{d10_top3.indices[j].item()}"
            print(f"    top{j+1}: 基线'{bt}'({base_top3.values[j]:.2f}) D10'{dt}'({d10_top3.values[j]:.2f})")

# 最终总结
print(f"\n{'='*60}")
print("v15 总结:")
print("  如果某个prompt出现了文本差异, 说明φ-Residual的微偏移")
print("  在长文本生成中确实能累积成可观测的语义分叉")
print("  如果全部相同, 需要更大的STRENGTH或换方案")
print(f"{'='*60}")
