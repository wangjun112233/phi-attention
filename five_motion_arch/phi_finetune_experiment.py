#!/usr/bin/env python3
"""
φ-Residual Fine-tuning Experiment (路径2)
==========================================
架构改造(φ-Residual hook) + LoRA微调 → 验证C5相位结构是否涌现

与后挂hook的区别:
- 后挂hook: 权重冻结, 缩放是在已凝固的表示上推 → 推不动
- 微调: 权重(LoRA部分)在学习, 模型学会利用φ-Residual结构 → 可能涌现

步骤:
1. 加载Qwen2.5-1.5B
2. 应用φ-Residual hooks (与v18一致)
3. 添加LoRA适配器 (peft)
4. 在多样化数据上微调300步
5. 微调后运行相位实验
6. 对比微调前后的C5结构指标

硬件: GTX 960 4GB, 1.5B fp16 + LoRA + gradient_checkpointing
"""

import sys
import os
import time
import math
import json
import subprocess

# 自动安装依赖
def ensure_deps():
    for pkg in ["peft", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  安装 {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_deps()

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ============================================================================
# 配置
# ============================================================================

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
LORA_R = int(sys.argv[3]) if len(sys.argv) > 3 else 16
TRAIN_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
LR = 2e-4
BATCH_SIZE = 1
MAX_SEQ_LEN = 256

PHI = (1 + math.sqrt(5)) / 2

MOTION_NAMES = ["认(Recognize)", "遇(Encounter)", "落(Settle)", "裂(Split)", "余(Residue)"]
MOTION_KEYS = ["ren", "yu", "luo", "lie", "yuu"]
MOTION_LABELS = ["认", "遇", "落", "裂", "余"]

C5_ADJACENT = [(0,1),(1,2),(2,3),(3,4),(4,0)]
C5_NONADJ = [(0,2),(0,3),(1,3),(1,4),(2,4)]

# ============================================================================
# 训练数据 — 5动多样化语料
# ============================================================================

TRAIN_TEXTS = {
    "ren": [
        "The pattern in this sequence follows a clear geometric progression. Each term is twice the previous one, starting from 2.",
        "By examining the spectral lines, we can identify the element as sodium. The characteristic yellow doublet at 589nm is unmistakable.",
        "The fingerprint analysis reveals twelve matching points of identification between the latent print and the suspect's print.",
        "Classification of this organism places it in the order Lepidoptera based on the scaled wings and proboscis structure.",
        "The diagnostic criteria for this condition include persistent patterns of inattention and at least six specific symptoms.",
        "Recognizing the signature of a Bach fugue requires attention to the subject entry, countersubject, and episodic material.",
        "The crystal structure can be identified as face-centered cubic from the X-ray diffraction pattern showing peaks at specific angles.",
        "Pattern recognition in this data reveals a cyclical trend with period approximately equal to 11 years, matching solar activity.",
        "The key identifier of this chemical compound is its melting point of 114 degrees and its characteristic bitter almond odor.",
        "Through careful observation, we can classify the cloud formation as cumulonimbus, indicating an approaching thunderstorm.",
        "The sequence 1,1,2,3,5,8 is immediately recognizable as the Fibonacci sequence, where each term equals the sum of the two preceding terms.",
        "Identifying the author of this passage requires noting the distinctive use of stream of consciousness and long, flowing sentences.",
        "The disease can be identified by the characteristic rash that spreads from the extremities toward the trunk.",
        "We can recognize this mathematical structure as a group because it satisfies closure, associativity, identity, and inverse properties.",
        "The distinct pattern of erosion suggests water flow from the northwest, identifiable by the V-shaped valleys pointing downstream.",
        "Classification of this star as a red giant is based on its spectral type M and its position above the main sequence on the HR diagram.",
        "The identification of this artifact as Neolithic depends on the stone tool technology and the absence of metalworking evidence.",
        "Pattern matching reveals that this melody is a variation of a folk tune commonly found in Scandinavian musical traditions.",
        "The diagnostic algorithm identifies this as a Type 2 error based on the confidence interval containing the null hypothesis value.",
        "We can classify this reaction as an SN2 mechanism based on the inversion of stereochemistry and second-order kinetics.",
    ],
    "yu": [
        "Surprisingly, the same mathematical structure appears in both quantum mechanics and musical theory, connected through the concept of resonance.",
        "The unexpected connection between spider silk and artificial muscles emerged when researchers noticed similar stress-strain curves.",
        "This reminds me of a principle in thermodynamics that mirrors the social dynamics we observe in crowd behavior.",
        "By linking the behavior of slime molds to urban transportation networks, we discover efficient routing algorithms.",
        "The intersection of cryptography and genetics produces a new field: DNA-based data storage with remarkable density.",
        "It struck me that the fractal patterns in Romanesco broccoli mirror the self-similar structures found in financial market data.",
        "The surprising analogy between immune system recognition and machine learning classification reveals deep structural parallels.",
        "What connects the spiral of a nautilus shell to the orbit of planets is the same golden ratio that governs growth and balance.",
        "I noticed an unexpected parallel: the way neurons prune connections mirrors how social networks shed inactive members.",
        "The bridge between number theory and physics appeared when Riemann zeta zeros were found to match quantum energy levels.",
        "An accidental discovery linked coffee ring stains to nanotechnology: the same capillary flow that ruins prints can assemble particles.",
        "The connection between bird flocking and crystal formation lies in local interaction rules producing global order.",
        "It turns out that the mathematics of voting theory shares deep structure with the physics of phase transitions.",
        "The surprising link between origami and space exploration: foldable solar panels inspired by paper cranes power satellites.",
        "What ties together earthquake prediction and financial crashes is the mathematics of self-organized criticality.",
        "I was struck by how the algorithm for maze solving mirrors the way ants discover the shortest path to food.",
        "The unexpected relationship between music harmony and crystal symmetry: both are governed by the same group theory.",
        "Connecting the dots between coral reef bleaching and Arctic ice melt reveals a global ocean current feedback loop.",
        "The parallel between learning in neural networks and evolution in biology is deeper than mere analogy: both optimize through variation and selection.",
        "What the inventor noticed was that burrs sticking to his dog's fur suggested the hook-and-loop mechanism of Velcro.",
    ],
    "luo": [
        "After considering all options, the committee finally settled on the hybrid approach that balances cost and performance.",
        "The iterative optimization converges after approximately 47 iterations to a stable minimum with loss value 0.003.",
        "When the debate concludes, what remains is the fundamental principle that evidence must precede conclusion.",
        "The system reaches thermal equilibrium when all temperature gradients disappear and entropy is maximized.",
        "After months of negotiation, the two parties reached a final agreement that satisfies the core requirements of both sides.",
        "The mathematical proof resolves by showing that the assumption leads to a contradiction, settling the conjecture in the negative.",
        "Resolution of the paradox comes from distinguishing between the rate of change and the total accumulated change.",
        "The search algorithm terminates when the priority queue is empty, having found the shortest path through the graph.",
        "Ultimately, the theory rests on three axioms that together provide a complete and consistent foundation.",
        "The recursive computation bottoms out at the base case, and the final result propagates back through the call stack.",
        "After all the evidence is weighed, the conclusion is inescapable: the effect is real and statistically significant.",
        "The chaotic transient eventually settles into a periodic orbit, a common phenomenon in nonlinear dynamical systems.",
        "At the end of the analysis, we arrive at a simple and elegant result: the sum equals n squared.",
        "The committee's final recommendation synthesizes the competing views into a coherent policy framework.",
        "The convergent series sums to pi over four, a beautiful result that connects discrete summation to continuous geometry.",
        "After exploring numerous alternatives, we settled on the simplest explanation consistent with all observations.",
        "The system stabilizes when the feedback gain drops below unity, preventing oscillation and ensuring convergence.",
        "What finally resolves the tension between these two frameworks is recognizing they operate at different scales.",
        "The proof concludes by showing the invariant is maintained at each step, guaranteeing termination and correctness.",
        "With all variables accounted for, the model settles into a steady state that matches the observed data within experimental error.",
    ],
    "lie": [
        "The critical difference between these two theories is that one predicts particle behavior while the other predicts wave behavior.",
        "What separates successful innovation from mere novelty is whether it solves a real problem that people actually have.",
        "The boundary between classical and quantum behavior is not sharp but depends on the scale and isolation of the system.",
        "By rejecting the null hypothesis, we establish that the observed effect is unlikely to be due to random chance alone.",
        "The contradiction in this argument becomes clear when we examine the second premise: it is inconsistent with the conclusion.",
        "The crucial distinction between correlation and causation is that only the latter implies a mechanism of production.",
        "What divides the two approaches is their treatment of uncertainty: one embraces it, the other seeks to eliminate it.",
        "The negation of the universal statement produces an existential one: not all swans are white means some swans are not white.",
        "These two phenomena, though superficially similar, differ fundamentally in their underlying mechanisms.",
        "The contrast between the two datasets is stark: one shows exponential growth, the other logarithmic saturation.",
        "We must distinguish between the map and the territory: the model is not the reality it represents.",
        "The key contrast is between syntax (form) and semantics (meaning), which are independent dimensions of language.",
        "Rejecting the assumption of linearity opens up a much richer space of possible behaviors and dynamics.",
        "The split between the two schools of thought runs deep: one believes in emergence, the other in reductionism.",
        "This argument contains a hidden contradiction: if the premise were true, the conclusion could not follow from it.",
        "The distinction between necessary and sufficient conditions is crucial: having water is necessary but not sufficient for life.",
        "By separating the signal from the noise, we reveal that the underlying trend is actually declining, not growing.",
        "The fundamental disagreement is about ontology: one side claims only particles exist, the other insists fields are primary.",
        "The paradox resolves when we recognize the equivocation: the word 'nothing' means different things in the two contexts.",
        "What contradicts the standard model prediction is the measured value, which differs by more than five standard deviations.",
    ],
    "yuu": [
        "Even decades after the war, its impact on the region's economy and culture remains profoundly visible in everyday life.",
        "The residual magnetization of the iron core persists long after the external field is removed, a phenomenon known as hysteresis.",
        "What remains after the storm passes is not just damage but a fundamentally altered landscape that will shape future development.",
        "The echo of that initial discovery continues to influence the field, with over ten thousand citations to the original paper.",
        "Despite extensive purification, trace amounts of the catalyst persist in the final product at concentrations below one part per million.",
        "The lasting influence of ancient Greek philosophy on modern scientific thinking is evident in our continued use of deductive reasoning.",
        "Residual stress in the metal from the manufacturing process can affect performance years later under the right conditions.",
        "The persistence of cultural traditions across generations demonstrates the remarkable stability of transmitted information.",
        "Long after the volcanic eruption, the ash layer in the soil continues to affect agriculture and water quality in the region.",
        "The aftermath of the financial crisis included regulatory reforms that still shape banking practices today.",
        "Even after removing the treatment, the subjects showed sustained improvement, suggesting a lasting neural adaptation.",
        "The ghost town stands as a residual monument to the mining boom that once made it the largest city in the territory.",
        "What lingers in the atmosphere after the factory closed is a measurable concentration of particulates that slowly decreases each year.",
        "The enduring legacy of the treaty is not its specific provisions but the diplomatic framework it established for future negotiations.",
        "Traces of the ancient Roman road network persist in the modern street layout of many European cities.",
        "The residual heat from Earth's formation continues to drive geological activity billions of years later.",
        "Even after the theory was superseded, its mathematical framework persisted and was repurposed for entirely new applications.",
        "The long-term effects of the educational reform are still emerging, with the first cohort now entering the workforce.",
        "What remains of the glacier is a fraction of its former extent, yet it continues to feed the river that sustains the valley.",
        "The persistent challenge of antibiotic resistance is a legacy of decades of overuse that will take generations to address.",
    ],
}

# ============================================================================
# φ-Residual Hook
# ============================================================================

def compute_layer_alphas(nl, strength):
    alphas = []
    for i in range(nl):
        phase = i % 5
        if phase == 0:
            a = 1.0 - strength
        elif phase == 1:
            a = 1.0 + strength * PHI
        elif phase == 2:
            a = 1.0 - strength * 0.5
        elif phase == 3:
            a = 1.0 + strength
        else:
            a = 1.0 + strength * PHI**(-3)
        alphas.append(a)
    return alphas

def make_attn_scale(alpha):
    def hook(m, inp, out):
        if isinstance(out, tuple):
            scaled = alpha * out[0]
            return (scaled,) + out[1:]
        return alpha * out
    return hook

def make_mlp_scale(alpha):
    def hook(m, inp, out):
        return alpha * out
    return hook

def get_transformer_layers(model):
    """穿透PeftModel/LoRA包装层, 找到实际的transformer layers"""
    # 可能的路径: model.model.layers (原始)
    #            model.base_model.model.model.layers (LoRA包装后)
    candidates = [
        lambda m: m.model.layers,           # 原始 Qwen2ForCausalLM
        lambda m: m.base_model.model.model.layers,  # PeftModel包装后
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise AttributeError("无法找到transformer layers, 模型结构: " + str(type(model)))

def apply_phi_hooks(model, alphas):
    hooks = []
    layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        a = alphas[i]
        hooks.append(layer.self_attn.register_forward_hook(make_attn_scale(a)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_scale(a)))
    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

# ============================================================================
# Dataset
# ============================================================================

class MotionDataset(Dataset):
    def __init__(self, texts_dict, tokenizer, max_len=256):
        self.examples = []
        for mk, texts in texts_dict.items():
            for t in texts:
                enc = tokenizer(t, truncation=True, max_length=max_len,
                                padding="max_length", return_tensors="pt")
                self.examples.append({
                    "input_ids": enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                    "labels": enc["input_ids"][0].clone(),
                })
                # Mask padding in labels
                self.examples[-1]["labels"][enc["attention_mask"][0] == 0] = -100

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ============================================================================
# Phase Experiment (简化版, 训练后直接跑)
# ============================================================================

# 测试用prompt (和训练数据不同)
TEST_PROMPTS = {
    "ren": [
        "The diagnostic test identifies the condition based on",
        "Recognizing the pattern in this data requires",
        "Classification of this specimen depends on",
        "The key identifying feature is its",
        "This pattern is characteristic of",
    ],
    "yu": [
        "The surprising connection between these two fields is",
        "What links these seemingly unrelated phenomena is",
        "This unexpected parallel suggests",
        "The bridge between these two ideas reveals",
        "An unusual association connects",
    ],
    "luo": [
        "After analyzing all the evidence, we conclude that",
        "The optimization converges to a stable solution when",
        "The final resolution of this debate rests on",
        "When all factors are considered, the result is",
        "The system reaches its equilibrium at",
    ],
    "lie": [
        "The fundamental difference between these approaches is",
        "What contradicts this assumption is the fact that",
        "The critical distinction here is between",
        "Rejecting this hypothesis reveals that",
        "These two theories diverge on the question of",
    ],
    "yuu": [
        "The lasting impact of this discovery continues to",
        "Even years later, the residual effect persists as",
        "What remains after removing all other factors is",
        "The enduring influence of this event shapes",
        "Despite the passage of time, this principle still",
    ],
}

def make_layer_capture_hook(captures, layer_idx):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captures[layer_idx] = hs.detach().cpu().float()
    return hook

def apply_capture_hooks(model, captures):
    hooks = []
    layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_layer_capture_hook(captures, i)))
    return hooks

def circular_structure_test(vectors_5xh):
    norms = np.linalg.norm(vectors_5xh, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vectors_5xh / norms
    sim_matrix = normalized @ normalized.T
    adj_sims = [sim_matrix[i, j] for i, j in C5_ADJACENT]
    adj_mean = np.mean(adj_sims)
    nonadj_sims = [sim_matrix[i, j] for i, j in C5_NONADJ]
    nonadj_mean = np.mean(nonadj_sims)
    circular_ratio = adj_mean / max(nonadj_mean, 1e-10)

    nearest_correct = 0
    for i in range(5):
        sims = sim_matrix[i].copy()
        sims[i] = -999
        nearest = np.argmax(sims)
        if nearest in [(i+1)%5, (i-1)%5]:
            nearest_correct += 1

    return {
        'adj_sim': float(adj_mean),
        'nonadj_sim': float(nonadj_mean),
        'circular_ratio': float(circular_ratio),
        'nearest_c5': nearest_correct,
    }

def dft_cycle_test(vectors_5xh):
    n = 5
    W = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
    dft_coeffs = W @ vectors_5xh
    freq_energy = np.zeros(n)
    for k in range(n):
        freq_energy[k] = np.mean(np.abs(dft_coeffs[k])**2)
    total_energy = freq_energy.sum()
    if total_energy < 1e-20:
        return 0.0
    return float((freq_energy[1] + freq_energy[4]) / total_energy)

def run_phase_experiment(model, tokenizer, num_layers):
    """跑相位实验, 返回每个motion在每层的hidden state"""
    motion_hs = [[[] for _ in range(num_layers)] for _ in range(5)]

    for mi, mk in enumerate(MOTION_KEYS):
        for prompt in TEST_PROMPTS[mk]:
            inputs = tokenizer(prompt, return_tensors="pt")
            captures = {}
            hooks = apply_capture_hooks(model, captures)
            with torch.no_grad():
                _ = model(**inputs)
            remove_hooks(hooks)
            for li in range(num_layers):
                if li in captures:
                    hs = captures[li]
                    if hs.dim() == 2:
                        hs = hs.unsqueeze(0)
                    vec = hs[0, -1, :].numpy()
                    motion_hs[mi][li].append(vec)
            del captures

    # 平均
    hidden_size = model.config.hidden_size
    motion_avg = np.zeros((5, num_layers, hidden_size))
    for mi in range(5):
        for li in range(num_layers):
            if motion_hs[mi][li]:
                motion_avg[mi, li] = np.mean(motion_hs[mi][li], axis=0)

    return motion_avg

def evaluate_c5_structure(motion_avg, num_layers, tag=""):
    """评估C5相位结构, 返回关键指标"""
    pentagon_scores = []
    circular_ratios = []
    k1_ratios = []
    nearest_c5_counts = []

    for li in range(num_layers):
        vecs = motion_avg[:, li, :]
        if np.max(np.abs(vecs)) < 1e-10:
            continue
        circ = circular_structure_test(vecs)
        k1 = dft_cycle_test(vecs)

        # 简化五边形得分: 基于循环比和最近邻
        pent = min(1.0, circ['circular_ratio'] * circ['nearest_c5'] / 5.0)

        pentagon_scores.append(pent)
        circular_ratios.append(circ['circular_ratio'])
        k1_ratios.append(k1)
        nearest_c5_counts.append(circ['nearest_c5'])

    n = len(pentagon_scores)
    if n == 0:
        return None

    result = {
        'pentagon_mean': float(np.mean(pentagon_scores)),
        'circular_ratio_mean': float(np.mean(circular_ratios)),
        'circular_ratio_gt1': sum(1 for r in circular_ratios if r > 1.0),
        'k1_mean': float(np.mean(k1_ratios)),
        'k1_gt04': sum(1 for r in k1_ratios if r > 0.4),
        'nearest_c5_mean': float(np.mean(nearest_c5_counts)),
        'total_layers': n,
    }

    print(f"\n  {tag} C5结构指标:")
    print(f"    循环比>1.0: {result['circular_ratio_gt1']}/{n} 层")
    print(f"    循环比均值: {result['circular_ratio_mean']:.4f}")
    print(f"    DFT k=1>0.4: {result['k1_gt04']}/{n} 层")
    print(f"    DFT k=1均值: {result['k1_mean']:.4f}")
    print(f"    最近邻C5均值: {result['nearest_c5_mean']:.1f}/5")

    # 打印关键层相似度矩阵
    key_layers = [0, num_layers//3, 2*num_layers//3, num_layers-1]
    for li in key_layers:
        if li >= num_layers:
            continue
        vecs = motion_avg[:, li, :]
        if np.max(np.abs(vecs)) < 1e-10:
            continue
        circ = circular_structure_test(vecs)
        sim = np.array(circ['sim_matrix']) if 'sim_matrix' in circ else None
        if sim is not None:
            print(f"\n    Layer {li} 相似度矩阵:")
            for ri in range(5):
                print(f"      {MOTION_LABELS[ri]}: {' '.join(f'{sim[ri,j]:.3f}' for j in range(5))}")

    return result

# ============================================================================
# 主实验
# ============================================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("φ-Residual Fine-tuning Experiment (路径2)")
    print("架构改造 + LoRA微调 → C5相位结构涌现验证")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"strength={STRENGTH}, lora_r={LORA_R}, steps={TRAIN_STEPS}, lr={LR}")
    print()

    # ===== 加载模型 =====
    print("[1] 加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ 模型路径不存在: {MODEL_PATH}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"

    num_layers = config.num_hidden_layers
    print(f"  layers={num_layers}, hidden={config.hidden_size}")

    # 检测GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

    # 加载模型 (fp16节省内存)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    layer_alphas = compute_layer_alphas(num_layers, STRENGTH)

    # ===== 基线评估 (微调前) =====
    print("\n[2] 基线评估 (微调前, 无φ-Residual)...")
    baseline_avg = run_phase_experiment(model, tokenizer, num_layers)
    baseline_result = evaluate_c5_structure(baseline_avg, num_layers, "标准模型(微调前)")

    # ===== 基线评估 (微调前, 有φ-Residual) =====
    print("\n[3] 基线评估 (微调前, 有φ-Residual)...")
    hooks = apply_phi_hooks(model, layer_alphas)
    phi_before_avg = run_phase_experiment(model, tokenizer, num_layers)
    remove_hooks(hooks)
    phi_before_result = evaluate_c5_structure(phi_before_avg, num_layers, "φ-Residual(微调前)")

    # ===== 添加LoRA =====
    print("\n[4] 添加LoRA适配器...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R,  # alpha=r → scaling=1
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ===== 应用φ-Residual hooks (训练期间保持) =====
    print("\n[5] 应用φ-Residual hooks (训练期间)...")
    hooks = apply_phi_hooks(model, layer_alphas)
    print(f"  已注册 {len(hooks)} 个φ-Residual hooks")

    # ===== 准备训练数据 =====
    print("\n[6] 准备训练数据...")
    dataset = MotionDataset(TRAIN_TEXTS, tokenizer, MAX_SEQ_LEN)
    print(f"  训练样本数: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ===== 训练 =====
    print(f"\n[7] 开始LoRA微调 ({TRAIN_STEPS}步)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 启用gradient checkpointing节省VRAM
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    step = 0
    total_loss = 0
    log_interval = 50

    for epoch in range(100):  # 最多100个epoch
        for batch in dataloader:
            if step >= TRAIN_STEPS:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
            loss = outputs.loss

            if loss is None:
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1

            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f"  Step {step}/{TRAIN_STEPS}: loss={avg_loss:.4f}")
                total_loss = 0

        if step >= TRAIN_STEPS:
            break

    print(f"  训练完成! {step}步")

    # ===== 微调后评估 =====
    print("\n[8] 微调后评估 (有φ-Residual)...")
    model.eval()

    # hooks已经在训练期间保持, 直接跑实验
    phi_after_avg = run_phase_experiment(model, tokenizer, num_layers)
    phi_after_result = evaluate_c5_structure(phi_after_avg, num_layers, "φ-Residual(微调后)")

    # 去掉hooks, 评估纯LoRA (无φ-Residual)
    print("\n[9] 微调后评估 (无φ-Residual, 只有LoRA)...")
    remove_hooks(hooks)
    lora_only_avg = run_phase_experiment(model, tokenizer, num_layers)
    lora_only_result = evaluate_c5_structure(lora_only_avg, num_layers, "LoRA only(微调后,无φ)")

    # ===== 对比总结 =====
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)

    results = {
        "baseline": baseline_result,
        "phi_before": phi_before_result,
        "phi_after": phi_after_result,
        "lora_only": lora_only_result,
    }

    print(f"\n  {'指标':<25} {'基线'} {'φ(前)'} {'φ(后)'} {'LoRA only'}")
    print(f"  {'-'*60}")

    for metric in ['circular_ratio_mean', 'circular_ratio_gt1', 'k1_mean', 'k1_gt04', 'nearest_c5_mean']:
        vals = []
        for key in ['baseline', 'phi_before', 'phi_after', 'lora_only']:
            if results[key]:
                vals.append(f"{results[key][metric]:.4f}" if isinstance(results[key][metric], float) else str(results[key][metric]))
            else:
                vals.append("N/A")
        print(f"  {metric:<25} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}")

    # 关键判定
    print(f"\n  关键判定:")
    if phi_after_result and phi_before_result:
        cr_after = phi_after_result['circular_ratio_mean']
        cr_before = phi_before_result['circular_ratio_mean']
        if cr_after > cr_before + 0.05:
            print(f"  ✅ 微调后C5循环比显著提升: {cr_before:.4f} → {cr_after:.4f}")
        elif cr_after > cr_before + 0.01:
            print(f"  ⚠️ 微调后C5循环比小幅提升: {cr_before:.4f} → {cr_after:.4f}")
        else:
            print(f"  ❌ 微调后C5循环比无显著变化: {cr_before:.4f} → {cr_after:.4f}")

        k1_after = phi_after_result['k1_mean']
        k1_before = phi_before_result['k1_mean']
        if k1_after > k1_before + 0.05:
            print(f"  ✅ 微调后DFT k=1占比显著提升: {k1_before:.4f} → {k1_after:.4f}")
        elif k1_after > k1_before + 0.01:
            print(f"  ⚠️ 微调后DFT k=1占比小幅提升: {k1_before:.4f} → {k1_after:.4f}")
        else:
            print(f"  ❌ 微调后DFT k=1占比无显著变化: {k1_before:.4f} → {k1_after:.4f}")

        # φ-Residual vs LoRA only: 是否φ是关键?
        if lora_only_result:
            cr_lora = lora_only_result['circular_ratio_mean']
            print(f"\n  φ-Residual贡献判定:")
            if cr_after > cr_lora + 0.02:
                print(f"  ✅ φ-Residual是C5结构的关键: φ(后)={cr_after:.4f} > LoRA only={cr_lora:.4f}")
            else:
                print(f"  ❌ φ-Residual不是C5结构的关键: φ(后)={cr_after:.4f} ≈ LoRA only={cr_lora:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  总耗时: {elapsed/60:.1f}分钟")

    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(os.path.join(output_dir, "phi_finetune_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, default=convert, indent=2, ensure_ascii=False)

    print(f"\n  结果已保存到: {os.path.join(output_dir, 'phi_finetune_results.json')}")


if __name__ == "__main__":
    main()
