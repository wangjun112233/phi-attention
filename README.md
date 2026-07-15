# C5-RPB: Diagnose & Inject C5 Phase Structure into LLMs

**5 minutes to diagnose your LLM's C5 phase compatibility, 0.6% cost to inject a detectable attention topology.**

---

## What It Does

- **Diagnose** — Measure head differentiation and C5 compatibility (k1 metric) in any transformer model
- **Inject** — C5-RPB bias makes heads exhibit 5-phase cyclic structure
- **Verify** — Z₂ flip is detectable (collapse shift), PPL cost < 1%

---

## Quick Start

```bash
# Step 1: Diagnose
python five_motion_arch/c5_rpb_qwen_verify_v4.py --model_path YOUR_MODEL --device cpu

# Step 2: Check PPL cost
python five_motion_arch/c5_rpb_perplexity_test.py --model_path YOUR_MODEL --device cpu
```

---

## Results

### Qwen2.5-1.5B

| Metric | Standard | C5-RPB (amp=0.5) | Delta |
|--------|----------|-------------------|-------|
| DFT k1 | 0.017 | 0.313 | +0.296 |
| PPL | 5.67 | 5.70 | +0.6% |
| Z₂ collapse shift | — | — | 0.184 |

### Qwen2.5-3B

| Metric | Standard | C5-RPB (amp=0.5) | Delta |
|--------|----------|-------------------|-------|
| DFT k1 | 0.026 | 0.267 | +0.241 |
| Z₂ collapse shift | — | — | 0.158 |

---

## How It Works

**C5-RPB** = Relative Position Bias with C5-cycle phase encoding

```
B[h,i,j] = A·cos(2π(h%5)/5 + φ_shift + π(i-j)/L)
```

- `h%5` → C5 rotation (phase group)
- `i-j` → relative position
- `φ_shift` → Z₂ negation

**Why attention, not residual?** Attention is a live carrier for C5 structure. Residual connections wash out topology — they mix everything uniformly. Attention bias preserves phase differentiation because each head selects *where to look* independently.

---

## Full Validation Chain

8 experiments, from concept to working solution:

| Experiment | Result | Key Finding |
|------------|--------|-------------|
| φ-Residual (v15–v18) | ✅ PASS | Changes "how much" (α scaling works) |
| Attribution Patching | ❌ FAIL | C5 ≠ spatial parcels |
| Phase Structure | ❌ FAIL | C5 ≠ layer rotation |
| φ-Residual + LoRA | ❌ FAIL | LoRA absorbs variance |
| Superposition (vector α) | ❌ FAIL | Residual washes C5 |
| C5-Q Coupling (numpy) | ❌ FAIL | Random projection drowns signal |
| C5-RPB (numpy) | ⚠️ WEAK | Z₂ grows but k1 ≈ 0 |
| C5-RPB (1.5B real) | ✅✅✅ PASS | k1=0.31, Z₂=0.184, PPL+0.6% |

**Takeaway:** 6 out of 8 approaches failed. The only path that works injects C5 structure into attention bias (not residual, not Q/K projection). The signal must go where heads make decisions.

---

## Two Orthogonal Knobs

| Knob | Controls | Mechanism |
|------|----------|-----------|
| **φ-Residual** | "how much" (α scaling) | Scales residual connection amplitude |
| **C5-RPB** | "where to look" (phase bias) | Phase-biased relative position encoding |

These are independent — you can use either or both.

---

<details>
<summary><strong>Technical Details (Background)</strong></summary>

## The φ-Attention Recipe (丹方)

φ-Attention is a coupled multi-head attention mechanism derived from the C5-cycle structure. The coupling weight `cos(72°)` is the **unique** value that makes the 5-cycle adjacency matrix's largest eigenvalue equal to `φ = 1.618...` (the golden ratio). This is not a hyperparameter — it's an algebraic necessity.

### ① Furnace (炉子) — The Structure

Replace independent attention heads with C5-cycle coupled heads:

```
A = cos72° · [[0,1,0,0,1],
               [1,0,1,0,0],
               [0,1,0,1,0],
               [0,0,1,0,1],
               [1,0,0,1,0]]
```

Each head talks only to its two neighbors. The coupling topology is a 5-cycle. No learned parameters.

### ② Nature (药性) — The Eigenvalue

```
λ_max(A) = 2·cos72°·cos(2π/5) = φ = 1.618034...  (error: 6.7×10⁻¹⁶)
```

`cos72°` is the **only** weight that makes `λ_max = φ`. This self-consistency holds only for n=5. Not n=4, not n=6 — only the pentagon.

### ③ Heat Control (火候) — How Information Flows

| Property | C5-coupled | C8-coupled | Uncoupled |
|----------|-----------|-----------|-----------|
| Mixing time | **11 steps** | 24 steps | ∞ (never) |
| PF eigenvector | [0.2, 0.2, 0.2, 0.2, 0.2] | [0.125×8] | — |
| Spectral gap | **0.4271** | 0.1810 | 0 |
| IB: I(Z;Y) at convergence | **0.9991** | — | 0 |

- Signal propagates to neighbors in 1 step, reaches uniform distribution by step 22
- PF eigenvector is perfectly uniform → symmetric equal-weight 分工
- C5 mixes 2.3× faster than C8 (shorter diameter)
- Information Bottleneck: C5 preserves more task-relevant information at the same compression rate

### ④ Refining (炼化) — What It Produces

| Model | JSD ↓ | Effective DOF ↓ | Spectral Gap ↑ |
|-------|-------|-----------------|----------------|
| **φ5 (C5-coupled)** | **0.451** | **4.50** | **0.0707** |
| Std5 (uncoupled) | 0.521 | 5.00 | 0.0034 |
| Std8 (uncoupled) | 0.542 | 7.99 | 0.0020 |
| φ8 (C8-coupled) | 0.513 | 7.23 | 0.0187 |

- **13% better long-range consistency** (JSD)
- **20× stronger long-range coupling** (spectral gap)
- L2 regularization completely fails to reproduce this (entropy → 0.999, degraded)

### ⑤ Pharmacology (药理) — Why It Must Work

Complete causal chain, verified 7/7:

```
C5 coupling (cos72°)
    │  λ₀=φ amplifies DC, λ₂=0.5 compresses harmonics
    ▼
DC energy concentration (0.439 vs 0.198)
    │  DFT diagonalization is algebraic, not empirical
    ▼
Representational redundancy (DOF 5→2.2, 56% redundancy)
    │  Concentrated DC → fewer effective modes
    ▼
Slower decay (α=0.000216 < 0.000249)
    │  Redundant channels preserve information longer
    ▼
Better long-range consistency (ret@2048: 0.622 > 0.606)
```

Every step is a direct algebraic consequence of the C5 structure. The chain does not depend on weights, inputs, or training — it follows from `λ_k = 1 + 2·cos72°·cos(2πk/5)`.

### Summary Table

| Claim | Evidence | Type |
|-------|----------|------|
| λ_max = φ (zero parameter) | Eigenvalue computation, error 6.7e-16 | Algebraic |
| cos72° is the unique weight | Only n=5 cycle satisfies self-consistency | Algebraic |
| DC ratio = 0.439 (theory 0.4396) | DFT analysis, error < 0.003 | Algebraic + Empirical |
| 20× spectral gap | 0.0707 vs 0.0034 | Empirical |
| 13% JSD improvement | 0.451 vs 0.521 | Empirical |
| Causal chain 7/7 | All steps verified | Mixed |

### Relation to Existing Work

| Method | Approach | φ-Attention difference |
|--------|----------|----------------------|
| MoBA (Kimi) | Sparse attention via MoE gating | φ-Attention couples heads, doesn't sparsify |
| AttnRes (Kimi) | Replace residual connections with depth-attention | φ-Attention couples within a layer, not across layers |
| Lightning Attention (MiniMax) | Linear attention for long context | φ-Attention improves decay rate via structural redundancy |
| Grouped Query Attention | Reduce KV heads | φ-Attention keeps all heads, adds coupling |
| Multi-Head Attention | Independent parallel heads | φ-Attention makes heads interdependent via C5 topology |

**Complementary, not competing.** C5-RPB can be combined with sparse/linear/cross-layer methods — it addresses a different bottleneck: the *phase structure* of the head interaction graph.

</details>

---

## Repository Structure

```
phi-attention/
├── README.md                                    # This file
├── phi_attention_module.py                      # Core φ-Attention module
├── phi_attention.py                             # Standalone experiment script
├── phi_attention_colab.py                       # Colab-compatible version
├── phi_attention_numpy.py                       # NumPy-only implementation
├── phi_attention_termux.py                      # Termux-compatible version
├── phi_attention_report.md                      # Toy experiment report
├── phi_c5_dynamics_report.md                    # Mixing time, PF, IB analysis
├── phi_pharmacology.py                          # Causal chain verification
├── phi_pharmacology_report.md                   # 7/7 chain closed
├── phi_longrange_100seed.py                     # 100-seed long-range experiment
├── phi_longrange_100seed_report.md              # Statistical analysis
├── phi_vs_l2_experiment.py                      # φ vs L2 comparison
├── phi_vs_l2_report.md                          # L2 fails to reproduce C5 effect
├── five_motion_arch/                            # C5-RPB diagnostic & injection tools
│   ├── c5_rpb_qwen_verify_v4.py                 # ★ C5-RPB diagnostic (head-level h%5)
│   ├── c5_rpb_perplexity_test.py                # ★ PPL cost measurement
│   ├── PHI_RESIDUAL_VALIDATION.md               # Full validation report (v15–v19)
│   ├── c5_rpb_qwen_verify.py                    # C5-RPB verify (v1)
│   ├── c5_rpb_qwen_verify_v2.py                 # C5-RPB verify (v2)
│   ├── c5_rpb_qwen_verify_v3.py                 # C5-RPB verify (v3)
│   ├── c5_rpb_attention_verify.py               # NumPy C5-RPB verification
│   ├── c5_attention_numpy_verify.py             # NumPy attention verification
│   ├── c5_attention_sweep.py                    # Parameter sweep
│   ├── d10_patch_qwen_v15.py                    # φ-Residual (v15, passing)
│   ├── d10_patch_qwen_v16.py                    # φ-Residual (v16)
│   ├── d10_patch_qwen_v17.py                    # φ-Residual (v17)
│   ├── five_motion_attribution.py               # Attribution patching experiment
│   ├── five_motion_phase.py                     # Phase structure experiment
│   ├── phi_finetune_experiment.py               # LoRA finetune experiment
│   └── ...                                     # Earlier versions (v2–v14)
└── five_motion_bridge/                          # Bridge deployment scripts
    ├── five_motion_bridge.py
    └── deploy_bridge.sh
```

---

## License

MIT

## Contact

- **Email:** fdr-factor@coze.email
