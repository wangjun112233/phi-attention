# φ-Residual: Zero-Weight Semantic Shift via Five-Motion Residual Scaling

## TL;DR

We found that scaling residual connections in a pretrained LLM according to a pentagonal (C5) rhythm produces **observable, non-random semantic shifts** — without changing any weights, RoPE, or attention computation.

- 10/10 test prompts show text divergence (baseline vs φ-Residual)
- Minimum effective dose: STRENGTH=0.05, norm ratio 1.38x (safe)
- Shift direction is consistent: **external classification → internal structure**

## What we did

Standard Transformer: `hidden = residual + sublayer_output` (α=1 for all layers)

φ-Residual: `hidden = residual + α_k × sublayer_output`

where α_k follows a **five-motion cycle** derived from pentagonal symmetry (C5):

| Phase | k | φ-power | α_base | Semantic role |
|-------|---|---------|--------|---------------|
| Recognize | 0 | φ⁰ | 1.000 | Identify/maintain |
| Encounter | 1 | φ⁻¹ | 0.618 | Contact/couple |
| Dissipate | 2 | φ⁻² | 0.382 | Consume/relax |
| Split | 3 | φ¹ | 1.618 | Amplify/diverge |
| Remain | 4 | φ⁻³ | 0.236 | Residual/memory |

The actual α per layer:
```
α = 1 + STRENGTH × (PHI_POWERS[k % 5] / √(layer + 2) - 1)
```
with Z₂ flip on odd groups: `α → 2 - α`.

**Implementation**: `register_forward_hook` on `self_attn` and `mlp` — zero modification to `DecoderLayer.forward`, RoPE, KV cache, or attention masks.

## Results

**Model**: Qwen2.5-1.5B, float32, CPU inference

**STRENGTH=0.05** (norm safe, max 1.38x baseline):

| Prompt | First divergence | Shift direction |
|--------|-----------------|-----------------|
| "The fundamental nature of reality is" | char 295 | Answer refinement |
| "Consciousness arises from" | char 267 | "a physical system" → "the physical basis of consciousness" |
| "The relationship between order and chaos is" | char 68 | "question" → "issue", "investigate" → "study" |
| "In physics, the most fundamental principle is" | char 298 | Exposition → question format |
| **"The meaning of existence is"** | **char 29** | **"to be in the world" → "the meaning of being"** |

**Three consistent shift directions across all 10 tests (2 strengths × 5 prompts)**:

1. **External → Internal**: "a physical system" → "the physical basis of consciousness"
2. **Concrete → Abstract**: "question" → "issue", "nonlinear" → "discrete-time"
3. **Narrative → Structural**: continuous prose → question/answer format, lowercase → capitalized

These aren't random — they're what you'd expect from a model whose residual connections are being differentially weighted toward structure-revealing rather than fact-listing.

## What doesn't work

We learned the hard way that **RoPE cannot be swapped in a pretrained model**:

| Attempt | Strategy | Result | Why |
|---------|----------|--------|-----|
| v8 | φ² RoPE (θ=2.618) | Gibberish | θ from 1,000,000 → 2.618 = 380,000× frequency shift |
| v9 | Dual-path mixing | Norm 50-120× | KV cache cross-contamination |
| v10 | C5 perturbation overlay | Still gibberish | RoPE still broken |
| v12 | Manual forward rewrite | TypeError | tuple vs tensor incompatibility |

**The RoPE lesson is critical**: position encoding is the most sensitive component. Any change to θ requires retraining from scratch or fine-tuning. φ-Residual works precisely because it doesn't touch it.

## Quantitative metrics

| STRENGTH | KL divergence (range) | Max norm ratio | Safe? |
|-----------|----------------------|----------------|-------|
| 0.05 | 0.006-0.008 | 1.38× | ✅ |
| 0.08 | 0.014-0.021 | ~2.0× | ⚠️ Borderline |
| 0.10 | 0.034 | 2.31× | ❌ |
| 0.30 | 0.338 | 6.55× | ❌ |

Minimum effective dose: **STRENGTH=0.05**, KL≈0.007, all prompts diverge within 80 tokens.

## Why this matters

1. **The shift is structural, not random** — α values come from C5 symmetry (pentagonal group), not from hyperparameter search
2. **Zero training cost** — no weight updates, no GPU needed, works on CPU with any pretrained model
3. **The five-motion cycle is a theoretical prediction, not a fit** — the same C5 algebra that produces V_us=0.2245 (Cabibbo angle, 0.35% error) and sin²θ_W=3/8 (weak mixing angle) also produces the α schedule that shifts semantics

The full theoretical chain:
```
C5 algebra → φ-scaling → five-motion PDE → φ-Residual → pretrained model semantic shift
```

## Reproduce it

```bash
pip install torch transformers accelerate
python d10_patch_qwen_v15.py
```

Code: https://github.com/wangjun112233/phi-attention/tree/main/five_motion_arch

Full validation report: https://github.com/wangjun112233/phi-attention/blob/main/five_motion_arch/PHI_RESIDUAL_VALIDATION.md

## Open questions

1. Does this scale to larger models (7B, 70B)? Norm safety at 1.5B doesn't guarantee safety at scale.
2. Is the "external → internal" shift direction robust across languages (Chinese, multilingual)?
3. Can φ-Residual improve long-context consistency? (Untested — the semantic shift observation doesn't directly imply consistency improvement)
4. What happens during fine-tuning with φ-Residual active? Does the model learn to compensate?
5. Can the C5 coupling (attention-level modification) be safely added on top of φ-Residual?

## Contact

GitHub: https://github.com/wangjun112233/phi-attention
Email: wangjun112233@users.noreply.github.com

---

*This work is part of the C5 framework — a pentagonal symmetry approach to understanding structure in physics and AI. The same algebraic structure that produces the Cabibbo angle and weak mixing angle from C5 geometry also predicts the residual scaling schedule that shifts LLM semantics.*
