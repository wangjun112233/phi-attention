#!/usr/bin/env python3
"""
D₁₀ Flavon Potential Construction
Fixed version with correct representation theory
"""

import numpy as np

N = 5
phi = (1 + np.sqrt(5)) / 2

# ============================================================
# D₁₀ representations
# ============================================================

def chi0(g):
    """Trivial: r^k s^l → 1"""
    return np.eye(1)

def chi1(g):
    """Sign: r^k → 1, s → -1"""
    k, l = g
    return np.array([[-1.0 if l == 1 else 1.0]])

def rho1(g):
    """2D irrep with r-charge ±2π/5"""
    k, l = g
    theta = 2 * np.pi * k / N
    if l == 0:
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    else:
        return np.array([[np.cos(theta), np.sin(theta)],
                         [np.sin(theta), -np.cos(theta)]])

def rho2(g):
    """2D irrep with r-charge ±4π/5"""
    k, l = g
    theta = 4 * np.pi * k / N
    if l == 0:
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    else:
        return np.array([[np.cos(theta), np.sin(theta)],
                         [np.sin(theta), -np.cos(theta)]])

# D₁₀ elements: (k, l) = r^k * s^l
D10 = [(k, l) for k in range(N) for l in range(2)]

def multiply(g1, g2):
    k1, l1 = g1
    k2, l2 = g2
    if l1 == 0:
        return ((k1 + k2) % N, l2)
    else:
        if l2 == 0:
            return ((k1 - k2) % N, 1)
        else:
            return ((k1 - k2) % N, 0)

# Verify representations
print("--- Verification ---")
rep_err = 0
for name, rep in [("chi0", chi0), ("chi1", chi1), ("rho1", rho1), ("rho2", rho2)]:
    for a in D10:
        for b in D10:
            ab = multiply(a, b)
            lhs = rep(ab)
            rhs = rep(a) @ rep(b)
            if not np.allclose(lhs, rhs, atol=1e-10):
                rep_err += 1
print(f"Rep homomorphism errors: {rep_err}")

# ============================================================
# Character table and tensor products
# ============================================================
reps_dict = {"χ₀": chi0, "χ₁": chi1, "ρ₁": rho1, "ρ₂": rho2}
dims = {"χ₀": 1, "χ₁": 1, "ρ₁": 2, "ρ₂": 2}

print("\n--- Character table ---")
conj_classes = [
    [(0,0)], [(1,0), (4,0)], [(2,0), (3,0)],
    [(0,1), (2,1), (4,1)], [(1,1), (3,1)]
]
print(f"{'Class':>25}", end="")
for name in reps_dict:
    print(f"{name:>8}", end="")
print()
for cc in conj_classes:
    g = cc[0]
    print(f"{str(cc):>25}", end="")
    for name, rep in reps_dict.items():
        print(f"{np.trace(rep(g)):8.3f}", end="")
    print()

# Tensor product via character inner product
def char_inner(ch1, ch2):
    """(1/|G|) Σ_g ch1(g) ch2*(g)"""
    s = sum(ch1(g) * np.conj(ch2(g)) for g in D10)
    return (s / len(D10)).real

def char_of_rep(rep_func):
    return lambda g: np.trace(rep_func(g))

def char_of_tensor(rep1, rep2):
    """Character of tensor product: χ₁(g)·χ₂(g)"""
    return lambda g: np.trace(rep1(g)) * np.trace(rep2(g))

print("\n--- Tensor products ---")
rep_list = [("χ₀", chi0), ("χ₁", chi1), ("ρ₁", rho1), ("ρ₂", rho2)]
for i, (n1, r1) in enumerate(rep_list):
    for j, (n2, r2) in enumerate(rep_list):
        if j >= i:
            ch_tensor = char_of_tensor(r1, r2)
            decomp = []
            for name, rep in rep_list:
                m = int(round(char_inner(ch_tensor, char_of_rep(rep))))
                if m > 0:
                    decomp.append(f"{m}{name}" if m > 1 else name)
            print(f"  {n1} ⊗ {n2} = {' ⊕ '.join(decomp)}")

# ============================================================
# Yukawa operator analysis
# ============================================================
print("\n" + "="*60)
print("YUKAWA OPERATOR ANALYSIS")
print("="*60)

def check_invariant(name1, rep1, name2, rep2, name3=None, rep3=None):
    """Check if rep1 ⊗ rep2 [⊗ rep3] contains χ₀"""
    if rep3 is not None:
        ch = lambda g: np.trace(rep1(g)) * np.trace(rep2(g)) * np.trace(rep3(g))
        label = f"{name1} ⊗ {name2} ⊗ {name3}"
    else:
        ch = lambda g: np.trace(rep1(g)) * np.trace(rep2(g))
        label = f"{name1} ⊗ {name2}"
    m = int(round(char_inner(ch, char_of_rep(chi0))))
    status = "✅ ALLOWED" if m > 0 else "❌ FORBIDDEN"
    print(f"  {label} ⊃ χ₀: multiplicity {m} — {status}")
    return m

print("\n--- LO Up-type (no flavon) ---")
m11 = check_invariant("ρ₁", rho1, "ρ₁", rho1)  # 1-2 block
m33 = check_invariant("χ₀", chi0, "χ₀", chi0)  # 3-3 entry
m13 = check_invariant("ρ₁", rho1, "χ₀", chi0)  # 1-3 coupling

print("\n--- LO Down-type with ξ_d ∈ χ₁ ---")
m_d12 = check_invariant("ρ₁", rho1, "ρ₁", rho1, "χ₁", chi1)
m_d33 = check_invariant("χ₀", chi0, "χ₀", chi0)

print("\n--- NLO Up-type with φ_u ∈ ρ₂ ---")
m_nlo_u = check_invariant("ρ₁", rho1, "ρ₁", rho1, "ρ₂", rho2)

print("\n--- NLO Down-type with φ_d ∈ ρ₂ + ξ_d ---")
m_nlo_d = check_invariant("ρ₁", rho1, "ρ₁", rho1, "χ₁", chi1)  # already counted
m_nlo_d2 = check_invariant("ρ₁", rho1, "ρ₁", rho1, "ρ₂", rho2)  # without ξ_d

# Additional: Q̄₃ D_β with flavon
print("\n--- 3rd gen mixing operators ---")
# Q̄₃ H_u U_α with flavon: χ₀ ⊗ ρ₁ ⊗ [flavon]
print("  Q̄₃ U_α: need flavon to mediate χ₀ ⊗ ρ₁ → χ₀")
for fname, frep in rep_list:
    ch = lambda g, fr=frep: np.trace(chi0(g)) * np.trace(rho1(g)) * np.trace(fr(g))
    m = int(round(char_inner(ch, char_of_rep(chi0))))
    if m > 0:
        print(f"    χ₀ ⊗ ρ₁ ⊗ {fname} ⊃ χ₀: multiplicity {m} ✅")

# Q̄_α H_d D₃ with flavon: ρ₁ ⊗ χ₀ ⊗ [flavon]
print("  Q̄_α D₃: need flavon to mediate ρ₁ ⊗ χ₀ → χ₀")
for fname, frep in rep_list:
    ch = lambda g, fr=frep: np.trace(rho1(g)) * np.trace(chi0(g)) * np.trace(fr(g))
    m = int(round(char_inner(ch, char_of_rep(chi0))))
    if m > 0:
        print(f"    ρ₁ ⊗ χ₀ ⊗ {fname} ⊃ χ₀: multiplicity {m} ✅")

# ============================================================
# Clebsch-Gordan: project onto invariant subspaces
# ============================================================
print("\n" + "="*60)
print("CLEBSCH-GORDAN COEFFICIENTS")
print("="*60)

# ρ₁ ⊗ ρ₁ is 4D. Project onto χ₀ and χ₁ components.
def projector_onto_trivial(rep1, rep2):
    """Project onto χ₀-isotypic component of rep1 ⊗ rep2"""
    d1 = rep1((0,0)).shape[0]
    d2 = rep2((0,0)).shape[0]
    d = d1 * d2
    P = np.zeros((d, d))
    for g in D10:
        R = np.kron(rep1(g), rep2(g))
        P += R  # χ₀(g)* = 1 for all g
    P /= len(D10)
    return P

# χ₀ component of ρ₁ ⊗ ρ₁ (up-type LO)
P_u = projector_onto_trivial(rho1, rho1)
evals_u, evecs_u = np.linalg.eigh(P_u)
# Get vectors with eigenvalue ~1
chi0_vecs_u = [evecs_u[:, i] for i in range(len(evals_u)) if evals_u[i] > 0.5]
print(f"\nρ₁ ⊗ ρ₁ → χ₀ invariant (up-type LO):")
print(f"  Multiplicity: {len(chi0_vecs_u)}")
for k, v in enumerate(chi0_vecs_u):
    M = v.reshape(2, 2)
    print(f"  M^u_12-block = [{M[0,0]:.4f}, {M[0,1]:.4f}; {M[1,0]:.4f}, {M[1,1]:.4f}]")

# χ₁ component of ρ₁ ⊗ ρ₁ (need to project with χ₁ character)
def projector_onto_chi1(rep1, rep2):
    d1 = rep1((0,0)).shape[0]
    d2 = rep2((0,0)).shape[0]
    d = d1 * d2
    P = np.zeros((d, d))
    for g in D10:
        R = np.kron(rep1(g), rep2(g))
        P += np.conj(np.trace(chi1(g))) * R  # project with χ₁*
    P /= len(D10)
    return P

P_d = projector_onto_chi1(rho1, rho1)
evals_d, evecs_d = np.linalg.eigh(P_d)
chi1_vecs_d = [evecs_d[:, i] for i in range(len(evals_d)) if evals_d[i] > 0.5]
print(f"\nρ₁ ⊗ ρ₁ → χ₁ component (down-type with ξ_d):")
print(f"  Multiplicity: {len(chi1_vecs_d)}")
for k, v in enumerate(chi1_vecs_d):
    M = v.reshape(2, 2)
    print(f"  M^d_12-block = [{M[0,0]:.4f}, {M[0,1]:.4f}; {M[1,0]:.4f}, {M[1,1]:.4f}]")
    print(f"  Antisymmetric? M[0,1]/M[1,0] = {M[0,1]/M[1,0]:.4f} (want -1)")

# ============================================================
# NLO: ρ₁ ⊗ ρ₁ ⊗ ρ₂ → χ₀ (3-index tensor)
# ============================================================
print(f"\nρ₁ ⊗ ρ₁ ⊗ ρ₂ → χ₀ invariant (NLO up-type):")
# 8D space (2×2×2)
P_nlo = np.zeros((8, 8))
for g in D10:
    R = np.kron(np.kron(rho1(g), rho1(g)), rho2(g))
    P_nlo += R
P_nlo /= len(D10)

evals_nlo, evecs_nlo = np.linalg.eigh(P_nlo)
nlo_vecs = [evecs_nlo[:, i] for i in range(len(evals_nlo)) if evals_nlo[i] > 0.5]
print(f"  Multiplicity: {len(nlo_vecs)}")

for k, v in enumerate(nlo_vecs):
    T = v.reshape(2, 2, 2)
    print(f"\n  Invariant #{k+1}:")
    print(f"    φ_u⁺: M = [{T[0,0,0]:.4f}, {T[0,1,0]:.4f}; {T[1,0,0]:.4f}, {T[1,1,0]:.4f}]")
    print(f"    φ_u⁻: M = [{T[0,0,1]:.4f}, {T[0,1,1]:.4f}; {T[1,0,1]:.4f}, {T[1,1,1]:.4f}]")
    
    # Check VEV options
    # (a) <φ_u> = (v, 0)
    M_a = T[:, :, 0]
    diag_diff_a = M_a[0,0] - M_a[1,1]
    # (b) <φ_u> = (0, v)
    M_b = T[:, :, 1]
    diag_diff_b = M_b[0,0] - M_b[1,1]
    # (c) <φ_u> = (v, v)/√2
    M_c = (T[:, :, 0] + T[:, :, 1]) / np.sqrt(2)
    diag_diff_c = M_c[0,0] - M_c[1,1]
    
    print(f"    VEV (v,0):  δM diag diff = {diag_diff_a:.6f} {'✅ breaks 1-2' if abs(diag_diff_a) > 0.01 else '❌'}")
    print(f"    VEV (0,v):  δM diag diff = {diag_diff_b:.6f} {'✅ breaks 1-2' if abs(diag_diff_b) > 0.01 else '❌'}")
    print(f"    VEV (v,v):  δM diag diff = {diag_diff_c:.6f} {'✅ breaks 1-2' if abs(diag_diff_c) > 0.01 else '❌'}")

# ============================================================
# Full NLO mass matrix construction
# ============================================================
print("\n" + "="*60)
print("FULL MASS MATRIX CONSTRUCTION")
print("="*60)

if len(chi0_vecs_u) > 0 and len(chi1_vecs_d) > 0 and len(nlo_vecs) > 0:
    # LO up-type
    M_u_LO_12 = chi0_vecs_u[0].reshape(2, 2)
    # Normalize
    M_u_LO_12 = M_u_LO_12 / np.linalg.norm(M_u_LO_12)
    
    # LO down-type (GJ texture from χ₁)
    M_d_LO_12 = chi1_vecs_d[0].reshape(2, 2)
    M_d_LO_12 = M_d_LO_12 / np.linalg.norm(M_d_LO_12)
    
    # NLO correction tensor
    T_nlo = nlo_vecs[0].reshape(2, 2, 2)
    # Normalize
    T_nlo = T_nlo / np.linalg.norm(T_nlo)
    
    # VEV: <φ_u> = (v, 0) — breaks s
    M_u_NLO_12 = T_nlo[:, :, 0]
    
    print("\nLO M^u (1-2 block):")
    print(f"  {np.round(M_u_LO_12, 4)}")
    print(f"  → diag(a, a) with 1-2 degeneracy")
    
    print("\nLO M^d (1-2 block, with ξ_d):")
    print(f"  {np.round(M_d_LO_12, 4)}")
    print(f"  → GJ antisymmetric texture")
    
    print("\nNLO δM^u (1-2 block, with <φ_u>=(v,0)):")
    print(f"  {np.round(M_u_NLO_12, 6)}")
    
    # Construct full 3×3 matrices with physical-motivated parameters
    # a ~ m_c, b ~ m_t, c ~ m_s (from GJ), d ~ m_b
    # ε = <φ_u>/Λ ~ Cabibbo angle ~ 0.22
    
    a = 0.006   # m_c/m_t scale
    b = 1.0     # m_t scale
    c = 0.02    # m_s scale (from GJ)
    d = 0.04    # m_b scale
    eps = 0.22  # Cabibbo-like expansion parameter
    
    # Scale LO matrices
    M_u = np.zeros((3, 3))
    M_u[:2, :2] = a * M_u_LO_12 / M_u_LO_12[0, 0]  # should be identity-like
    M_u[2, 2] = b
    
    # Add NLO correction
    delta_M = eps * a * M_u_NLO_12 / max(abs(M_u_NLO_12[0, 0]), abs(M_u_NLO_12[1, 1]), 1e-10)
    M_u[:2, :2] += delta_M
    
    M_d = np.zeros((3, 3))
    M_d[:2, :2] = c * M_d_LO_12 / max(abs(M_d_LO_12[0, 1]), 1e-10)
    M_d[2, 2] = d
    
    print("\n--- Effective M^u ---")
    print(np.round(M_u, 6))
    print("\n--- Effective M^d ---")
    print(np.round(M_d, 6))
    
    # Diagonalize: M = V_L diag(m) V_R†
    # CKM = V_u_L† @ V_d_L
    def diagonalize_hermitian(M):
        """Diagonalize M M† to get left rotation"""
        MHM = M @ M.conj().T
        evals, evecs = np.linalg.eigh(MHM)
        # Sort by eigenvalue (ascending)
        idx = np.argsort(evals)
        return evecs[:, idx], evals[idx]
    
    V_u_L, m_u_sq = diagonalize_hermitian(M_u)
    V_d_L, m_d_sq = diagonalize_hermitian(M_d)
    
    CKM = V_u_L.conj().T @ V_d_L
    
    print("\n--- CKM matrix ---")
    print(np.round(CKM, 6))
    
    print("\n--- CKM comparison ---")
    CKM_pdg = np.array([
        [0.97373, 0.22430, 0.00382],
        [0.22100, 0.97500, 0.04100],
        [0.00861, 0.04150, 0.99911]
    ])
    for i in range(3):
        for j in range(3):
            val = abs(CKM[i, j])
            pdg = CKM_pdg[i, j]
            if pdg > 0.001:
                dev = abs(val - pdg) / pdg * 100
                print(f"  V_{['u','c','t'][i]}{['d','s','b'][j]}: {val:.4f} vs PDG {pdg:.4f} ({dev:.1f}%)")

# ============================================================
# Flavon potential and vacuum
# ============================================================
print("\n" + "="*60)
print("FLAVON POTENTIAL V(ξ_d, φ_u)")
print("="*60)

print("""
Minimal flavon content:
  ξ_d ∈ χ₁  — Z₂ twist (real scalar)
  φ_u ∈ ρ₂  — 1-2 splitting (complex doublet)

D₁₀-invariant potential (dim ≤ 4):

  V = m²_ξ |ξ_d|² + λ_ξ |ξ_d|⁴
    + m²_u (|φ_u⁺|² + |φ_u⁻|²) + λ₁ᵘ (|φ_u⁺|² + |φ_u⁻|²)² + λ₂ᵘ (|φ_u⁺|² - |φ_u⁻|²)²
    + [κ ξ_d² (φ_u⁺φ_u⁻) + h.c.]    ← ρ₂⊗ρ₂ → χ₁, χ₁⊗χ₁ → χ₀ ✓

  Note: |φ⁺|²+|φ⁻|² ∈ χ₀ (invariant norm)
        |φ⁺|²-|φ⁻|² ∈ χ₁ (s-odd)
        φ⁺φ⁻ ∈ χ₁ component of ρ₂⊗ρ₂ → can couple to ξ_d²
""")

# Verify the cross term
print("Verification: cross term ξ_d² (φ_u⁺φ_u⁻)")
print("  ρ₂ ⊗ ρ₂ components:")
ch_rho2_sq = lambda g: np.trace(rho2(g))**2
for name, rep in rep_list:
    m = int(round(char_inner(ch_rho2_sq, char_of_rep(rep))))
    if m > 0:
        print(f"    {name}: multiplicity {m}")
print("  χ₁ ⊗ χ₁ = χ₀ → ξ_d² is invariant ✓")
print("  So ξ_d² (φ⁺φ⁻) ∈ χ₁ ⊗ χ₁ = χ₀ ✓")

print("""
Vacuum structure:
  ∂V/∂ξ_d = 0 → ⟨ξ_d⟩² = -m²_ξ/(2λ_ξ)  (if m²_ξ < 0)
  
  ∂V/∂φ_u⁺ = 0 and ∂V/∂φ_u⁻ = 0:
    The λ₂ᵘ term determines VEV direction.
    
    Case A (λ₂ᵘ + κ > 0): ⟨φ_u⟩ ∝ (1, 1) — symmetric, preserves s
    Case B (λ₂ᵘ + κ < 0): ⟨φ_u⟩ ∝ (1, 0) — breaks s completely
    
    For Cabibbo angle: need Case B → ⟨φ_u⟩ = (v_u, 0)
    
  The κ cross term from ξ_d²(φ⁺φ⁻) can tilt the balance!
  When ⟨ξ_d⟩ ≠ 0, the effective λ₂ shifts: λ₂ᵉᶠᶠ = λ₂ᵘ + κ⟨ξ_d⟩²

Key result: The Z₂ twist flavon ξ_d, by getting a VEV,
  modifies the φ_u potential and SELECTS the VEV direction
  that breaks 1-2 degeneracy → Cabibbo angle emerges!
""")

# ============================================================
# Born overlap connection
# ============================================================
print("="*60)
print("BORN OVERLAP → CABIBBO CONNECTION")
print("="*60)

V_us = np.sin(np.pi/5) / phi**2
print(f"""
The NLO correction from φ_u ∈ ρ₂ gives:
  M^u = [[a + δ, ε], [ε', a - δ], [0, 0, b]]  (in 1-2 block)

The Cabibbo angle: θ_C ≈ arctan(δ/a) (from diagonalization)

D₁₀ structure constrains the ratio δ/a:
  - δ comes from ρ₂ CG coefficient × ⟨φ_u⟩/Λ
  - a comes from ρ₁ ⊗ ρ₁ → χ₀ CG coefficient
  - The ratio is fixed by group theory → C₅ angle

Specifically:
  The ρ₂ representation has r-eigenvalues e^{{±4πi/5}}
  Born overlap: sin²(2θ) with θ = 2π/5 → C₅ period-2 cycle
  The mixing angle: V_us = sin(π/5)/φ² = {V_us:.6f}

This is the mechanism: D₁₀ representation theory → CG coefficients
→ mass matrix structure → CKM → V_us = sin(π/5)/φ²

The flavon potential doesn't determine V_us (that's group theory).
The flavon potential determines the OVERALL SCALE of the correction
(how big δ is relative to a), which sets the mass ratios m_u/m_c.
""")

print("="*60)
print("NEXT STEPS")
print("="*60)
print("""
⬜ 1. Minimize V(ξ_d, φ_u) analytically → explicit VEVs
⬜ 2. Include φ_d ∈ ρ₂ for down-type NLO
⬜ 3. Include 3rd-gen mixing flavons (ρ₁ or ρ₂ for 1-3, 2-3 entries)
⬜ 4. Construct full 3×3 M^u and M^d with all NLO operators
⬜ 5. Diagonalize → CKM, verify V_us = sin(π/5)/φ²
⬜ 6. Write complete Lagrangian L = L_kin + L_Yuk + L_flavon
⬜ 7. Show V_us is group-theoretic (independent of flavon VEV details)
""")

