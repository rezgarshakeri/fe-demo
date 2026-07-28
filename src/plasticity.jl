# Multiplicative Hencky (finite-strain, von Mises/J2) plasticity, matrix-free,
# initial (total Lagrangian) configuration — mirrors Ratel's
# `plasticity-hencky-*-principal-initial` model. Plane strain, as with
# elasticity.jl/neohookean.jl (F embedded 3x3 with F33=1). Linear + Voce
# (saturation) isotropic hardening.
#
# Unlike linear elasticity/Neo-Hookean, this model has genuine HISTORY STATE:
# the inverse plastic right Cauchy-Green tensor Cp_inv and the accumulated
# plastic strain, both frozen during a load step's Newton iterations and only
# advanced (via `AdvancePlasticityState`) once that step has converged.
#
# Cp_inv stays block-diagonal under in-plane-only loading (by isotropy,
# induction from Cp_inv_0 = I), but unlike Neo-Hookean's elastic C33 (always
# exactly 1, fixed by F33=1), the plastic Cp_inv_33 genuinely evolves under
# plastic flow and must be tracked — it feeds into trace(eh) (the volumetric
# stress term) and the yield/hardening evolution even though it never
# directly appears in the 2D residual.
#
# The Jacobian is obtained via Enzyme.jl automatic differentiation through the
# closed-form 2x2 eigendecomposition and the (fixed-iteration-count) local
# return-mapping Newton solve, rather than Ratel's hand-derived eigenvector-
# sensitivity formulas — Ratel's own `*-ad-enzyme.h` model variants validate
# this as a legitimate strategy, not a shortcut. A FIXED iteration count is
# used (not Ratel's data-dependent early-break): differentiating through a
# converged fixed-point iteration equals the true implicit derivative once
# convergence is reached, and a fixed trip count is simpler/more robust for
# AD than a data-dependent loop exit.

"""
    PlasticityParams(bulk, mu, sigma_0, H, sigma_inf, omega)

Material parameters for multiplicative Hencky plasticity: bulk/shear moduli
plus von Mises yield stress `sigma_0`, linear hardening modulus `H`, Voce
saturation stress `sigma_inf`, and saturation decay rate `omega`.
"""
struct PlasticityParams
    bulk::Float64
    mu::Float64
    sigma_0::Float64
    H::Float64
    sigma_inf::Float64
    omega::Float64
end

"""
    PlasticityState(accumulated_plastic, Cp_inv11, Cp_inv12, Cp_inv22, Cp_inv33)

Persistent per-quadrature-point history state: the accumulated (equivalent)
plastic strain and the inverse plastic right Cauchy-Green tensor `Cp_inv`
(block-diagonal: in-plane 2x2 block + a genuinely-evolving out-of-plane
scalar, see module notes). The virgin/undeformed state is `Cp_inv = I`,
`accumulated_plastic = 0`.
"""
struct PlasticityState
    accumulated_plastic::Float64
    Cp_inv11::Float64
    Cp_inv12::Float64
    Cp_inv22::Float64
    Cp_inv33::Float64
end

VirginPlasticityState() = PlasticityState(0.0, 1.0, 0.0, 1.0, 1.0)

flow_stress(p::PlasticityParams, alpha) = p.sigma_0 + p.H * alpha + (p.sigma_inf - p.sigma_0) * (1 - exp(-p.omega * alpha))
hardening_slope(p::PlasticityParams, alpha) = p.H + p.omega * (p.sigma_inf - p.sigma_0) * exp(-p.omega * alpha)

"""
    ReturnMappingDeltaGamma(p, q_trial, alpha_n)

Fixed-iteration-count (8) Newton solve for the plastic multiplier `Δγ`
satisfying the von Mises consistency condition
`q_trial - 3*mu*Δγ - flow_stress(alpha_n + Δγ) = 0`.
"""
function ReturnMappingDeltaGamma(p::PlasticityParams, q_trial, alpha_n)
    dgamma = zero(q_trial)
    for _ = 1:8
        alpha = alpha_n + dgamma
        phi = q_trial - 3 * p.mu * dgamma - flow_stress(p, alpha)
        dphi = -3 * p.mu - hardening_slope(p, alpha)
        dgamma = dgamma - phi / dphi
    end
    return dgamma
end

"""
    Eig2x2Sym(a11, a12, a22)

Closed-form eigendecomposition of a symmetric 2x2 matrix `[a11 a12; a12 a22]`.
Returns `(lam1, lam2, v1, v2)` with `lam1 >= lam2`.
"""
function Eig2x2Sym(a11, a12, a22)
    tr2 = (a11 + a22) / 2
    diff2 = (a11 - a22) / 2
    r = sqrt(diff2^2 + a12^2)
    lam1 = tr2 + r
    lam2 = tr2 - r
    theta = 0.5 * atan(2 * a12, a11 - a22)
    c, s = cos(theta), sin(theta)
    v1 = [c, s]
    v2 = [-s, c]
    return lam1, lam2, v1, v2
end

"""
    Inv2x2(A)

Closed-form inverse of a 2x2 matrix, returned as a stack-allocated `SMatrix`.
`LinearAlgebra.inv` on a plain `Matrix` dispatches to a LAPACK-based LU
factorization that Enzyme cannot differentiate through; heap-allocated plain
`Matrix` results also generate enough GC pressure inside Enzyme's
differentiated hot path (thousands of calls per Newton solve) to crash the
process outright, not just run slowly — `SMatrix` avoids both problems.
"""
Inv2x2(A) = SMatrix{2,2}(A[2, 2], -A[2, 1], -A[1, 2], A[1, 1]) ./ (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])

"""
    Eig2x2SymValues(a11, a12, a22)

Closed-form eigenvalues `(lam1, lam2, r)` of a symmetric 2x2 matrix
`[a11 a12; a12 a22]`, with `r = (lam1-lam2)/2`. `r` is computed with a tiny
regularizing `+eps^2` inside the square root: the exact `sqrt(x)` has
unbounded derivative at `x=0`, which would otherwise poison every downstream
AD derivative whenever the matrix has (near-)repeated eigenvalues — e.g. at
`F=I`, the start of every simulation. The regularization changes eigenvalues
by a negligible `O(eps^2)` while keeping derivatives finite everywhere,
including exactly at repeated eigenvalues.
"""
function Eig2x2SymValues(a11, a12, a22)
    tr2 = (a11 + a22) / 2
    diff2 = (a11 - a22) / 2
    r = sqrt(diff2^2 + a12^2 + 1e-24)
    return tr2 + r, tr2 - r, r
end

"""
    IsoTensorFromEigVals2x2(A, g1, g2, r)

Reconstruct the symmetric 2x2 isotropic tensor function `g1*P1 + g2*P2` (`P1`,
`P2` the eigenprojections of `A`, eigenvalues `A_avg +/- r`) directly from
`A`'s entries, `g1`, `g2` and `r` — without ever forming eigenvectors.

This matters because eigenvectors are multivalued (undefined direction) at
repeated eigenvalues, giving a genuinely singular AD derivative there — even
though the *reconstructed tensor* is perfectly smooth at that point (when
`g1 == g2`, `g1*P1+g2*P2 = g1*(P1+P2) = g1*I` regardless of eigenvector
choice). This formula sidesteps eigenvectors entirely via
`P1 - P2 = (A - A_avg*I) / r`, so it stays smooth (matching Ratel's own
hand-derived eigenvector-sensitivity formulas in spirit, just derived
differently) through the point that broke the naive eigenvector approach.
"""
function IsoTensorFromEigVals2x2(A, g1, g2, r)
    A_avg = (A[1, 1] + A[2, 2]) / 2
    coeff = (g1 - g2) / (2r)
    gavg = (g1 + g2) / 2
    T11 = gavg + coeff * (A[1, 1] - A_avg)
    T22 = gavg + coeff * (A[2, 2] - A_avg)
    T12 = coeff * A[1, 2]
    return SMatrix{2,2}(T11, T12, T12, T22)
end

"""
    PlasticityLocal(p, grad_u, state_n)

Local (single quadrature point) constitutive update: given `grad_u = du/dX`
(2x2) and the frozen state `state_n` from the last converged load step,
returns `(P, state_trial)` — the first Piola-Kirchhoff stress and the
trial-updated state (only meaningful to keep once the outer Newton solve for
this load step has converged; see [`AdvancePlasticityState`](@ref)).
"""
function PlasticityLocal(p::PlasticityParams, grad_u, state_n::PlasticityState)
    gu = SMatrix{2,2}(grad_u) # stack-allocated: see Inv2x2 docstring for why this matters
    F = gu + I
    Finv = Inv2x2(F)

    Cpinv = SMatrix{2,2}(state_n.Cp_inv11, state_n.Cp_inv12, state_n.Cp_inv12, state_n.Cp_inv22)
    Cpinv33 = state_n.Cp_inv33

    be2 = F * Cpinv * F' # trial left Cauchy-Green tensor, in-plane block
    be33 = Cpinv33       # trial be, out-of-plane (F33 = 1)

    lam1, lam2, r = Eig2x2SymValues(be2[1, 1], be2[1, 2], be2[2, 2])

    eh1 = 0.5 * log1p(lam1 - 1) # Hencky strain eigenvalues (log1p: stable near F=I)
    eh2 = 0.5 * log1p(lam2 - 1)
    eh3 = 0.5 * log1p(be33 - 1)
    trace_eh = eh1 + eh2 + eh3
    eh1d, eh2d, eh3d = eh1 - trace_eh / 3, eh2 - trace_eh / 3, eh3 - trace_eh / 3

    s1, s2, s3 = 2 * p.mu * eh1d, 2 * p.mu * eh2d, 2 * p.mu * eh3d
    q_trial = sqrt(1.5 * (s1^2 + s2^2 + s3^2))
    flow0 = flow_stress(p, state_n.accumulated_plastic)
    phi_trial = q_trial - flow0

    # factor is only ever divided by q_trial inside the yielding branch, where
    # q_trial > flow0 > 0 is guaranteed by the phi_trial trigger condition —
    # q_trial can be exactly 0 in the elastic branch (e.g. grad_u = 0), which
    # would otherwise give 0/0 = NaN.
    if phi_trial > 1e-8 * flow0
        dgamma = ReturnMappingDeltaGamma(p, q_trial, state_n.accumulated_plastic)
        factor = 1 - 3 * p.mu * dgamma / q_trial
    else
        dgamma = zero(q_trial)
        factor = one(q_trial)
    end
    eh1d, eh2d, eh3d = eh1d * factor, eh2d * factor, eh3d * factor

    tau1 = 2 * p.mu * eh1d + p.bulk * trace_eh
    tau2 = 2 * p.mu * eh2d + p.bulk * trace_eh
    # tau3 (out-of-plane) is never needed: it doesn't enter the 2D residual.

    tau2x2 = IsoTensorFromEigVals2x2(be2, tau1, tau2, r)
    P = tau2x2 * Finv'

    be1_new = exp(2 * (eh1d + trace_eh / 3))
    be2_new = exp(2 * (eh2d + trace_eh / 3))
    be3_new = exp(2 * (eh3d + trace_eh / 3))
    be2x2_new = IsoTensorFromEigVals2x2(be2, be1_new, be2_new, r)
    Cpinv_new = Finv * be2x2_new * Finv'

    state_trial = PlasticityState(state_n.accumulated_plastic + dgamma,
        Cpinv_new[1, 1], Cpinv_new[1, 2], Cpinv_new[2, 2], be3_new)

    return P, state_trial
end

PlasticityStressOnly(p::PlasticityParams, grad_u, state_n::PlasticityState) = PlasticityLocal(p, grad_u, state_n)[1]

"""
    PlasticityResidual(p, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc, states)

Matrix-free residual for 2D multiplicative Hencky plasticity, given the
per-(element, quadrature point) frozen history `states::Matrix{PlasticityState}`
(size `Bu.Q^2 x num_elem`) from the last converged load step.
"""
function PlasticityResidual(p::PlasticityParams, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, f, bc_idx, u_bc, states)
    u = copy(u_in)
    v = zero(u)
    u[bc_idx] .= u_bc

    num_elem = size(Ind.idx_u, 2)
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        q, wdetJ, dXdx_T = GetQdata(Coord_E, Bx)

        ue = Ind.Er_u[e] * u
        Duq = Bu.D * ue

        f0 = Compute_f0(p.bulk, p.mu, wdetJ, q, f) # f(bulk, mu, x, y): plasticity's material params, not Lame's lambda
        s = zeros(4 * Bu.Q^2)
        for i = 1:Bu.Q^2
            grad_u = GradU(dXdx_T[i, :, :], Duq, i, Bu.Q^2)
            P, _ = PlasticityLocal(p, grad_u, states[i, e])
            Pmap = dXdx_T[i, :, :]' * P'
            s[i] = wdetJ[i] * Pmap[1, 1]
            s[i+Bu.Q^2] = wdetJ[i] * Pmap[2, 1]
            s[i+2*Bu.Q^2] = wdetJ[i] * Pmap[1, 2]
            s[i+3*Bu.Q^2] = wdetJ[i] * Pmap[2, 2]
        end
        ve = Bu.B' * f0 + Bu.D' * s
        v += sparse(Ind.Er_u[e]') * ve
    end
    v[bc_idx] = u_in[bc_idx] - u[bc_idx]
    return v
end

"""
    PlasticityResidualNeumann(p, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc, states, Ft)

[`PlasticityResidual`](@ref) with an additional (constant, precomputed)
traction load vector `Ft` from `GetTractionGlobal` subtracted in.
"""
function PlasticityResidualNeumann(p::PlasticityParams, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, f, bc_idx, u_bc, states, Ft)
    v = PlasticityResidual(p, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc, states)
    v .-= vec(Ft)
    v[bc_idx] = u_in[bc_idx] - u_bc
    return v
end

"""
    PlasticityJacobian(p, u_in, xc, yc, Ind, Bx, Bu, bc_idx, states)

Tangent stiffness for 2D multiplicative Hencky plasticity, linearized about
the current state `u_in` (frozen history `states`), via Enzyme.jl forward-mode
differentiation of [`PlasticityStressOnly`](@ref).
"""
function PlasticityJacobian(p::PlasticityParams, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, bc_idx, states)
    num_elem = size(Ind.idx_u, 2)
    rows, cols, vals = Int[], Int[], Float64[]
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        _, wdetJ, dXdx_T = GetQdata(Coord_E, Bx)

        ue = Ind.Er_u[e] * u_in
        Duq = Bu.D * ue
        # SMatrix (stack-allocated): see Inv2x2's docstring — heap-allocated
        # Matrix inputs here generate enough GC pressure inside Enzyme's
        # differentiated hot path (called Q^2 * P^2 * num_comp times per
        # element) to crash the process outright on non-trivial problems.
        grad_u_qp = [SMatrix{2,2}(GradU(dXdx_T[i, :, :], Duq, i, Bu.Q^2)) for i = 1:Bu.Q^2]

        Jac = zeros(Bu.num_comp * Bu.P * Bu.P, Bu.num_comp * Bu.P * Bu.P)
        for c = 0:Bu.num_comp-1
            for j = 1:Bu.P*Bu.P
                Ddu = Bu.D[:, j+c*Bu.P*Bu.P]
                df1 = zeros(4 * Bu.Q^2)
                for i = 1:Bu.Q^2
                    grad_du = SMatrix{2,2}(GradU(dXdx_T[i, :, :], Ddu, i, Bu.Q^2))
                    dP = Enzyme.autodiff(Enzyme.Forward, PlasticityStressOnly,
                        Enzyme.Const(p), Enzyme.Duplicated(grad_u_qp[i], grad_du), Enzyme.Const(states[i, e]))[1]
                    dPmap = dXdx_T[i, :, :]' * dP'
                    df1[i] = wdetJ[i] * dPmap[1, 1]
                    df1[i+Bu.Q^2] = wdetJ[i] * dPmap[2, 1]
                    df1[i+2*Bu.Q^2] = wdetJ[i] * dPmap[1, 2]
                    df1[i+3*Bu.Q^2] = wdetJ[i] * dPmap[2, 2]
                end
                Jac[:, j+c*Bu.P*Bu.P] = Bu.D' * df1
            end
        end
        inds = rowvals(sparse(Ind.Er_u[e]'))
        append!(rows, kron(ones(Bu.num_comp * Bu.P * Bu.P), inds))
        append!(cols, kron(inds, ones(Bu.num_comp * Bu.P * Bu.P)))
        append!(vals, vec(Jac))
    end
    A = sparse(rows, cols, vals)
    A[bc_idx, :] .= 0.0
    A[:, bc_idx] .= 0.0
    for i in eachindex(bc_idx)
        A[bc_idx[i], bc_idx[i]] = 1.0
    end
    return A
end

"""
    AdvancePlasticityState(p, u, xc, yc, Ind, Bx, Bu, states)

Recompute the local plasticity state at the converged displacement `u` and
return the advanced `states` array to use as the frozen history for the next
load step. Call this once per load step, only after the outer Newton solve
for that step has converged.
"""
function AdvancePlasticityState(p::PlasticityParams, u, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, states)
    num_elem = size(Ind.idx_u, 2)
    new_states = similar(states)
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        _, _, dXdx_T = GetQdata(Coord_E, Bx)
        ue = Ind.Er_u[e] * u
        Duq = Bu.D * ue
        for i = 1:Bu.Q^2
            grad_u = GradU(dXdx_T[i, :, :], Duq, i, Bu.Q^2)
            _, new_states[i, e] = PlasticityLocal(p, grad_u, states[i, e])
        end
    end
    return new_states
end
