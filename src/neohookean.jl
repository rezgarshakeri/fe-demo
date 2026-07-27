# Isochoric-volumetric split Neo-Hookean hyperelasticity, matrix-free, initial
# (total Lagrangian) configuration — mirrors Ratel's
# `elasticity-neo-hookean-isochoric-initial` model.
#
# Plane strain: F is embedded as 3x3 with F33=1, E33=0 (out-of-plane
# displacement is identically zero). This is what makes our 2D problem match
# Ratel's numbers, since Ratel has no native 2D solver — it always works with
# the true 3D isochoric split (E_dev = E - tr(E)/3 * I, J^(-2/3)). A genuinely
# 2D isochoric split (1/2 factor, J^(-1)) is a physically different model and
# would not agree with Ratel on the same nominal problem.
#
# Given grad_u = du/dX (2x2, plane strain in-plane block), C = F'F, E =
# (C-I)/2 are also block-diagonal (only the in-plane 2x2 block plus C33=1,
# E33=0 are nonzero), and since our 2D residual only ever contracts with
# in-plane test-function gradients, only the in-plane 2x2 blocks of every
# tensor below are ever needed.

bulk_modulus(lambda, mu) = lambda + 2 * mu / 3

"""
    NeoHookeanState

Per-quadrature-point state for isochoric-volumetric split Neo-Hookean
hyperelasticity, cached so [`neo_hookean_dS`](@ref) doesn't need to
recompute it for every basis-function direction in the Jacobian assembly.
"""
struct NeoHookeanState
    F::Matrix{Float64}
    S::Matrix{Float64}     # second Piola-Kirchhoff stress (symmetric)
    Cinv::Matrix{Float64}
    Edev::Matrix{Float64}
    trE::Float64
    Jm1::Float64
    J_pow::Float64         # J^(-2/3)
end

"""
    NeoHookeanState(lambda, mu, grad_u)

Compute the isochoric-volumetric split second Piola-Kirchhoff stress
`S = S_vol + S_iso` at a point, given `grad_u = du/dX` (2x2).

`S_vol = bulk * (J dV/dJ) * C_inv`, `S_iso = 2 mu J^(-2/3) * C_inv * E_dev`,
with `V(J) = (J^2 - 1 - 2 log J) / 4` (Ratel's stable convex volumetric
potential). `J - 1`, `J dV/dJ` are computed via the stable (cancellation-free)
forms in terms of `grad_u`, following Ratel's "Stable numerics for
finite-strain elasticity" convention.
"""
function NeoHookeanState(lambda, mu, grad_u)
    bulk = bulk_modulus(lambda, mu)
    F = grad_u + I

    u00, u01, u10, u11 = grad_u[1, 1], grad_u[1, 2], grad_u[2, 1], grad_u[2, 2]
    Jm1 = u00 + u11 + u00 * u11 - u01 * u10 # stable J - 1
    Jp1 = Jm1 + 2
    J = Jm1 + 1

    E = 0.5 * (grad_u + grad_u' + grad_u' * grad_u) # Green-Lagrange strain
    trE = E[1, 1] + E[2, 2]

    detC = J^2 # = det(C), computed stably via J rather than C's own cofactors
    C11, C12, C22 = 1 + 2E[1, 1], 2E[1, 2], 1 + 2E[2, 2]
    Cinv = [C22 -C12; -C12 C11] ./ detC

    Edev = E - (trE / 3) * I

    J_dVdJ = Jm1 * Jp1 / 2 # stable (J^2 - 1)/2
    J_pow = J^(-2/3)

    S_vol = (bulk * J_dVdJ) .* Cinv
    S_iso = (2 * mu * J_pow) .* (Cinv * Edev)

    return NeoHookeanState(F, S_vol + S_iso, Cinv, Edev, trE, Jm1, J_pow)
end

"""
    neo_hookean_dS(lambda, mu, st::NeoHookeanState, grad_du)

Directional derivative `dS = dS_vol + dS_iso` of the second Piola-Kirchhoff
stress at the state `st`, in direction `grad_du = d(du)/dX`.
"""
function neo_hookean_dS(lambda, mu, st::NeoHookeanState, grad_du)
    bulk = bulk_modulus(lambda, mu)
    Jm1, J = st.Jm1, st.Jm1 + 1
    J_dVdJ = Jm1 * (Jm1 + 2) / 2
    J2_d2VdJ2 = (J^2 + 1) / 2
    Cinv = st.Cinv

    dE = 0.5 * (grad_du' * st.F + st.F' * grad_du)
    trdE = dE[1, 1] + dE[2, 2]
    Cinv_dE = sum(Cinv .* dE) # C_inv : dE (full double contraction)
    dCinv = -2 .* (Cinv * dE * Cinv)

    I1C = 3 + 2 * st.trE
    Cinv_Edev = Cinv * st.Edev

    dS_iso = (-4 / 3 * mu * st.J_pow * Cinv_dE) .* Cinv_Edev .+
             (-1 / 3 * mu * st.J_pow) .* (2 * trdE .* Cinv .+ I1C .* dCinv)
    dS_vol = (bulk * (J2_d2VdJ2 + J_dVdJ) * Cinv_dE) .* Cinv .+ (bulk * J_dVdJ) .* dCinv

    return dS_vol + dS_iso
end

"""
    GradU(dXdx_Ti, Duq, i, Q2)

Reconstruct `grad_u = du/dX` (2x2) at quadrature point `i` from the
reference-gradient dof vector `Duq` (as laid out by `Bu.D * ue`, see
`Compute_f1`) and the pulled-back Jacobian `dXdx_Ti = (dX/dx)^T` at that point.
"""
function GradU(dXdx_Ti, Duq, i, Q2)
    grad_u1 = dXdx_Ti * [Duq[i], Duq[i+Q2]]
    grad_u2 = dXdx_Ti * [Duq[i+2*Q2], Duq[i+3*Q2]]
    return [grad_u1'; grad_u2']
end

"""
    Compute_f1_NeoHookean(lambda, mu, Q, wdetJ, dXdx_T, Duq)

Stress QFunction contribution `∫ ∇v : P dX` for isochoric Neo-Hookean
hyperelasticity, `P = F*S` (first Piola-Kirchhoff), mapped to reference-
element gradient space (mirrors `Compute_f1` for linear elasticity).
"""
function Compute_f1_NeoHookean(lambda, mu, Q, wdetJ, dXdx_T, Duq)
    s = zeros(4 * Q^2)
    for i = 1:Q^2
        grad_u = GradU(dXdx_T[i, :, :], Duq, i, Q^2)
        st = NeoHookeanState(lambda, mu, grad_u)
        P = st.F * st.S
        Pmap = dXdx_T[i, :, :]' * P' # row i of P pairs with component i's test gradient
        s[i] = wdetJ[i] * Pmap[1, 1]
        s[i+Q^2] = wdetJ[i] * Pmap[2, 1]
        s[i+2*Q^2] = wdetJ[i] * Pmap[1, 2]
        s[i+3*Q^2] = wdetJ[i] * Pmap[2, 2]
    end
    return s
end

"""
    HyperelasticResidual(lambda, mu, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc)

Matrix-free residual for 2D isochoric Neo-Hookean hyperelasticity (plane
strain), with a body force `f(lambda, mu, x, y)` and Dirichlet data `u_bc` on
dofs `bc_idx`.
"""
function HyperelasticResidual(lambda, mu, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, f, bc_idx, u_bc)
    u = copy(u_in)
    v = zero(u)
    u[bc_idx] .= u_bc

    num_elem = size(Ind.idx_u, 2)
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        q, wdetJ, dXdx_T = GetQdata(Coord_E, Bx)

        ue = Ind.Er_u[e] * u
        Duq = Bu.D * ue

        f0 = Compute_f0(lambda, mu, wdetJ, q, f)
        f1 = Compute_f1_NeoHookean(lambda, mu, Bu.Q, wdetJ, dXdx_T, Duq)
        ve = Bu.B' * f0 + Bu.D' * f1
        v += sparse(Ind.Er_u[e]') * ve
    end
    v[bc_idx] = u_in[bc_idx] - u[bc_idx]
    return v
end

"""
    HyperelasticJacobian(lambda, mu, u_in, xc, yc, Ind, Bx, Bu, bc_idx)

Tangent stiffness for 2D isochoric Neo-Hookean hyperelasticity, linearized
about the current state `u_in` (unlike linear elasticity, this genuinely
depends on `u_in`).
"""
function HyperelasticJacobian(lambda, mu, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, bc_idx)
    num_elem = size(Ind.idx_u, 2)
    rows, cols, vals = Int[], Int[], Float64[]
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        _, wdetJ, dXdx_T = GetQdata(Coord_E, Bx)

        ue = Ind.Er_u[e] * u_in
        Duq = Bu.D * ue
        states = [NeoHookeanState(lambda, mu, GradU(dXdx_T[i, :, :], Duq, i, Bu.Q^2)) for i = 1:Bu.Q^2]

        Jac = zeros(Bu.num_comp * Bu.P * Bu.P, Bu.num_comp * Bu.P * Bu.P)
        for c = 0:Bu.num_comp-1
            for j = 1:Bu.P*Bu.P
                Ddu = Bu.D[:, j+c*Bu.P*Bu.P]
                df1 = zeros(4 * Bu.Q^2)
                for i = 1:Bu.Q^2
                    grad_du = GradU(dXdx_T[i, :, :], Ddu, i, Bu.Q^2)
                    dS = neo_hookean_dS(lambda, mu, states[i], grad_du)
                    dP = grad_du * states[i].S + states[i].F * dS
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
