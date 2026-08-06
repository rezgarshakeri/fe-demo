# Frictionless rigid flat-platen contact via the Nitsche method (theta=0),
# matching Ratel's boundary/contact formulation
# (~/ratel/doc/modeling/material-models/boundary-conditions/contact.md)
# restricted to the simplest case: linear elasticity, no friction, a flat
# (non-rotating) platen. Full derivation in notebooks/Contact2D.ipynb.
#
# Unlike a traction BC (a fixed function, assembled once via
# GetTractionGlobal), the contact face residual/Jacobian depend on the
# current trial displacement (gap, normal stress, and the active-set
# indicator are all functions of u) and must be recomputed every Newton iteration

"""
    ContactParams(n_p, c, gamma)

Rigid flat platen: a point `c` and a constant outward unit normal `n_p`
(pointing away from the platen, into the body), plus the Nitsche parameter
`gamma > 0` (Ratel's rule of thumb: `gamma ~ 200*E`, `E` = Young's modulus).
"""
struct ContactParams
    n_p::Vector{Float64}
    c::Vector{Float64}
    gamma::Float64
end

"""
    GapFunction0(cp, xq, yq)

Initial (reference-configuration) gap `g0 = n_p . (X - c)` at physical points
`(xq, yq)`. The full gap is `g(u) = n_p . u + g0`.
"""
GapFunction0(cp::ContactParams, xq, yq) = cp.n_p[1] .* (xq .- cp.c[1]) .+ cp.n_p[2] .* (yq .- cp.c[2])

"""
    ContactStressOnly(lambda, mu, grad_u)

Cauchy stress for linear elasticity, `sigma = lambda*tr(eps)*I + 2*mu*eps`,
`eps = sym(grad_u)`, given `grad_u = du/dX` (2x2). Differentiated by Enzyme.jl
in [`ContactFaceJacobian`](@ref).
"""
function ContactStressOnly(lambda, mu, grad_u)
    tr_e = grad_u[1, 1] + grad_u[2, 2]
    s11 = lambda * tr_e + 2 * mu * grad_u[1, 1]
    s12 = mu * (grad_u[1, 2] + grad_u[2, 1])
    s22 = lambda * tr_e + 2 * mu * grad_u[2, 2]
    return SMatrix{2,2}(s11, s12, s12, s22)
end

"""
    ContactHatSigmaAt(cp, lambda, mu, ue, dXdx_Ti, g0i, FaceBu, i)

Nitsche-penalized normal contact pressure `hat_sigma_n = min(sigma_n + gamma*g, 0)`
at ONE face quadrature point `i`, given the FULL element solution dofs `ue`
(all `num_comp*P^2` of them -- the gradient at a face point depends on every
basis function of the element, not just the face-local ones). Written as a
scalar-returning function of `ue` (rather than mutating/returning a `Vector`)
specifically so it can be Enzyme-differentiated per point in
[`ContactJacobian`](@ref) -- matching `PlasticityStressOnly`'s pattern in
`plasticity.jl`, this avoids an Enzyme "runtime activity" error that a
mutated-array return triggers.
"""
function ContactHatSigmaAt(cp::ContactParams, lambda, mu, ue, dXdx_Ti, g0i, FaceBu::FEFaceBasis, i::Int)
    Qf = FaceBu.Q
    Pu2 = FaceBu.P^2
    ue_x = ue[1:Pu2]
    ue_y = ue[Pu2+1:2*Pu2]

    Duq = vcat(FaceBu.D1 * ue_x, FaceBu.D1 * ue_y) # [dux/dxi;dux/deta;duy/dxi;duy/deta]
    ux_i = (FaceBu.B1*ue_x)[i]
    uy_i = (FaceBu.B1*ue_y)[i]

    grad_u = SMatrix{2,2}(GradU(dXdx_Ti, Duq, i, Qf))
    sigma = ContactStressOnly(lambda, mu, grad_u)
    sigma_n = cp.n_p' * sigma * cp.n_p
    g = cp.n_p[1] * ux_i + cp.n_p[2] * uy_i + g0i
    return min(sigma_n + cp.gamma * g, 0.0)
end

"""
    ContactFaceLocalHatSigma(cp, lambda, mu, ue, dXdx_T_face, g0, FaceBu)

`hat_sigma_n` (see [`ContactHatSigmaAt`](@ref)) at all `FaceBu.Q` face
quadrature points, for the (non-differentiated) residual assembly.
"""
function ContactFaceLocalHatSigma(cp::ContactParams, lambda, mu, ue, dXdx_T_face, g0, FaceBu::FEFaceBasis)
    return [ContactHatSigmaAt(cp, lambda, mu, ue, dXdx_T_face[i, :, :], g0[i], FaceBu, i) for i = 1:FaceBu.Q]
end

"""
    ContactFaceResidual(cp, lambda, mu, u, xc, yc, Ind, FaceBu, FaceBx, Fx)

Global (`num_comp*num_nodes`-length) contact face contribution
`∫_Γc (v . n_p) hat_sigma_n dS`, to be ADDED to the elasticity residual (see
module notes for the sign).
"""
function ContactFaceResidual(cp::ContactParams, lambda, mu, u, xc, yc, Ind::FEIndices,
    FaceBu::FEFaceBasis, FaceBx::FEFaceBasis, Fx::FEFaceIndices)
    global_dof = FaceBu.num_comp * maximum(Ind.idx_u)
    v = spzeros(global_dof, 1)
    for k in eachindex(Fx.elems)
        e = Fx.elems[k]
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        dXdx_T_face = GetQdataFace(Coord_E, FaceBx)

        Coord_face = Fx.Er_face_x[k] * [xc; yc]
        xq, yq, wds = GetFaceQdata(Coord_face, FaceBu.Q, FaceBu.Qmode)
        g0 = GapFunction0(cp, xq, yq)

        ue = Ind.Er_u[e] * u
        hat_sigma_n = ContactFaceLocalHatSigma(cp, lambda, mu, ue, dXdx_T_face, g0, FaceBu)

        re1 = FaceBu.B1' * (wds .* hat_sigma_n)
        re = vcat(cp.n_p[1] .* re1, cp.n_p[2] .* re1)

        v += Ind.Er_u[e]' * re
    end
    return v
end

"""
    ContactResidual(lambda, mu, cp, u_in, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, f, bc_idx, u_bc)

[`ElasticityResidual`](@ref) with the frictionless rigid-platen contact
contribution ADDED (not subtracted -- see module notes and
notebooks/Contact2D.ipynb section 3 for the sign derivation).
"""
function ContactResidual(lambda, mu, cp::ContactParams, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis,
    FaceBu::FEFaceBasis, FaceBx::FEFaceBasis, Fx::FEFaceIndices, f, bc_idx, u_bc)
    v = ElasticityResidual(lambda, mu, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc)
    u = copy(u_in)
    u[bc_idx] .= u_bc
    v .+= vec(ContactFaceResidual(cp, lambda, mu, u, xc, yc, Ind, FaceBu, FaceBx, Fx))
    v[bc_idx] = u_in[bc_idx] - u_bc
    return v
end

"""
    ContactJacobian(lambda, mu, cp, u_in, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, bc_idx)

Tangent stiffness for linear elasticity + frictionless rigid-platen contact:
[`ElasticityJacobian`](@ref)'s constant volume block plus a state-dependent
face contribution, obtained via Enzyme.jl forward-mode AD through the
semismooth `min(sigma_n + gamma*g, 0)` Nitsche projection (see module notes
-- the flat platen's constant normal means no shape-curvature terms, unlike
Ratel's general curved-shape linearization).
"""
function ContactJacobian(lambda, mu, cp::ContactParams, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis,
    FaceBu::FEFaceBasis, FaceBx::FEFaceBasis, Fx::FEFaceIndices, bc_idx)
    A = ElasticityJacobian(lambda, mu, xc, yc, Ind, Bx, Bu, bc_idx)

    Pu2 = FaceBu.P^2
    rows, cols, vals = Int[], Int[], Float64[]
    for k in eachindex(Fx.elems)
        e = Fx.elems[k]
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        dXdx_T_face = GetQdataFace(Coord_E, FaceBx)

        Coord_face = Fx.Er_face_x[k] * [xc; yc]
        xq, yq, wds = GetFaceQdata(Coord_face, FaceBu.Q, FaceBu.Qmode)
        g0 = GapFunction0(cp, xq, yq)

        ue = Ind.Er_u[e] * u_in

        # Build the local Jacobian one column at a time: due is a one-hot
        # direction (nudge local dof j by 1), and Duplicated(ue, due) makes
        # Enzyme propagate that nudge through ContactHatSigmaAt via forward-
        # mode AD, so dhat_i = exact d(hat_sigma_n)/d(ue[j]) at point i (not
        # a finite-difference approximation). set_runtime_activity works
        # around an Enzyme limitation, not a math concern: ContactHatSigmaAt
        # slices the heap Vector `ue` (ue_x = ue[1:Pu2]), and Enzyme's
        # compile-time activity analysis can't always prove whether such a
        # slice needs derivative tracking, throwing EnzymeRuntimeActivityError
        # without this, it defers that check to runtime instead (Enzyme's
        # own documented fix), verified harmless via the FD-Jacobian test.
        Jac = zeros(2 * Pu2, 2 * Pu2)
        for j = 1:2*Pu2
            due = zeros(2 * Pu2)
            due[j] = 1.0
            dre1 = zeros(Pu2)
            for i = 1:FaceBu.Q
                dhat_i = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward), ContactHatSigmaAt,
                    Enzyme.Const(cp), Enzyme.Const(lambda), Enzyme.Const(mu),
                    Enzyme.Duplicated(ue, due), Enzyme.Const(dXdx_T_face[i, :, :]), Enzyme.Const(g0[i]),
                    Enzyme.Const(FaceBu), Enzyme.Const(i))[1]
                dre1 .+= FaceBu.B1[i, :] .* (wds[i] * dhat_i)
            end
            Jac[:, j] = vcat(cp.n_p[1] .* dre1, cp.n_p[2] .* dre1)
        end

        inds = rowvals(sparse(Ind.Er_u[e]'))
        append!(rows, kron(ones(2 * Pu2), inds))
        append!(cols, kron(inds, ones(2 * Pu2)))
        append!(vals, vec(Jac))
    end
    A += sparse(rows, cols, vals, size(A, 1), size(A, 2))
    A[bc_idx, :] .= 0.0
    A[:, bc_idx] .= 0.0
    for i in eachindex(bc_idx)
        A[bc_idx[i], bc_idx[i]] = 1.0
    end
    return A
end
