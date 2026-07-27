# These are like libCEED/Ratel Residual/Jacobian QFunctions for 2D linear
# (Cauchy) elasticity: sigma = lambda*tr(eps)*I + 2*mu*eps, eps = sym(grad(u)).
# Assumes 2-component (num_comp=2) displacement fields, as required by 2D elasticity.

"""
    Compute_f0(lambda, mu, wdetJ, q, f)

Body-force QFunction contribution `-∫ v·f dx` (like a libCEED/Ratel f0).
`f(lambda, mu, x, y)` returns the body force stacked as `[fx; fy]` at the
quadrature points `q`.
"""
function Compute_f0(lambda, mu, wdetJ, q, f)
    W2 = diagm([wdetJ; wdetJ])
    Q2 = length(wdetJ)
    xq = reshape(q, Q2, :)
    return -W2 * f(lambda, mu, xq[:, 1], xq[:, 2])
end

"""
    Compute_f1(lambda, mu, Q, wdetJ, dXdx_T, Duq)

Stress QFunction contribution `∫ ∇v : σ(u) dx` (like a libCEED/Ratel f1).
`Duq` is `∇u` in reference coordinates at the `Q^2` quadrature points.
"""
function Compute_f1(lambda, mu, Q, wdetJ, dXdx_T, Duq)
    W2 = diagm([wdetJ; wdetJ; wdetJ; wdetJ])
    s = zeros(4 * Q^2)
    for i = 1:Q^2
        grad_u1 = dXdx_T[i, :, :] * [Duq[i], Duq[i+Q^2]]
        grad_u2 = dXdx_T[i, :, :] * [Duq[i+2*Q^2], Duq[i+3*Q^2]]
        tr_e = grad_u1[1] + grad_u2[2]
        s11 = lambda * tr_e + 2 * mu * grad_u1[1]
        s12 = mu * (grad_u1[2] + grad_u2[1])
        s22 = lambda * tr_e + 2 * mu * grad_u2[2]
        S1 = dXdx_T[i, :, :]' * vcat(s11', s12')
        S2 = dXdx_T[i, :, :]' * vcat(s12', s22')

        s[i] = S1[1]
        s[i+Q^2] = S1[2]
        s[i+2*Q^2] = S2[1]
        s[i+3*Q^2] = S2[2]
    end
    return W2 * s
end

"""
    Compute_df1(lambda, mu, Q, wdetJ, dXdx_T, Ddu)

Directional derivative of [`Compute_f1`](@ref) w.r.t. `u`, in direction
`Ddu = ∇(basis function)`. Linear elasticity's tangent doesn't depend on the
current state, so unlike `f1` this takes no `Duq`.
"""
function Compute_df1(lambda, mu, Q, wdetJ, dXdx_T, Ddu)
    W2 = diagm([wdetJ; wdetJ; wdetJ; wdetJ])
    ds = zeros(4 * Q^2)
    for i = 1:Q^2
        grad_du1 = dXdx_T[i, :, :] * [Ddu[i], Ddu[i+Q^2]]
        grad_du2 = dXdx_T[i, :, :] * [Ddu[i+2*Q^2], Ddu[i+3*Q^2]]
        tr_de = grad_du1[1] + grad_du2[2]
        ds11 = lambda * tr_de + 2 * mu * grad_du1[1]
        ds12 = mu * (grad_du1[2] + grad_du2[1])
        ds22 = lambda * tr_de + 2 * mu * grad_du2[2]
        dS1 = dXdx_T[i, :, :]' * vcat(ds11', ds12')
        dS2 = dXdx_T[i, :, :]' * vcat(ds12', ds22')

        ds[i] = dS1[1]
        ds[i+Q^2] = dS1[2]
        ds[i+2*Q^2] = dS2[1]
        ds[i+3*Q^2] = dS2[2]
    end
    return W2 * ds
end

"""
    ElasticityResidual(lambda, mu, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc)

Matrix-free residual for 2D linear elasticity with a manufactured body force
`f(lambda, mu, x, y)` and Dirichlet data `u_bc` on dofs `bc_idx`.
"""
function ElasticityResidual(lambda, mu, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, f, bc_idx, u_bc)
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
        f1 = Compute_f1(lambda, mu, Bu.Q, wdetJ, dXdx_T, Duq)
        ve = Bu.B' * f0 + Bu.D' * f1
        v += sparse(Ind.Er_u[e]') * ve
    end
    v[bc_idx] = u_in[bc_idx] - u[bc_idx]
    return v
end

"""
    ElasticityJacobian(lambda, mu, xc, yc, Ind, Bx, Bu, bc_idx)

Tangent stiffness for 2D linear elasticity. Constant (independent of `u`),
consistent with [`ElasticityResidual`](@ref).
"""
function ElasticityJacobian(lambda, mu, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, bc_idx)
    num_elem = size(Ind.idx_u, 2)
    rows, cols, vals = Int[], Int[], Float64[]
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        _, wdetJ, dXdx_T = GetQdata(Coord_E, Bx)

        Jac = zeros(Bu.num_comp * Bu.P * Bu.P, Bu.num_comp * Bu.P * Bu.P)
        for c = 0:Bu.num_comp-1
            for j = 1:Bu.P*Bu.P
                Ddu = Bu.D[:, j+c*Bu.P*Bu.P]
                df1 = Compute_df1(lambda, mu, Bu.Q, wdetJ, dXdx_T, Ddu)
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
    ElasticityResidualNeumann(lambda, mu, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc, Ft)

[`ElasticityResidual`](@ref) with an additional (constant, precomputed)
traction load vector `Ft` from [`GetTractionGlobal`](@ref) subtracted in.
"""
function ElasticityResidualNeumann(lambda, mu, u_in, xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, f, bc_idx, u_bc, Ft)
    v = ElasticityResidual(lambda, mu, u_in, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc)
    v .-= vec(Ft)
    v[bc_idx] = u_in[bc_idx] - u_bc
    return v
end

"""
    GetL2Error(xc, yc, Ind, Bx, Bu, uh, uex)

L2 error `sqrt(∑_e ∫_e (uh - uex)^2 dx)`, evaluated pointwise at quadrature
points.
"""
function GetL2Error(xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, uh, uex)
    num_elem = size(Ind.idx_u, 2)
    e_u = zeros(num_elem)
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        q, wdetJ = GetQdata(Coord_E, Bx)
        wdetJ = [wdetJ; wdetJ]

        dh = Ind.Er_u[e] * uh
        u_num = Bu.B * dh

        xq = reshape(q, Bx.Q^2, :)
        u_ex = uex(xq[:, 1], xq[:, 2])

        err_u = vec(u_num - u_ex)
        e_u[e] = wdetJ' * (err_u .* err_u)
    end
    return sqrt(sum(e_u))
end

"""
    GetL2ErrorDisc(xc, yc, Ind, Bx, Bu, uh, uex)

L2 error computed via the discrete (Galerkin-projected) norm, as in Ratel:
`sqrt(∑_e (v, (uh - uex)^2))`.
"""
function GetL2ErrorDisc(xc, yc, Ind::FEIndices, Bx::FEBasis, Bu::FEBasis, uh, uex)
    num_elem = size(Ind.idx_u, 2)
    global_dof = Bu.num_comp * maximum(Ind.idx_u)
    e_u = spzeros(global_dof, 1)

    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        q, wdetJ = GetQdata(Coord_E, Bx)

        dh = Ind.Er_u[e] * uh
        u_num = Bu.B * dh

        xq = reshape(q, Bx.Q^2, :)
        u_ex = uex(xq[:, 1], xq[:, 2])

        err_u = vec(u_num - u_ex)
        W2 = diagm([wdetJ; wdetJ])
        er = Bu.B' * W2 * (err_u .* err_u)

        e_u = e_u + Ind.Er_u[e]' * er
    end
    return sqrt(sum(e_u))
end
