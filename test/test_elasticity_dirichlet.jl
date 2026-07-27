using NLsolve

function solve_elasticity_dirichlet(P, nx, ny, lambda, mu)
    Q, num_comp, Qmode = P, 2, "GAUSS"
    Bx = FEBasis(2, Q, 2, Qmode)
    Bu = FEBasis(P, Q, num_comp, Qmode)
    xc, yc = GetCoordMesh("uniform", nx, ny, 1)
    Ind = FEIndices(P, num_comp, nx, ny)

    bc_idx = GetDirichletBCsIndex(num_comp, Ind) # all 4 sides
    x = GetNodalCoordinate(xc, yc, Bu.Q, Ind)
    x_bc = reshape(x[bc_idx], :, 2)
    u_bc = uex(x_bc[:, 1], x_bc[:, 2])

    u_dof = num_comp * maximum(Ind.idx_u)
    u0 = zeros(u_dof)
    uh = nlsolve(u -> ElasticityResidual(lambda, mu, u, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc),
        u -> ElasticityJacobian(lambda, mu, xc, yc, Ind, Bx, Bu, bc_idx), u0; method=:newton)

    return GetL2Error(xc, yc, Ind, Bx, Bu, uh.zero, uex)
end

@testset "MMS, full Dirichlet BCs" begin
    P = 3
    lambda, mu = 3, 1.0
    res = [4, 8, 16]
    eu = [solve_elasticity_dirichlet(P, n, n, lambda, mu) for n in res]

    @test issorted(eu, rev=true) # error decreases monotonically under refinement
    order = log2(eu[end-1] / eu[end])
    @test order ≈ P atol = 0.5 # expect ~P-th order convergence
end
