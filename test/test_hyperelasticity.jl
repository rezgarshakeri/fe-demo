using NLsolve

function make_hyperelastic_problem(P, nx, ny)
    Q, num_comp, Qmode = P, 2, "GAUSS"
    Bx = FEBasis(2, Q, 2, Qmode)
    Bu = FEBasis(P, Q, num_comp, Qmode)
    xc, yc = GetCoordMesh("uniform", nx, ny, 1)
    Ind = FEIndices(P, num_comp, nx, ny)
    return Bx, Bu, xc, yc, Ind
end

f_zero(lambda, mu, x, y) = zeros(2 * length(x))

@testset "Jacobian matches residual (finite differences)" begin
    # No BCs: exercises the raw nonlinear operator, independent of BC bookkeeping
    # (already validated separately for the linear elasticity case).
    Bx, Bu, xc, yc, Ind = make_hyperelastic_problem(2, 2, 2)
    lambda, mu = 3.0, 1.0
    u_dof = Bu.num_comp * maximum(Ind.idx_u)
    bc_idx, u_bc = Int[], Float64[]

    rng_u0 = 0.02 .* randn(u_dof)
    du = randn(u_dof)
    du ./= norm(du)

    J0 = HyperelasticJacobian(lambda, mu, rng_u0, xc, yc, Ind, Bx, Bu, bc_idx)
    Jdu = J0 * du

    h = 1e-6
    Rp = HyperelasticResidual(lambda, mu, rng_u0 .+ h .* du, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc)
    Rm = HyperelasticResidual(lambda, mu, rng_u0 .- h .* du, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc)
    Jdu_fd = (Rp .- Rm) ./ (2h)

    @test norm(Jdu_fd .- Jdu) / norm(Jdu) < 1e-8
end

@testset "Homogeneous deformation patch test" begin
    # For ANY material law, a constant deformation gradient F0 (u = (F0-I)*X)
    # exactly satisfies equilibrium (div P = 0) with zero body force, since P
    # is constant. A consistent FE discretization must reproduce this exactly
    # at every interior dof.
    Bx, Bu, xc, yc, Ind = make_hyperelastic_problem(2, 4, 4)
    lambda, mu = 3.0, 1.0
    F0 = [1.15 0.08; 0.04 0.92]
    Hmat = F0 - I

    bc_idx = GetDirichletBCsIndex(Bu.num_comp, Ind) # all 4 sides
    x = GetNodalCoordinate(xc, yc, Bu.Q, Ind)
    num_nodes = maximum(Ind.idx_u)
    X, Y = x[1:num_nodes], x[num_nodes+1:end]

    u_dof = Bu.num_comp * num_nodes
    u_exact = zeros(u_dof)
    u_exact[1:num_nodes] = Hmat[1, 1] .* X .+ Hmat[1, 2] .* Y
    u_exact[num_nodes+1:end] = Hmat[2, 1] .* X .+ Hmat[2, 2] .* Y
    u_bc = u_exact[bc_idx]

    R = HyperelasticResidual(lambda, mu, u_exact, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc)
    interior = setdiff(1:u_dof, bc_idx)

    @test maximum(abs.(R[interior])) < 1e-12
end

@testset "Small-strain limit matches linear elasticity" begin
    # At tiny strain, Neo-Hookean -> St. Venant-Kirchhoff -> linear elasticity,
    # so solving with the SAME tiny-amplitude MMS load used for linear
    # elasticity (test_elasticity_dirichlet.jl) should land close to (but not
    # exactly at, since it solves a different PDE) the linear MMS solution.
    P = 3
    Bx, Bu, xc, yc, Ind = make_hyperelastic_problem(P, 4, 4)
    lambda, mu = 3.0, 1.0

    bc_idx = GetDirichletBCsIndex(Bu.num_comp, Ind)
    x = GetNodalCoordinate(xc, yc, Bu.Q, Ind)
    x_bc = reshape(x[bc_idx], :, 2)
    u_bc = uex(x_bc[:, 1], x_bc[:, 2])

    u_dof = Bu.num_comp * maximum(Ind.idx_u)
    u0 = zeros(u_dof)
    result = nlsolve(u -> HyperelasticResidual(lambda, mu, u, xc, yc, Ind, Bx, Bu, f, bc_idx, u_bc),
        u -> HyperelasticJacobian(lambda, mu, u, xc, yc, Ind, Bx, Bu, bc_idx), u0; method=:newton)

    @test result.f_converged
    L2_error = GetL2Error(xc, yc, Ind, Bx, Bu, result.zero, uex)
    @test L2_error < 1e-4 # close to the linear-elasticity MMS error (~3.5e-5) at this mesh
end
