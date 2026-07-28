using NLsolve

function make_plasticity_problem(P, nx, ny)
    Q, num_comp, Qmode = P, 2, "GAUSS"
    Bx = FEBasis(2, Q, 2, Qmode)
    Bu = FEBasis(P, Q, num_comp, Qmode)
    xc, yc = GetCoordMesh("uniform", nx, ny, 1)
    Ind = FEIndices(P, num_comp, nx, ny)
    return Bx, Bu, xc, yc, Ind
end

lambda0, mu0 = 3.0, 1.0
bulk0 = lambda0 + 2mu0 / 3
params_elastic = PlasticityParams(bulk0, mu0, 1e6, 0.0, 1e6, 1.0)      # sigma_0 huge: never yields
params_yield = PlasticityParams(bulk0, mu0, 0.01, 0.5, 0.05, 5.0)      # sigma_0 tiny: yields readily

@testset "Jacobian matches residual (finite differences)" begin
    # No BCs: exercises the raw nonlinear operator, independent of BC bookkeeping.
    Bx, Bu, xc, yc, Ind = make_plasticity_problem(2, 2, 2)
    u_dof = Bu.num_comp * maximum(Ind.idx_u)
    bc_idx, u_bc = Int[], Float64[]
    states = fill(VirginPlasticityState(), Bu.Q^2, size(Ind.idx_u, 2))

    u0 = 0.05 .* randn(u_dof)
    du = randn(u_dof)
    du ./= norm(du)

    J0 = PlasticityJacobian(params_yield, u0, xc, yc, Ind, Bx, Bu, bc_idx, states)
    Jdu = J0 * du

    h = 1e-6
    Rp = PlasticityResidual(params_yield, u0 .+ h .* du, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc, states)
    Rm = PlasticityResidual(params_yield, u0 .- h .* du, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc, states)
    Jdu_fd = (Rp .- Rm) ./ (2h)

    @test norm(Jdu_fd .- Jdu) / norm(Jdu) < 1e-6
end

@testset "Homogeneous deformation patch test" begin
    # For ANY material law (elastic or actively yielding), a constant deformation
    # gradient F0 (u = (F0-I)*X) exactly satisfies equilibrium (div P = 0) with
    # zero body force, since P is constant everywhere.
    Bx, Bu, xc, yc, Ind = make_plasticity_problem(2, 4, 4)
    states = fill(VirginPlasticityState(), Bu.Q^2, size(Ind.idx_u, 2))

    F0 = [1.05 0.02; 0.01 0.97]
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
    interior = setdiff(1:u_dof, bc_idx)

    R_elastic = PlasticityResidual(params_elastic, u_exact, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc, states)
    @test maximum(abs.(R_elastic[interior])) < 1e-12

    R_yield = PlasticityResidual(params_yield, u_exact, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc, states)
    @test maximum(abs.(R_yield[interior])) < 1e-12
end

@testset "Load-stepping Newton solve and state advance" begin
    # Never-yielding material: accumulated plastic strain must stay exactly 0
    # after advancing state, regardless of load (a regression/sanity check on
    # the elastic branch of AdvancePlasticityState).
    Bx, Bu, xc, yc, Ind = make_plasticity_problem(2, 3, 3)
    num_elem = size(Ind.idx_u, 2)
    states = fill(VirginPlasticityState(), Bu.Q^2, num_elem)

    bc_idx = GetDirichletBCsIndex(Bu.num_comp, Ind, (:left,)) # clamp left
    u_bc = zeros(length(bc_idx))

    Fx = FEFaceIndices(2, Bu.num_comp, 3, 3, Ind, "right")
    t_right(x, y) = [0.3 .* ones(length(x)); zeros(length(x))]
    Ft_full = GetTractionGlobal(xc, yc, Fx, Ind, Bu, t_right)

    u_dof = Bu.num_comp * maximum(Ind.idx_u)
    u = zeros(u_dof)
    for s = 1:3
        frac = s / 3
        Ft = frac .* Ft_full
        result = nlsolve(
            uu -> PlasticityResidualNeumann(params_elastic, uu, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc, states, Ft),
            uu -> PlasticityJacobian(params_elastic, uu, xc, yc, Ind, Bx, Bu, bc_idx, states),
            u; method=:newton)
        @test result.f_converged
        u = result.zero
        states = AdvancePlasticityState(params_elastic, u, xc, yc, Ind, Bx, Bu, states)
    end

    @test all(s -> s.accumulated_plastic == 0.0, states)
end
