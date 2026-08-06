using NLsolve

function make_contact_problem(P, nx, ny, side)
    Q, num_comp, Qmode = P, 2, "GAUSS"
    Bx = FEBasis(2, Q, 2, Qmode)
    Bu = FEBasis(P, Q, num_comp, Qmode)
    xc, yc = GetCoordMesh("uniform", nx, ny, 1)
    Ind = FEIndices(P, num_comp, nx, ny)
    Fx = FEFaceIndices(P, num_comp, nx, ny, Ind, side)
    FaceBu = febasis2D_face(P, Q, num_comp, Qmode, side)
    FaceBx = febasis2D_face(2, Q, 2, Qmode, side)
    return Bx, Bu, xc, yc, Ind, Fx, FaceBu, FaceBx
end

lambda0, mu0 = 3.0, 1.0 # f_zero comes from mms_fixtures.jl

@testset "Jacobian matches residual (finite differences)" begin
    # No BCs on the volume; a "bottom" contact face with a mix of penetrating
    # and separated quadrature points (both branches of the semismooth min()
    # exercised) at a generic (non-switching-point) trial displacement.
    Bx, Bu, xc, yc, Ind, Fx, FaceBu, FaceBx = make_contact_problem(2, 2, 2, "bottom")
    cp = ContactParams([0.0, 1.0], [0.0, 0.0], 100.0)
    bc_idx, u_bc = Int[], Float64[]

    u_dof = Bu.num_comp * maximum(Ind.idx_u)
    u0 = 0.02 .* randn(u_dof)
    du = randn(u_dof)
    du ./= norm(du)

    J0 = ContactJacobian(lambda0, mu0, cp, u0, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, bc_idx)
    Jdu = J0 * du

    h = 1e-6
    Rp = ContactResidual(lambda0, mu0, cp, u0 .+ h .* du, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, f_zero, bc_idx, u_bc)
    Rm = ContactResidual(lambda0, mu0, cp, u0 .- h .* du, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, f_zero, bc_idx, u_bc)
    Jdu_fd = (Rp .- Rm) ./ (2h)

    @test norm(Jdu_fd .- Jdu) / norm(Jdu) < 1e-6
end

@testset "Homogeneous touching-contact patch test" begin
    # u = (F0-I)*X with F0 = diag(1, 1+e22) is zero at y=0 for ANY e22, so a
    # body resting exactly on a platen at y=0 (g0=0) stays exactly touching
    # (g=0) under this deformation, for any compressive e22 and any gamma
    # (since gamma*g = gamma*0 = 0 regardless of gamma). With sigma_n < 0,
    # the Nitsche pressure hat_sigma_n = min(sigma_n+gamma*0, 0) = sigma_n
    # exactly matches the true compressive contact reaction, so this should
    # be an exact patch test (residual = 0 at every free dof), same as the
    # existing homogeneous-deformation patch tests for the other models.
    Bx, Bu, xc, yc, Ind, Fx, FaceBu, FaceBx = make_contact_problem(2, 4, 4, "bottom")
    F0 = [1.0 0.0; 0.0 0.92]
    Hmat = F0 - I

    bc_idx = GetDirichletBCsIndex(Bu.num_comp, Ind, (:right, :top, :left)) # bottom -> contact
    x = GetNodalCoordinate(xc, yc, Bu.Q, Ind)
    num_nodes = maximum(Ind.idx_u)
    X, Y = x[1:num_nodes], x[num_nodes+1:end]

    u_dof = Bu.num_comp * num_nodes
    u_exact = zeros(u_dof)
    u_exact[1:num_nodes] = Hmat[1, 1] .* X .+ Hmat[1, 2] .* Y
    u_exact[num_nodes+1:end] = Hmat[2, 1] .* X .+ Hmat[2, 2] .* Y
    u_bc = u_exact[bc_idx]
    interior = setdiff(1:u_dof, bc_idx)

    for gamma in (1.0, 1e6) # patch test must hold regardless of gamma (g=0 exactly)
        cp = ContactParams([0.0, 1.0], [0.0, 0.0], gamma)
        R = ContactResidual(lambda0, mu0, cp, u_exact, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, f_zero, bc_idx, u_bc)
        @test maximum(abs.(R[interior])) < 1e-10
    end
end

@testset "Fully-separated limit reduces to plain elasticity" begin
    # A platen far below the body (g0 >> 0 everywhere) never activates
    # (hat_sigma_n = 0 identically), so ContactResidual/Jacobian must exactly
    # reproduce plain ElasticityResidual/Jacobian with the same (bottom-free) BCs.
    Bx, Bu, xc, yc, Ind, Fx, FaceBu, FaceBx = make_contact_problem(2, 3, 3, "bottom")
    cp = ContactParams([0.0, 1.0], [0.0, -100.0], 1.0)

    bc_idx = GetDirichletBCsIndex(Bu.num_comp, Ind, (:right, :top, :left))
    u_bc = zeros(length(bc_idx))
    u_dof = Bu.num_comp * maximum(Ind.idx_u)
    u0 = 0.01 .* randn(u_dof)
    u0[bc_idx] .= 0.0

    Rc = ContactResidual(lambda0, mu0, cp, u0, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, f_zero, bc_idx, u_bc)
    Re = ElasticityResidual(lambda0, mu0, u0, xc, yc, Ind, Bx, Bu, f_zero, bc_idx, u_bc)
    @test Rc == Re

    Jc = ContactJacobian(lambda0, mu0, cp, u0, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, bc_idx)
    Je = ElasticityJacobian(lambda0, mu0, xc, yc, Ind, Bx, Bu, bc_idx)
    @test Jc == Je
end

@testset "Load-stepping Newton solve: no penetration" begin
    # A block clamped (and pushed down) on top, free on the sides, resting on
    # a platen at y=0 (contact on bottom): Newton must converge at each step,
    # and the bottom face must stay at y >= 0 (up to the Nitsche penalty's
    # regularization tolerance) despite the increasing downward push.
    Bx, Bu, xc, yc, Ind, Fx, FaceBu, FaceBx = make_contact_problem(2, 4, 3, "bottom")
    E = mu0 * (3lambda0 + 2mu0) / (lambda0 + mu0)
    cp = ContactParams([0.0, 1.0], [0.0, 0.0], 200 * E) # Ratel's gamma ~ 200E rule of thumb

    bc_idx = GetDirichletBCsIndex(Bu.num_comp, Ind, (:top,))
    num_nodes = maximum(Ind.idx_u)
    n_top = length(bc_idx) ÷ 2
    push_down = -0.05

    u_dof = Bu.num_comp * num_nodes
    u = zeros(u_dof)
    for s = 1:3
        frac = s / 3
        u_bc = zeros(length(bc_idx))
        u_bc[n_top+1:end] .= frac * push_down # y-component half of bc_idx
        result = nlsolve(
            uu -> ContactResidual(lambda0, mu0, cp, uu, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, f_zero, bc_idx, u_bc),
            uu -> ContactJacobian(lambda0, mu0, cp, uu, xc, yc, Ind, Bx, Bu, FaceBu, FaceBx, Fx, bc_idx),
            u; method=:newton)
        @test result.f_converged
        u = result.zero
    end

    uy_bottom = u[num_nodes+1:end][Ind.idx_b]
    @test minimum(uy_bottom) > -1e-4 # negligible penetration (Nitsche regularization tolerance)
end
