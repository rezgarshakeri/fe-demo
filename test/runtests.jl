using FESolid2D
using LinearAlgebra
using Random
using Test

Random.seed!(42)

@testset verbose = true "FESolid2D.jl" begin
    @testset verbose = true "Basis functions" begin
        include("test_basis.jl")
    end
    @testset verbose = true "Linear elasticity" begin
        include("mms_fixtures.jl")
        include("test_elasticity_dirichlet.jl")
        include("test_elasticity_traction.jl")
    end
    @testset verbose = true "Hyperelasticity" begin
        # relies on uex/f from mms_fixtures.jl, already loaded above
        include("test_hyperelasticity.jl")
    end
end
