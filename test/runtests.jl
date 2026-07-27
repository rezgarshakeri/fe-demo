using FESolid2D
using LinearAlgebra
using Random
using Test

Random.seed!(42)

@testset verbose = true "FESolid2D.jl" begin
    @testset verbose = true "Basis functions" begin
        include("test_basis.jl")
    end
    # Future test files, each with its own top-level testset:
    #   @testset "Linear elasticity: MMS, Dirichlet BCs" begin
    #       include("test_elasticity_dirichlet.jl")
    #   end
    #   @testset "Linear elasticity: MMS, Dirichlet + traction BCs" begin
    #       include("test_elasticity_traction.jl")
    #   end
end
