module FESolid2D

using LinearAlgebra
using SparseArrays
using Random
using FastGaussQuadrature

include("basis.jl")
include("mesh.jl")
include("bcs.jl")
include("elasticity.jl")
include("neohookean.jl")

export vander_legendre_deriv, febasis1D, febasis2D, FEBasis

export GetConnectivity, FEIndices, GetCoordMesh, GetCoordElem, GetNodalCoordinate, GetQdata,
    GetFaceLocalIndex, GetBoundaryElements, FEFaceIndices, GetFaceQdata,
    GetTractionLocal, GetTractionGlobal

export GetDirichletBCsIndex

export Compute_f0, Compute_f1, Compute_df1,
    ElasticityResidual, ElasticityJacobian, ElasticityResidualNeumann,
    GetL2Error, GetL2ErrorDisc

export bulk_modulus, NeoHookeanState, neo_hookean_dS,
    Compute_f1_NeoHookean, HyperelasticResidual, HyperelasticJacobian

end # module FESolid2D
