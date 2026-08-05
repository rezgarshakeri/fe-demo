module FESolid2D

using LinearAlgebra
using SparseArrays
using Random
using FastGaussQuadrature
using Enzyme
using StaticArrays

include("basis.jl")
include("mesh.jl")
include("bcs.jl")
include("elasticity.jl")
include("neohookean.jl")
include("plasticity.jl")
include("contact.jl")

export vander_legendre_deriv, febasis1D, febasis2D, FEBasis, feval1D_at, febasis2D_face, FEFaceBasis

export GetConnectivity, FEIndices, GetCoordMesh, GetCoordElem, GetNodalCoordinate, GetQdata, GetQdataFace,
    GetFaceLocalIndex, GetBoundaryElements, FEFaceIndices, GetFaceQdata,
    GetTractionLocal, GetTractionGlobal

export GetDirichletBCsIndex

export Compute_f0, Compute_f1, Compute_df1,
    ElasticityResidual, ElasticityJacobian, ElasticityResidualNeumann,
    GetL2Error, GetL2ErrorDisc

export bulk_modulus, NeoHookeanState, neo_hookean_dS,
    Compute_f1_NeoHookean, HyperelasticResidual, HyperelasticJacobian, HyperelasticResidualNeumann

export PlasticityParams, PlasticityState, VirginPlasticityState, Eig2x2Sym, ReturnMappingDeltaGamma,
    PlasticityLocal, PlasticityStressOnly, PlasticityResidual, PlasticityResidualNeumann,
    PlasticityJacobian, AdvancePlasticityState

export ContactParams, GapFunction0, ContactStressOnly, ContactHatSigmaAt, ContactFaceLocalHatSigma,
    ContactFaceResidual, ContactResidual, ContactJacobian

end # module FESolid2D
