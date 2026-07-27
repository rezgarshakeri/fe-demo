module FESolid2D

using LinearAlgebra
using SparseArrays
using FastGaussQuadrature

include("basis.jl")

export vander_legendre_deriv, febasis1D, febasis2D, FEBasis

end # module FESolid2D
