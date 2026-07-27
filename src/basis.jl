"""
    vander_legendre_deriv(x, k=nothing)

Return the generalized Vandermonde matrix `Q` (Legendre polynomials evaluated
at `x`, up to degree `k-1`) and its derivative `dQ`.
"""
function vander_legendre_deriv(x, k=nothing)
    if isnothing(k)
        k = length(x) # Square matrix by default
    end
    m = length(x)
    Q = ones(m, k)
    dQ = zeros(m, k)
    Q[:, 2] = x
    dQ[:, 2] .= 1
    for n in 1:k-2
        Q[:, n+2] = ((2*n + 1) * x .* Q[:, n+1] - n * Q[:, n]) / (n + 1)
        dQ[:, n+2] = (2*n + 1) * Q[:, n+1] + dQ[:, n]
    end
    Q, dQ
end

"""
    febasis1D(P, Q, Qmode)

1D nodal (Lagrange) basis of `P` Gauss-Lobatto nodes, evaluated and
differentiated at `Q` quadrature points on the reference element `[-1, 1]`.
`Qmode` is either `"GAUSS"` (Gauss-Legendre) or `"GLL"` (Gauss-Lobatto)
quadrature.

Returns `(x, q, w, B, D)`: nodes, quadrature points, quadrature weights,
interpolation matrix (`Q x P`), and derivative matrix (`Q x P`).
"""
function febasis1D(P, Q, Qmode)
    x, _ = gausslobatto(P)
    if Qmode == "GAUSS"
        q, w = gausslegendre(Q)
    elseif Qmode == "GLL"
        q, w = gausslobatto(Q)
    else
        error("Qmode error! Choose GAUSS or GLL Quadrature points!")
    end
    V, _ = vander_legendre_deriv(x)
    Bp, Dp = vander_legendre_deriv(q, P)
    B = Bp / V
    D = Dp / V
    x, q, w, B, D
end

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

"""
    febasis2D(P, Q, num_comp, Qmode)

Tensor-product (2D) extension of [`febasis1D`](@ref) via Kronecker products,
for a `num_comp`-component field.

Returns `(x, q, w, B, D)`: reference nodes/quad points as `[x; y]`-stacked
vectors, quadrature weights, interpolation matrix `B`, and gradient matrix
`D` (stacked `[dudx; dudy]` blocks per component).
"""
function febasis2D(P, Q, num_comp, Qmode)
    # coordinate, quadrature, and basis on reference element [-1,1]
    x1_ref, q1_ref, w1_ref, B1, D1 = febasis1D(P, Q, Qmode)

    x, y = meshgrid(x1_ref, x1_ref)
    x2_ref = [x; y]
    qx, qy = meshgrid(q1_ref, q1_ref)
    q2_ref = [qx; qy]
    w2_ref = kron(w1_ref, w1_ref)

    B2 = kron(I(num_comp), kron(B1, B1))
    Dx = kron(B1, D1)
    Dy = kron(D1, B1)
    # Grad for num_comp = 1
    Dx = kron([1, 0], Dx)
    Dy = kron([0, 1], Dy)
    D2 = Dx + Dy
    D2 = kron(I(num_comp), D2)

    x2_ref, q2_ref, w2_ref, B2, D2
end

"""
    FEBasis(P, Q, num_comp, Qmode)

A `num_comp`-component 2D tensor-product finite element basis: `P` nodes per
direction, `Q` quadrature points per direction, on the reference element
`[-1, 1]^2`.
"""
struct FEBasis
    P::Int
    Q::Int
    num_comp::Int
    Qmode::String
    x_ref::Vector  # nodes at 2D ref element
    q_ref::Vector  # quad pts for 2D ref element
    w_ref::Vector  # weights for 2D ref element
    B::Matrix  # Interpolation
    D::Matrix  # Derivative
    function FEBasis(P, Q, num_comp, Qmode)
        x_ref, q_ref, w_ref, B, D = febasis2D(P, Q, num_comp, Qmode)
        new(P, Q, num_comp, Qmode, x_ref, q_ref, w_ref, B, D)
    end
end
