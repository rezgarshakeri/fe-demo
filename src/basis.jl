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

"""
    feval1D_at(P, xi0)

1D nodal (Lagrange) basis of `P` Gauss-Lobatto nodes (same nodes as
[`febasis1D`](@ref)), interpolated and differentiated at a single point `xi0`
(rather than a quadrature rule). Used to evaluate a field's value/derivative
exactly at a reference-element boundary (`xi0 = -1` or `xi0 = 1`).

Returns `(B, D)` as `1 x P` row vectors.
"""
function feval1D_at(P, xi0)
    x, _ = gausslobatto(P)
    V, _ = vander_legendre_deriv(x)
    Bp, Dp = vander_legendre_deriv([xi0], P)
    return Bp / V, Dp / V
end

"""
    FEFaceBasis

Face-restricted counterpart of [`FEBasis`](@ref) (see [`febasis2D_face`](@ref)).
`B1`/`D1` are single-component operators (`Q x P^2` / `2Q x P^2`, stacked
`[d/dxi; d/deta]`) -- convenient for per-component gap/gradient evaluation in
`contact.jl`. `B`/`D` are the `num_comp`-block-diagonal versions, matching
`FEBasis`'s layout, for use as a face-restricted geometry basis (mirrors
`Bx::FEBasis` in [`GetQdataFace`](@ref)).
"""
struct FEFaceBasis
    P::Int
    Q::Int
    num_comp::Int
    Qmode::String
    side::String
    w_ref::Vector  # tangential (1D) quadrature weights on the reference face
    B1::Matrix     # Q x P^2, 1-component interpolation
    D1::Matrix     # 2Q x P^2, 1-component [d/dxi; d/deta] derivative
    B::Matrix      # num_comp*Q x num_comp*P^2
    D::Matrix      # num_comp*2Q x num_comp*P^2
end

"""
    febasis2D_face(P, Q, num_comp, Qmode, side)

Face-restricted counterpart of [`febasis2D`](@ref): the same `P`-node
tensor-product basis, but its interpolation/derivative operators are
evaluated only at the `Q` quadrature points running along one edge (`side`)
of the reference element, with the perpendicular reference coordinate fixed
at its boundary value (`feval1D_at`). Needed to evaluate a volume field's
*gradient* (e.g. stress) at points on a boundary face -- unlike
[`GetFaceQdata`](@ref), which only interpolates physical position/arc-length
from the (always-linear) geometry corners.

Returns an [`FEFaceBasis`](@ref).
"""
function febasis2D_face(P, Q, num_comp, Qmode, side)
    _, _, w1_ref, B1, D1 = febasis1D(P, Q, Qmode) # tangential (along-face) direction
    xi0 = side in ("bottom", "left") ? -1.0 : 1.0
    Bpt, Dpt = feval1D_at(P, xi0) # fixed (boundary) direction

    if side == "bottom" || side == "top"
        # tangential = reference direction 1 (xi), fixed = reference direction 2 (eta)
        Bface1 = kron(Bpt, B1)
        Dxi1 = kron(Bpt, D1)
        Deta1 = kron(Dpt, B1)
    elseif side == "left" || side == "right"
        # tangential = reference direction 2 (eta), fixed = reference direction 1 (xi)
        Bface1 = kron(B1, Bpt)
        Dxi1 = kron(B1, Dpt)
        Deta1 = kron(D1, Bpt)
    else
        error("side must be one of: bottom, right, top, left")
    end
    Dref1 = kron([1, 0], Dxi1) + kron([0, 1], Deta1) # stacked [d/dxi; d/deta], 1 component

    B = kron(I(num_comp), Bface1)
    D = kron(I(num_comp), Dref1)

    return FEFaceBasis(P, Q, num_comp, Qmode, side, w1_ref, Bface1, Dref1, B, D)
end
