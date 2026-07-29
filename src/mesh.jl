"""
    GetConnectivity(P, nx, ny)

Connectivity for a structured `nx x ny` mesh of tensor-product elements with
`P` solution nodes per direction (bilinear, 2-node, geometry).

    4-------5--------6
    |       |        |
    1-------2--------3
    local numbering of one element is
    3-------4
    |       |
    1-------2

Returns `(idx_x, idx_u, idx_b, idx_r, idx_t, idx_l, nodes_u)`: coordinate and
solution connectivity (size `elem_dof x num_elem`), global node indices along
each of the 4 boundary faces (`[bottom, right, top, left]`), and the number
of solution nodes per direction.
"""
function GetConnectivity(P, nx, ny)
    n_elems = [nx, ny]
    nodes_u = zeros(Int64, 2)
    nodes_x = zeros(Int64, 2)
    for d = 1:2
        nodes_u[d] = n_elems[d] * (P - 1) + 1
        nodes_x[d] = n_elems[d] * (2 - 1) + 1
    end

    num_elem = n_elems[1] * n_elems[2]
    idx_x = zeros(Int64, 4, num_elem) # Coordinate connectivity
    idx_u = zeros(Int64, P * P, num_elem) # Solution connectivity
    for i = 1:n_elems[1]
        for j = 1:n_elems[2]
            ele = (j - 1) * n_elems[1] + i
            for ii = 1:P
                for jj = 1:P
                    if (ii < 3) && (jj < 3)
                        idx_x[(ii-1)*2+jj, ele] = ((j - 1) * (2 - 1) + ii - 1) * nodes_x[1] + (i - 1) * (2 - 1) + jj
                    end
                    idx_u[(ii-1)*P+jj, ele] = ((j - 1) * (P - 1) + ii - 1) * nodes_u[1] + (i - 1) * (P - 1) + jj
                end
            end
        end
    end

    idx_b = Int64[] # bottom face
    idx_t = Int64[] # top face
    for i = 1:n_elems[1]
        if i == 1
            append!(idx_b, idx_u[1:P, i])
            append!(idx_t, idx_u[P*(P-1)+1:P*P, i+n_elems[1]*(n_elems[2]-1)])
        else
            append!(idx_b, idx_u[2:P, i])
            append!(idx_t, idx_u[P*(P-1)+2:P*P, i+n_elems[1]*(n_elems[2]-1)])
        end
    end

    idx_l = Int64[] # left face
    idx_r = Int64[] # right face
    for i = 1:n_elems[1]:num_elem
        if i == 1
            append!(idx_l, idx_u[1:P:P*(P-1)+1, i])
            append!(idx_r, idx_u[P:P:P*P, i+n_elems[1]-1])
        else
            append!(idx_l, idx_u[P+1:P:P*(P-1)+1, i])
            append!(idx_r, idx_u[2*P:P:P*P, i+n_elems[1]-1])
        end
    end

    return idx_x, idx_u, idx_b, idx_r, idx_t, idx_l, nodes_u
end

"""
    FEIndices(P, num_comp, nx, ny)

Mesh connectivity plus element restriction operators for a structured
`nx x ny` mesh: `Er_x[e]`/`Er_u[e]` map the global coordinate/solution dof
vectors down to element `e`'s local dofs.
"""
struct FEIndices
    P::Int
    nx::Int
    ny::Int
    idx_x::Matrix
    idx_u::Matrix
    idx_b::Vector
    idx_r::Vector
    idx_t::Vector
    idx_l::Vector
    nodes_u::Vector
    Er_x::Vector
    Er_u::Vector
    function FEIndices(P, num_comp, nx, ny)
        idx_x, idx_u, idx_b, idx_r, idx_t, idx_l, nodes_u = GetConnectivity(P, nx, ny)

        dof_u = maximum(idx_u)
        dof_x = maximum(idx_x)
        elem_dof_u = size(idx_u, 1)
        elem_dof_x = size(idx_x, 1)
        num_elem = size(idx_x, 2)

        Er_x = []
        Er_u = []
        for e = 1:num_elem
            Lx = spzeros(Int64, elem_dof_x, dof_x)
            Lu = spzeros(Int64, elem_dof_u, dof_u)
            for i = 1:elem_dof_u
                if i < elem_dof_x + 1
                    Lx[i, idx_x[i, e]] = 1
                end
                Lu[i, idx_u[i, e]] = 1
            end
            Lx = kron(I(2), Lx)
            Lu = kron(I(num_comp), Lu)
            push!(Er_x, Lx)
            push!(Er_u, Lu)
        end

        new(P, nx, ny, idx_x, idx_u, idx_b, idx_r, idx_t, idx_l, nodes_u, Er_x, Er_u)
    end
end

"""
    GetCoordMesh(mesh, nelx, nely, aspect_ratio)

Physical nodal coordinates for an `nelx x nely` structured mesh on
`[0,1] x [0,1/aspect_ratio]`. `mesh` is `"uniform"`, `"random"` (perturbed
interior nodes), or `"trapezoid"` (skewed interior nodes); the boundary is
always axis-aligned.
"""
function GetCoordMesh(mesh, nelx, nely, aspect_ratio)
    nodex = nelx + 1
    nodey = nely + 1
    numnodes = nodex * nodey
    interiornodex = nelx - 1
    interiornodey = nely - 1
    interiornodes = interiornodex * interiornodey

    hx = 1 / nelx
    hy = 1 / (nely * aspect_ratio)
    h = minimum([hx hy])

    if aspect_ratio < 1
        error("aspect_ratio must be bigger than 1")
    end

    x0 = LinRange(0, 1, nodex)
    if mesh == "uniform"
        y0 = 0.0 * x0
        y = zeros(numnodes, 1)
        for i = 1:nodex
            y1 = LinRange(y0[i], 1 / aspect_ratio, nodey)
            for j = 1:nodey
                y[i+(j-1)*nodex] = y1[j]
            end
        end
        x = zeros(numnodes, 1)
        for i = 1:nodey
            for j = 1:nodex
                x[j+(i-1)*nodex] = x0[j]
            end
        end

    elseif mesh == "random"
        y0 = 0.0 * x0
        y = zeros(numnodes, 1)
        for i = 1:nodex
            y1 = LinRange(y0[i], 1 / aspect_ratio, nodey)
            for j = 1:nodey
                y[i+(j-1)*nodex] = y1[j]
            end
        end
        x = zeros(numnodes, 1)
        for i = 1:nodey
            for j = 1:nodex
                x[j+(i-1)*nodex] = x0[j]
            end
        end
        Random.seed!(1234)
        randnodes_x = rand(interiornodes, 1) * hx / 2 .- hx / 4
        randnodes_y = rand(interiornodes, 1) * hy / 2 .- hy / 4
        for i = 1:interiornodey
            for j = 1:interiornodex
                x[i*(nodex)+j+1] = x[i*(nodex)+j+1] - randnodes_x[j+(i-1)*interiornodex]
                y[i*(nodex)+j+1] = y[i*(nodex)+j+1] - randnodes_y[j+(i-1)*interiornodex]
            end
        end

    elseif mesh == "trapezoid"
        y0 = 0.0 * x0
        y = zeros(numnodes, 1)
        for i = 1:nodex
            y1 = LinRange(y0[i], 1, nodey)
            for j = 1:nodey
                y[i+(j-1)*nodex] = y1[j]
            end
        end
        x = zeros(numnodes, 1)
        for i = 1:nodey
            for j = 1:nodex
                x[j+(i-1)*nodex] = x0[j]
            end
        end
        for i = 1:interiornodey
            for j = 1:interiornodex
                y[i*(nodex)+j+1] = y[i*(nodex)+j+1] - (-1)^j * (h / 4)
            end
        end

    else
        error("Enter one of the mesh option: 'uniform', 'trapezoid', 'random' ")
    end

    return x, y
end

"""
    GetCoordElem(xc, yc, Er_x, e)

Physical coordinates of element `e`'s nodes, as a `(8,1)` array `[x; y]`.
"""
function GetCoordElem(xc, yc, Er_x, e)
    return Er_x[e] * [xc; yc]
end

"""
    GetNodalCoordinate(xc, yc, Q, Ind)

Physical `(x, y)` coordinates of every solution dof (for plotting/BC
evaluation), assembled from a `Q`-point GLL interpolation of the geometry.
"""
function GetNodalCoordinate(xc, yc, Q, Ind::FEIndices)
    num_elem = size(Ind.idx_u, 2)
    global_dof = 2 * maximum(Ind.idx_u)

    _, _, _, BL, _ = febasis2D(2, Q, 2, "GLL")
    x = spzeros(global_dof, 1)
    mult = spzeros(global_dof, 1)
    for e = 1:num_elem
        Coord_E = GetCoordElem(xc, yc, Ind.Er_x, e)
        xe = BL * Coord_E
        x = x + Ind.Er_u[e]' * xe
        mult += Ind.Er_u[e]' * Ind.Er_u[e] * ones(global_dof)
    end
    return x ./ mult
end

"""
    GetQdata(Coord_E, Bx::FEBasis)

Quadrature-point coordinates `q`, `wdetJ = w .* det(J)`, and `dXdx_T = J^{-T}`
for element `e` with physical corner coordinates `Coord_E`.
"""
function GetQdata(Coord_E, Bx::FEBasis)
    J1 = Bx.D * Coord_E
    J2 = reshape(J1, Bx.Q^2, :)
    J = zeros(Bx.Q^2, 2, 2)
    detJ = zeros(Bx.Q^2)
    dXdx_T = zeros(Bx.Q^2, 2, 2)
    for i = 1:Bx.Q^2
        J[i, :, :] = reshape(J2[i, :], 2, 2)
        detJ[i] = det(J[i, :, :])
        dXdx_T[i, :, :] = I / J[i, :, :]
    end

    q = Bx.B * Coord_E
    wdetJ = Bx.w_ref .* detJ

    return q, wdetJ, dXdx_T
end

"""
    GetFaceLocalIndex(P, side)

Local (element-level) node indices lying on a given face of a structured
`Q_{P-1}` element, consistent with the local numbering used in
[`GetConnectivity`](@ref) (solution nodes: `(ii-1)*P+jj`; geometry corners:
1=bottom-left, 2=bottom-right, 3=top-left, 4=top-right).
"""
function GetFaceLocalIndex(P, side)
    if side == "bottom"
        loc_u, loc_x = 1:P, [1, 2]
    elseif side == "top"
        loc_u, loc_x = (P-1)*P+1:P*P, [3, 4]
    elseif side == "left"
        loc_u, loc_x = 1:P:(P-1)*P+1, [1, 3]
    elseif side == "right"
        loc_u, loc_x = P:P:P*P, [2, 4]
    else
        error("side must be one of: bottom, right, top, left")
    end
    return collect(loc_u), loc_x
end

"""Element numbers (in the `nx x ny` structured mesh) whose face lies on `side`."""
function GetBoundaryElements(nx, ny, side)
    if side == "bottom"
        return collect(1:nx)
    elseif side == "top"
        return collect((ny-1)*nx+1:ny*nx)
    elseif side == "left"
        return collect(1:nx:(ny-1)*nx+1)
    elseif side == "right"
        return collect(nx:nx:ny*nx)
    else
        error("side must be one of: bottom, right, top, left")
    end
end

"""
    FEFaceIndices(P, num_comp, nx, ny, Ind, side)

Face element restriction for the boundary elements on `side` of a structured
mesh: `Er_face_x[k]`/`Er_face_u[k]` map global coordinate/solution dofs down
to the `k`-th boundary element's face-local dofs. Used to apply a traction
(Neumann) BC via [`GetTractionGlobal`](@ref).
"""
struct FEFaceIndices
    side::String
    elems::Vector{Int}        # boundary element numbers touching this side
    Er_face_x::Vector         # element restriction: [xc;yc] -> face corner coords [x1;x2;y1;y2]
    Er_face_u::Vector         # element restriction: global u dof -> face-local u dof
    normal::Vector{Float64}   # outward unit normal (mesh boundary is axis-aligned)
    function FEFaceIndices(P, num_comp, nx, ny, Ind::FEIndices, side)
        elems = GetBoundaryElements(nx, ny, side)
        loc_u, loc_x = GetFaceLocalIndex(P, side)

        dof_u = maximum(Ind.idx_u)
        dof_x = maximum(Ind.idx_x)

        Er_face_x = []
        Er_face_u = []
        for e in elems
            Lx = spzeros(Int64, 2, dof_x)
            Lu = spzeros(Int64, P, dof_u)
            for (k, i) in enumerate(loc_x)
                Lx[k, Ind.idx_x[i, e]] = 1
            end
            for (k, i) in enumerate(loc_u)
                Lu[k, Ind.idx_u[i, e]] = 1
            end
            push!(Er_face_x, kron(I(2), Lx))
            push!(Er_face_u, kron(I(num_comp), Lu))
        end

        normal = Dict("bottom" => [0.0, -1.0], "top" => [0.0, 1.0],
            "left" => [-1.0, 0.0], "right" => [1.0, 0.0])[side]

        new(side, elems, Er_face_x, Er_face_u, normal)
    end
end

"""
    GetFaceQdata(Coord_face, Bu::FEBasis)

Physical quadrature points and length element (`ds`) along a straight
element face, given its two corner coordinates `Coord_face = [x1;x2;y1;y2]`
(mirrors [`GetQdata`](@ref) but for the 1D boundary of a 2D element).
"""
function GetFaceQdata(Coord_face, Bu::FEBasis)
    _, _, w1, B1g, D1g = febasis1D(2, Bu.Q, Bu.Qmode)  # linear (corner) geometry basis
    xy = reshape(Coord_face, 2, 2)
    dxds = D1g * xy[:, 1]
    dyds = D1g * xy[:, 2]
    wds = w1 .* sqrt.(dxds .^ 2 .+ dyds .^ 2)
    xq = B1g * xy[:, 1]
    yq = B1g * xy[:, 2]
    return xq, yq, wds
end

# This is like a libCEED/Ratel Neumann (traction) boundary QFunction: f0_face = v * t
function GetTractionLocal(xq, yq, wds, t, Bu::FEBasis)
    _, _, _, B1u, _ = febasis1D(Bu.P, Bu.Q, Bu.Qmode)
    Bface = kron(I(Bu.num_comp), B1u)
    W = diagm(repeat(wds, Bu.num_comp))
    return Bface' * W * t(xq, yq)  # v^T * t
end

"""
    GetTractionGlobal(xc, yc, Fx, Ind, Bu, t)

Assemble the global traction load vector `Ft = ∫_Γ v · t ds` over `Fx.elems`,
where `t(x, y)` returns the traction stacked as `[tx; ty]`.
"""
function GetTractionGlobal(xc, yc, Fx::FEFaceIndices, Ind::FEIndices, Bu::FEBasis, t)
    global_dof = Bu.num_comp * maximum(Ind.idx_u)
    Ft = spzeros(global_dof, 1)
    for k in eachindex(Fx.elems)
        Coord_face = Fx.Er_face_x[k] * [xc; yc]
        xq, yq, wds = GetFaceQdata(Coord_face, Bu)
        Fe = GetTractionLocal(xq, yq, wds, t, Bu)
        Ft += Fx.Er_face_u[k]' * Fe
    end
    return Ft
end
