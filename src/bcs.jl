"""
    GetDirichletBCsIndex(num_comp, Ind, sides=(:bottom, :right, :top, :left))

Global dof indices with a Dirichlet BC on the given `sides` of a structured
mesh (any subset of `:bottom`, `:right`, `:top`, `:left`) — the remaining
sides are left free for a natural (e.g. traction) BC.

Each side contributes its own full node range, including its two corners, so
any single side (e.g. just `:left`) is fully and correctly clamped on its
own. A corner shared by two requested sides is harmlessly listed twice (both
the x- and y-component blocks get the same duplicate, so the pairing used to
build `x_bc = reshape(x[bc_idx], :, 2)` elsewhere stays consistent).
"""
function GetDirichletBCsIndex(num_comp, Ind::FEIndices, sides=(:bottom, :right, :top, :left))
    num_nodes = Ind.nodes_u[1] * Ind.nodes_u[2]
    bc_idx = Int64[]
    for c = 0:num_comp-1
        if :bottom in sides
            for i = 1:Ind.nodes_u[1]
                append!(bc_idx, Ind.idx_b[i] .+ c * num_nodes)
            end
        end
        if :top in sides
            for i = 1:Ind.nodes_u[1]
                append!(bc_idx, Ind.idx_t[i] .+ c * num_nodes)
            end
        end
        if :right in sides
            for j = 1:Ind.nodes_u[2]
                append!(bc_idx, Ind.idx_r[j] .+ c * num_nodes)
            end
        end
        if :left in sides
            for j = 1:Ind.nodes_u[2]
                append!(bc_idx, Ind.idx_l[j] .+ c * num_nodes)
            end
        end
    end
    return bc_idx
end
