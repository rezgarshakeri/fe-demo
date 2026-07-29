using Random

@testset verbose = true "1D basis" begin

    @testset "interpolation exact for P=5" begin
        P = Q = 5
        x, q, w, B, _ = febasis1D(P, Q, "GAUSS")
        c = randn(P) # coefficients of a degree P-1 polynomial
        f(t) = sum(c[k] * t^(k-1) for k in 1:P)
        u = f.(x)
        @test B * u ≈ f.(q)
        @test sum(w) ≈ 2.0 # quadrature integrates constants exactly over [-1,1]
    end

    @testset "gradient exact for P=4" begin
        P = Q = 4
        x, q, _, _, D = febasis1D(P, Q, "GAUSS")
        c = randn(P)
        f(t) = sum(c[k] * t^(k-1) for k in 1:P)
        df(t) = sum((k-1) * c[k] * t^(k-2) for k in 2:P)
        u = f.(x)
        @test D * u ≈ df.(q)
    end

end

@testset verbose = true "2D basis" begin

    @testset "interpolation exact for P=5 (num_comp=$num_comp)" for num_comp in (1, 2)
        P = Q = 5
        x_ref, q_ref, w_ref, B, _ = febasis2D(P, Q, num_comp, "GAUSS")
        xn = reshape(x_ref, P^2, :)
        xq = reshape(q_ref, Q^2, :)

        c = [randn(P, P) for _ in 1:num_comp] # coeffs c[comp][i,j] * xi^(i-1) * eta^(j-1)
        f(xi, eta, cc) = sum(cc[i, j] * xi^(i-1) * eta^(j-1) for i in 1:P, j in 1:P)

        U = vcat([f.(xn[:, 1], xn[:, 2], Ref(cc)) for cc in c]...)
        Bex = vcat([f.(xq[:, 1], xq[:, 2], Ref(cc)) for cc in c]...)

        @test B * U ≈ Bex
        @test sum(w_ref) ≈ 4.0 # area of the reference element [-1,1]^2

        # partition of unity: shape functions for one component sum to 1 everywhere
        B1comp = B[1:Q^2, 1:P^2]
        @test B1comp * ones(P^2) ≈ ones(Q^2)
    end

    @testset "gradient exact for P=4 (num_comp=$num_comp)" for num_comp in (1, 2)
        P = Q = 4
        x_ref, q_ref, _, _, D = febasis2D(P, Q, num_comp, "GAUSS")
        xn = reshape(x_ref, P^2, :)
        xq = reshape(q_ref, Q^2, :)

        c = [randn(P, P) for _ in 1:num_comp]
        f(xi, eta, cc) = sum(cc[i, j] * xi^(i-1) * eta^(j-1) for i in 1:P, j in 1:P)
        dfdxi(xi, eta, cc) = sum((i-1) * cc[i, j] * xi^(i-2) * eta^(j-1) for i in 2:P, j in 1:P)
        dfdeta(xi, eta, cc) = sum((j-1) * cc[i, j] * xi^(i-1) * eta^(j-2) for i in 1:P, j in 2:P)

        U = vcat([f.(xn[:, 1], xn[:, 2], Ref(cc)) for cc in c]...)
        Dex = vcat([vcat(dfdxi.(xq[:, 1], xq[:, 2], Ref(cc)), dfdeta.(xq[:, 1], xq[:, 2], Ref(cc))) for cc in c]...)

        @test D * U ≈ Dex
    end

end
