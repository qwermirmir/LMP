# Properties of the clover field strength and the clover Dirac operator.

using Test
using Random
using LinearAlgebra
using StaticArrays
using Gaugefields
using LatticeDiracOperators
using LMP

_dist(a, b) = norm(a.f .- b.f)

function clover_params()
    return Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => 0.16,
        "eps_CG" => 1.0e-8,
        "verbose_level" => 0,
    )
end

function apply(D, x)
    y = similar(x)
    mul!(y, D, x)
    return y
end

function random_fermion(U, rng)
    x = Initialize_pseudofermion_fields(U[1], "Wilson"; nowing=true)
    x.f .= randn(rng, ComplexF64, size(x.f))
    return x
end

@testset "clover" begin
    rng = Xoshiro(11)
    U_free = Initialize_Gaugefields(3, 0, 4, 4, 4, 4; condition="cold")
    U_hot = Initialize_Gaugefields(3, 0, 4, 4, 4, 4; condition="hot",
                                   randomnumber="Reproducible")

    @testset "free field has F_munu = 0" begin
        for mu in 1:4, nu in mu+1:4
            @test maximum(abs, compute_Fmunu(U_free, mu, nu)) < 1.0e-14
        end
    end

    @testset "F_munu is anti-Hermitian and traceless" begin
        worst_ah = 0.0
        worst_tr = 0.0
        for mu in 1:4, nu in mu+1:4
            F = compute_Fmunu(U_hot, mu, nu)
            for it in 1:4, iz in 1:4, iy in 1:4, ix in 1:4
                M = SMatrix{3,3,ComplexF64,9}(view(F, :, :, ix, iy, iz, it))
                worst_ah = max(worst_ah, maximum(abs, M + M'))
                worst_tr = max(worst_tr, abs(tr(M)))
            end
        end
        @test worst_ah < 1.0e-13
        # compute_Fmunu returns (Q - Q^dagger)/8 without removing the trace, so
        # F is not su(3)-valued. The standard Sheikholeslami-Wohlert definition
        # subtracts tr(F)/3. Pending a decision on whether to change the physics.
        @test_broken worst_tr < 1.0e-13
    end

    x = random_fermion(U_hot, rng)
    D_W = Dirac_operator(U_hot, x, clover_params())
    y_wilson = apply(D_W, x)

    @testset "c_SW = 0 reduces to Wilson" begin
        y = apply(CloverDiracOperator(D_W, U_hot, 0.0), x)
        @test _dist(y, y_wilson) < 1.0e-14 * norm(y_wilson.f)
    end

    y1 = apply(CloverDiracOperator(D_W, U_hot, 1.0), x)
    @testset "c_SW = 1 differs from Wilson on a hot field" begin
        @test _dist(y1, y_wilson) > 1.0e-6 * norm(y_wilson.f)
    end

    @testset "clover term is linear in c_SW" begin
        y2 = apply(CloverDiracOperator(D_W, U_hot, 2.0), x)
        part1 = y1.f .- y_wilson.f
        part2 = y2.f .- y_wilson.f
        @test norm(2 .* part1 .- part2) < 1.0e-13 * norm(part2)
    end

    @testset "free field clover equals Wilson" begin
        xf = random_fermion(U_free, rng)
        D_free = Dirac_operator(U_free, xf, clover_params())
        yw = apply(D_free, xf)
        ycl = apply(CloverDiracOperator(D_free, U_free, 1.0), xf)
        @test _dist(ycl, yw) < 1.0e-14 * norm(yw.f)
    end

    @testset "refresh_clover! recomputes the clover term" begin
        D_cl = CloverDiracOperator(D_W, U_hot, 1.0)
        D_cl.U = U_free
        refresh_clover!(D_cl)
        # The Wilson part still uses U_hot; the clover part is now zero.
        @test _dist(apply(D_cl, x), y_wilson) < 1.0e-14 * norm(y_wilson.f)
    end
end
