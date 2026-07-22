using Test
using LinearAlgebra
using Random
using LMP.SubspaceSimilarity

@testset "subspace similarity" begin
    Q = Matrix{ComplexF64}(I, 8, 8)[:, 1:3]

    same = subspace_similarity(Q, Q)
    @test same.mean_capture ≈ 1.0 atol=1e-14
    @test same.worst_capture ≈ 1.0 atol=1e-14
    @test per_mode_capture(Q, Q) ≈ ones(3) atol=1e-14

    Qorth = Matrix{ComplexF64}(I, 8, 8)[:, 4:6]
    orthogonal = subspace_similarity(Q, Qorth)
    @test orthogonal.mean_capture ≈ 0.0 atol=1e-14
    @test orthogonal.worst_capture ≈ 0.0 atol=1e-14

    Random.seed!(42)
    U = Matrix(qr(randn(ComplexF64, 3, 3)).Q)
    rotated = subspace_similarity(Q, Q * U)
    @test rotated.singular_values ≈ ones(3) atol=1e-13
    @test rotated.principal_angles_deg ≈ zeros(3) atol=1e-5

    @test_throws ArgumentError orthonormal_basis(hcat(Q[:, 1], Q[:, 1]))
end
