#!/usr/bin/env julia

using CUDA
using LinearAlgebra
using Random
using Gaugefields
using LatticeDiracOperators
using Test
using LMP

const DDAMG = LMP.DDalphaAMG

@testset "DDalphaAMG setup modes retain GPU storage" begin
    if !CUDA.functional()
        @info "CUDA unavailable; skipping setup-mode GPU smoke test"
        @test true
        return
    end

    CUDA.allowscalar(false)
    println("GPU setup-mode smoke settings:")
    println("  device=$(CUDA.device())")
    println("  lattice=4^4, kappa=0.12, block=2^4, Nv=1")
    println("  external vector, nsmooth=0, nvcycle=0, chirality=false")
    println("  precompute_coarse=true, precision=ComplexF64")

    U = Initialize_Gaugefields(3, 0, 4, 4, 4, 4;
                               condition="cold", cuda=true,
                               blocks=(1, 1, 1, 1))
    template = Initialize_pseudofermion_fields(U[1], "Wilson", nowing=true)
    D = Dirac_operator(U, template, Dict(
        "Dirac_operator" => "Wilson",
        "κ" => 0.12,
        "faster version" => true,
        "verbose_level" => 0,
    ))
    seed = similar(template)
    DDAMG.random_fermion!(seed)

    for mode in (:inverse_sap, :direct_near_null)
        setup = DDAMG.setup_DDalphaAMG(
            D, template, 2, 1;
            setup_mode=mode,
            nsmooth=0,
            nvcycle=0,
            coarse_vectors=[seed],
            use_chirality=false,
            precompute_coarse=true,
            verbose=false)

        @test setup.setup_mode == mode
        @test setup.test_vectors[1].f isa CuArray
        @test setup.coarse_stencil !== nothing
        @test setup.coarse_stencil.self isa CuArray
    end
end

# The odd-even steps apply only the color parts of the coarse stencil that can
# be nonzero. Their results must equal, bit for bit, the reference that applies
# the full stencil (the implementation before 2026-11-11).
@testset "coarse odd-even steps match full stencil applications on GPU" begin
    if !CUDA.functional()
        @info "CUDA unavailable; skipping coarse odd-even GPU test"
        @test true
        return
    end

    # NBX = NBY = NBZ = 2 makes the +mu and -mu neighbors the same block;
    # NBT = 4 gives distinct ones. Both cases occur in the stencil kernels.
    nv, nbx, nby, nbz, nbt = 6, 2, 2, 2, 4
    nblocks = nbx * nby * nbz * nbt
    rng = MersenneTwister(20261111)
    self = Array{ComplexF64}(undef, nv, nv, nblocks)
    for b in 1:nblocks
        self[:, :, b] .= 4 .* Matrix{ComplexF64}(I, nv, nv) .+ 0.05 .* randn(rng, ComplexF64, nv, nv)
    end
    plus = 0.1 .* randn(rng, ComplexF64, nv, nv, nblocks, 4)
    minus = 0.1 .* randn(rng, ComplexF64, nv, nv, nblocks, 4)
    nb_plus = [DDAMG._neighbor_block(b, mu, +1, nbx, nby, nbz, nbt) for b in 1:nblocks, mu in 1:4]
    nb_minus = [DDAMG._neighbor_block(b, mu, -1, nbx, nby, nbz, nbt) for b in 1:nblocks, mu in 1:4]
    op = DDAMG.CoarseStencilOperator(nv, nblocks, nbx, nby, nbz, nbt,
                                     CuArray(self), CuArray(plus), CuArray(minus),
                                     CuArray(Int32.(nb_plus)), CuArray(Int32.(nb_minus)))
    self_inv = DDAMG._amg_coarse_self_inverse(op)
    coarse(data) = DDAMG.CoarseVector(CuArray(data), nv, nblocks)
    rhs = coarse(randn(rng, ComplexF64, nv * nblocks))
    x_even = similar(rhs)
    DDAMG._amg_coarse_color_mask!(x_even, op, coarse(randn(rng, ComplexF64, nv * nblocks)), 0)
    t1, t2, t3 = similar(rhs), similar(rhs), similar(rhs)
    # Compare bits, with -0.0 counted as +0.0.
    same_bits(a, b) = Array(a.data) .+ 0.0 == Array(b.data) .+ 0.0

    ref = similar(rhs)
    DDAMG._amg_coarse_color_mask!(t1, op, rhs, 1)
    DDAMG._amg_coarse_blockdiag_solve!(t2, op, self_inv, t1, 1)
    DDAMG.apply_coarse_stencil!(t3, op, t2)
    DDAMG._amg_coarse_color_diff!(ref, op, rhs, t3, 0)
    out = DDAMG._amg_coarse_odd_even_rhs!(similar(rhs), op, self_inv, rhs, t1, t2, t3)
    @test same_bits(out, ref)

    DDAMG.apply_coarse_stencil!(t1, op, x_even)
    DDAMG._amg_coarse_blockdiag_solve!(t2, op, self_inv, t1, 1)
    DDAMG.apply_coarse_stencil!(t3, op, t2)
    DDAMG._amg_coarse_color_diff!(ref, op, t1, t3, 0)
    out = DDAMG._amg_apply_coarse_odd_even_schur!(similar(rhs), op, self_inv, x_even, t1, t2, t3)
    @test same_bits(out, ref)

    DDAMG.apply_coarse_stencil!(t1, op, x_even)
    DDAMG._amg_coarse_color_diff!(t2, op, rhs, t1, 1)
    DDAMG._amg_coarse_blockdiag_solve!(t3, op, self_inv, t2, 1)
    ref.data .= x_even.data
    DDAMG.coarse_axpby!(1.0, t3, 1.0, ref)
    out = DDAMG._amg_coarse_odd_even_reconstruct!(similar(rhs), op, self_inv, x_even, rhs, t1, t2, t3)
    @test same_bits(out, ref)
end
