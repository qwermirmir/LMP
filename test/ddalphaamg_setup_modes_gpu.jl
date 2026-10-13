#!/usr/bin/env julia

using CUDA
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
