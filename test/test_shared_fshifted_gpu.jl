# deepcopy of a GPU field shares one fshifted scratch buffer. The copies, and
# the AMG work fields made from them, must stay correct under
# LatticeDiracOperators' mul!, which shifts its input through fshifted. Before
# the buffer was shared, copies got a 1-element placeholder and mul! returned
# wrong results on them, while the fused Wilson kernel was unaffected.

using Test
using CUDA
using LinearAlgebra

include(joinpath(@__DIR__, "golden", "model.jl"))

const GPU_BLOCKS = (2, 2, 2, 2)

function fused(D, v; dagger=false)
    y = deepcopy(v)
    LMP.fused_wilson_mul_gpu!(y, v, D.U, LMP._dirac_kappa(D), dagger)
    return Array(y.f)
end

function via_mul(D, v)
    y = deepcopy(v)
    mul!(y, D, v)
    return Array(y.f)
end

@testset "deepcopy shares a full-size fshifted buffer" begin
    layers = golden_model().layers
    substitute_Layer!(layers, GPU_BLOCKS)
    D = layers[end].D
    u, _ = forward_propagation(layers, D, D)
    # Network output vectors, the input of the Ritz layer.
    v = u[end][1]
    @test v.f isa CUDA.CuArray

    @test size(v.fshifted) == size(v.f)
    @test LMP._is_shared_fshifted(v.fshifted)
    @test deepcopy(v).fshifted === v.fshifted

    @test isapprox(via_mul(D, v), fused(D, v); rtol=1.0e-12)
    @test isapprox(via_mul(D', v), fused(D, v; dagger=true); rtol=1.0e-12)

    # AMG work fields copied from a network vector keep the shared buffer.
    w = LMP.DDalphaAMG._copy_wf(v)
    @test w.fshifted === v.fshifted
    @test isapprox(via_mul(D, w), fused(D, w); rtol=1.0e-12)

    # A field with its own buffer still gets its own one in AMG work fields.
    own = LMP.DDalphaAMG._amg_similar(similar(v))
    @test !LMP._is_shared_fshifted(own.fshifted)
    @test size(own.fshifted) == size(v.f)
end
