# Golden regression test: a small fixed model must reproduce the loss,
# activations, gradients, and one Adam step recorded in test/golden/reference.jl.

using Test

include(joinpath(@__DIR__, "golden", "model.jl"))
include(joinpath(@__DIR__, "golden", "reference.jl"))

@testset "golden regression" begin
    result = golden_measure()
    rtol = 1.0e-10
    @test keys(result) == keys(GOLDEN_REFERENCE)
    for k in keys(GOLDEN_REFERENCE)
        @testset "$k" begin
            @test isapprox(getfield(result, k), getfield(GOLDEN_REFERENCE, k); rtol)
        end
    end
end
