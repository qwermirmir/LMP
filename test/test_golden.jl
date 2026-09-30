# Golden regression test: small fixed models must reproduce the loss,
# activations, gradients, and one Adam step recorded in test/golden/reference.jl.

using Test

include(joinpath(@__DIR__, "golden", "model.jl"))
include(joinpath(@__DIR__, "golden", "reference.jl"))

const CASES = (("4^4 lattice", golden_model, GOLDEN_REFERENCE),
               ("asymmetric lattice", asymmetric_model, ASYM_REFERENCE))

@testset "golden regression, $name" for (name, build, reference) in CASES
    result = golden_measure(build().layers)
    rtol = 1.0e-10
    @test keys(result) == keys(reference)
    for k in keys(reference)
        @testset "$k" begin
            @test isapprox(getfield(result, k), getfield(reference, k); rtol)
        end
    end
end
