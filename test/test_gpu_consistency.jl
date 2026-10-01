# The GPU forward pass, backward pass, gradients and Adam step must reproduce
# the CPU results on the golden models.
#
# Norms do not see a conjugated gradient, but the loss after one Adam step does:
# with the wrong convention the imaginary parts move uphill.

using Test
using CUDA

include(joinpath(@__DIR__, "golden", "model.jl"))

# Blocks of 2^4 sites, so every lattice is split into several GPU blocks.
const GPU_BLOCKS = (2, 2, 2, 2)

# grad_proj is left out: it projects the flattened gradient, and the GPU stores
# fermion fields in a blocked layout.
const COMPARED_KEYS = (:loss, :activations, :grad_norm2, :loss_after_step)

@testset "GPU matches CPU" begin
    @testset "$name" for (name, build) in (
            ("4^4 model", golden_model),
            ("asymmetric model", asymmetric_model))
        cpu = golden_measure(build().layers)
        layers = build().layers
        substitute_Layer!(layers, GPU_BLOCKS)
        @test layers[1].x[1].f isa CUDA.CuArray
        gpu = golden_measure(layers)
        for k in COMPARED_KEYS
            @test isapprox(getfield(gpu, k), getfield(cpu, k); rtol=1.0e-10)
        end
    end
end
