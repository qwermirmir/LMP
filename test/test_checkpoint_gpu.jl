# Resuming from a checkpoint must continue exactly like an uninterrupted run:
# weights, Adam moments and the Adam timestep all have to be restored.

using Test
using CUDA

include(joinpath(@__DIR__, "golden", "model.jl"))

const GPU_BLOCKS = (2, 2, 2, 2)

function gpu_model(build)
    layers = build().layers
    substitute_Layer!(layers, GPU_BLOCKS)
    return layers
end

@testset "GPU checkpoint resume, $name" for (name, build) in
        (("4^4 model", golden_model), ("asymmetric model", asymmetric_model))
    path = tempname() * ".jld2"

    layers = gpu_model(build)
    opt = Adam(1.0e-3)
    golden_train_step!(layers, opt)
    save_checkpoint(path, layers, opt)
    continued = [golden_train_step!(layers, opt) for _ in 1:2]

    # Loaded before any GPU pass, as the training scripts do.
    layers = gpu_model(build)
    opt = Adam(1.0e-3)
    load_checkpoint!(path, layers, opt)
    rm(path)
    @test opt.t == 1
    resumed = [golden_train_step!(layers, opt) for _ in 1:2]

    # The first step checks the weights; the second also checks the Adam state.
    for (r, c) in zip(resumed, continued)
        @test isapprox(r.loss, c.loss; rtol=1.0e-12)
        @test isapprox(r.grad_norm2, c.grad_norm2; rtol=1.0e-12)
    end
end
