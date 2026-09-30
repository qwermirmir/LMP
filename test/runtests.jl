# Runs the test suite. Each file is evaluated in its own module so that
# top-level definitions do not collide.
#
#     julia --project=. test/runtests.jl
#
# GPU tests run only when CUDA is functional. Set CUDA_VISIBLE_DEVICES="" to
# force a CPU-only run on a machine with GPUs.

using Test
using CUDA

const CPU_TESTS = [
    "test_golden.jl",
    "test_gradients.jl",
    "adam_global_step.jl",
    "test_gauge_covariance.jl",
    "test_clover.jl",
    "subspace_similarity.jl",
    "ddalphaamg_setup_modes.jl",
    "gdlambda_amg.jl",
]

const GPU_TESTS = [
    "test_gpu_consistency.jl",
    "ddalphaamg_setup_modes_gpu.jl",
]

function run_test_file(file)
    @testset "$file" begin
        # A `module` expression, unlike `Module()`, defines `include`. The prefix
        # keeps the module name from shadowing a function named after the file.
        name = Symbol("TestFile_", splitext(file)[1])
        mod = Core.eval(Main, :(module $name end))
        Base.include(mod, joinpath(@__DIR__, file))
    end
end

@testset "LMP" begin
    foreach(run_test_file, CPU_TESTS)
    if CUDA.functional()
        foreach(run_test_file, GPU_TESTS)
    else
        @info "CUDA is not functional; skipping GPU tests" GPU_TESTS
    end
end
