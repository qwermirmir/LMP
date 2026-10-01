# The production training loop, `train(opt, layers, gauge_paths, U_template,
# params; ...)`, run for a few steps on the golden model.
#
# With batch = 1 its first two losses must equal the golden loss before and
# after one Adam step, whatever the gauge source. This checks the loop's own
# backward pass (backward_and_gradient! with ping-pong buffers) against the
# plain one used by golden_measure.

using Test
using CUDA
using JLD2

include(joinpath(@__DIR__, "golden", "model.jl"))
include(joinpath(@__DIR__, "golden", "reference.jl"))

const GPU_BLOCKS = (2, 2, 2, 2)
const TRAIN_PARAMS = Dict{String,Any}(
    "Dirac_operator" => "Wilson",
    "κ" => GOLDEN_KAPPA,
    "faster version" => true,
    "verbose_level" => 0,
)

gpu_gauge_template() =
    Initialize_Gaugefields(3, 0, GOLDEN_DIMS...; condition="cold", cuda=true, blocks=GPU_BLOCKS)

function gpu_golden_layers()
    layers = golden_model().layers
    substitute_Layer!(layers, GPU_BLOCKS)
    return layers
end

# Runs `train` and returns the per-step losses it logs.
function train_losses!(layers, opt; epochs, gauge_paths=String[], kwargs...)
    log = tempname() * ".log"
    train(opt, layers, gauge_paths, gpu_gauge_template(), TRAIN_PARAMS;
          epochs, logfile=log, kwargs...)
    losses = [parse(Float64, match(r"Loss: ([-+0-9.eE]+)", line).captures[1])
              for line in eachline(log) if startswith(line, "Step:")]
    rm(log)
    return losses
end

@testset "train on GPU" begin
    golden = real.([GOLDEN_REFERENCE.loss, GOLDEN_REFERENCE.loss_after_step])
    U = golden_model().U
    dir = mktempdir()
    jld2_path = joinpath(dir, "config.jld2")
    JLD2.save(jld2_path, "U", U)
    raw_path = joinpath(dir, "config.gbin")
    LMP.save_raw_gauge(jld2_path, raw_path, gpu_gauge_template())

    @testset "batch 1, gauge from $source" for (source, kwargs) in (
            ("memory", (; gauge_cache=[U])),
            ("JLD2 file", (; gauge_paths=[jld2_path], gauge_format=:jld2)),
            ("raw file", (; gauge_paths=[raw_path], gauge_format=:raw)))
        losses = train_losses!(gpu_golden_layers(), Adam(1.0e-3); epochs=2, kwargs...)
        @test isapprox(losses, golden; rtol=1.0e-12)
    end

    # Keeping only every second activation and recomputing the rest in the
    # backward pass must not change the result.
    @testset "activation checkpointing" begin
        losses = train_losses!(gpu_golden_layers(), Adam(1.0e-3); epochs=2,
                               gauge_paths=[raw_path], gauge_format=:raw,
                               activation_checkpoint_interval=2)
        @test losses[1] ≈ golden[1] rtol=1.0e-12
        # Freeing the forward buffers makes BlockScalar_Layer rebuild its GPU
        # cache from the stale CPU weights, so its update is lost every step.
        # The second loss equals that of a step with BlockScalar frozen.
        @test_broken isapprox(losses[2], golden[2]; rtol=1.0e-12)
    end

    # The test loss is evaluated after the update, so at step 1 on the
    # training configuration it equals the golden loss after one step.
    @testset "test loss, $(fmt)" for (fmt, source) in (
            (:raw, (; gauge_paths=[raw_path], test_paths=[raw_path], gauge_format=:raw)),
            (:jld2, (; gauge_paths=[jld2_path], test_paths=[jld2_path], gauge_format=:jld2)))
        log = tempname() * ".log"
        train(Adam(1.0e-3), gpu_golden_layers(), source.gauge_paths, gpu_gauge_template(),
              TRAIN_PARAMS; epochs=1, logfile=log, test_paths=source.test_paths,
              test_interval=1, gauge_format=source.gauge_format)
        test_losses = [parse(Float64, match(r"Test Loss: ([-+0-9.eE]+)", line).captures[1])
                       for line in eachline(log) if occursin("[Test]", line)]
        rm(log)
        @test length(test_losses) == 1
        @test isapprox(test_losses[1], golden[2]; rtol=1.0e-12)
    end

    @testset "checkpoints written during training" begin
        written = String[]
        checkpoint_fn(step) = (path = joinpath(dir, "step_$(step).jld2"); push!(written, path); path)
        train_losses!(gpu_golden_layers(), Adam(1.0e-3); epochs=2, gauge_paths=[raw_path],
                      gauge_format=:raw, checkpoint_every=1, checkpoint_fn=checkpoint_fn)
        @test length(written) == 2
        @test all(isfile, written)
        @test [JLD2.load(p, "opt_t") for p in written] == [1, 2]
    end

    @testset "resuming from a checkpoint continues the run" begin
        source = (; gauge_paths=[raw_path], gauge_format=:raw)
        continuous = train_losses!(gpu_golden_layers(), Adam(1.0e-3); epochs=3, source...)

        layers = gpu_golden_layers()
        opt = Adam(1.0e-3)
        first = train_losses!(layers, opt; epochs=1, source...)
        checkpoint = joinpath(dir, "checkpoint.jld2")
        save_checkpoint(checkpoint, layers, opt)

        layers = gpu_golden_layers()
        opt = Adam(1.0e-3)
        load_checkpoint!(checkpoint, layers, opt)
        rest = train_losses!(layers, opt; epochs=2, epoch_offset=1, source...)
        @test isapprox(vcat(first, rest), continuous; rtol=1.0e-12)
    end

    # With eps = 1 the Adam step is proportional to the gradient, so a batch
    # gradient that is a sum instead of a mean shows up.
    @testset "batch 2" begin
        linear_adam() = Adam(1.0e-3, 0.9, 0.999, 1.0)
        U_cold = Initialize_Gaugefields(3, 0, GOLDEN_DIMS...; condition="cold", cuda=false)
        single = train_losses!(gpu_golden_layers(), linear_adam(); epochs=2, gauge_cache=[U])
        doubled = train_losses!(gpu_golden_layers(), linear_adam(); epochs=2,
                                gauge_cache=[U, U], batch=2)
        # The batch gradient is summed over samples rather than averaged.
        @test_broken isapprox(doubled, single; rtol=1.0e-12)

        forward = train_losses!(gpu_golden_layers(), linear_adam(); epochs=2,
                                gauge_cache=[U, U_cold], batch=2)
        backward = train_losses!(gpu_golden_layers(), linear_adam(); epochs=2,
                                 gauge_cache=[U_cold, U], batch=2)
        # Some layers return a cached gradient buffer that the next sample
        # overwrites, so the result depends on the sample order.
        @test_broken isapprox(forward, backward; rtol=1.0e-12)
    end
    rm(dir; recursive=true)
end
