using Test
using LMP

const OPTIMIZER_SOURCE = read(joinpath(@__DIR__, "..", "src", "Optimizer.jl"), String)
const TRAINING_SOURCE = read(joinpath(@__DIR__, "..", "src", "Layer_sequence.jl"), String)
const LMP_SOURCE = read(joinpath(@__DIR__, "..", "src", "LMP.jl"), String)

@testset "Adam advances once per optimizer sweep" begin
    @test isdefined(LMP, :advance_step!)
    if isdefined(LMP, :advance_step!)
        opt = Adam()
        @test opt.t == 0
        @test LMP.advance_step!(opt) == 1
        @test LMP.advance_step!(opt) == 2
        @test opt.t == 2
    end

    @test count("opt.t += 1", OPTIMIZER_SOURCE) == 1
    @test occursin("advance_step!(opt::Adam)", OPTIMIZER_SOURCE)
    @test occursin("export Adam, update!, advance_step!", OPTIMIZER_SOURCE)
    @test occursin("export Opt, Adam, update!, advance_step!", LMP_SOURCE)

    # Five distinct training/update paths must each advance once per sweep.
    @test count("advance_step!(opt)", TRAINING_SOURCE) == 5
    @test occursin("import ..Optimizer: Opt, update!, advance_step!, Adam", TRAINING_SOURCE)

    # Warm-up initializes Adam buffers at a valid temporary step, then restores t.
    prealloc_start = findfirst("function prealloc_optimizer_state!", OPTIMIZER_SOURCE)
    prealloc_end = findnext("function sync_layers!", OPTIMIZER_SOURCE, last(prealloc_start))
    @test prealloc_start !== nothing
    @test prealloc_end !== nothing
    if prealloc_start !== nothing && prealloc_end !== nothing
        prealloc = OPTIMIZER_SOURCE[first(prealloc_start):first(prealloc_end)-1]
        @test occursin("advance_step!(opt)", prealloc)
        @test occursin("opt.t = t0", prealloc)
    end
end
