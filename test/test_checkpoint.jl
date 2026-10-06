# Checkpoints written from a CPU model.

using Test
using JLD2

include(joinpath(@__DIR__, "golden", "model.jl"))

@testset "CPU checkpoint" begin
    layers = golden_model().layers
    opt = Adam(1.0e-3)
    golden_train_step!(layers, opt)
    path = tempname() * ".jld2"
    save_checkpoint(path, layers, opt)
    saved = keys(JLD2.load(path))
    rm(path)

    @test "opt_t" in saved
    @test "layer_1_fermion_x_1" in saved
    # save_checkpoint stores PTLayer and DenseLinearLayer weights only when
    # they live on the GPU, so a CPU model loses them.
    @test_broken all("layer_$(i)_W" in saved
                     for (i, l) in enumerate(layers) if l isa Union{PTLayer,DenseLinearLayer})
end

@testset "checkpoint refuses a model with other paths" begin
    layers = golden_model().layers
    opt = Adam(1.0e-3)
    path = tempname() * ".jld2"
    save_checkpoint(path, layers, opt)
    @test haskey(JLD2.load(path), "layer_3_paths")

    other = golden_model(max_len=4).layers
    @test_throws ArgumentError load_checkpoint!(path, other, Adam(1.0e-3))
    rm(path)
end
