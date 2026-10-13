# Scripts written before the restructuring use the old submodule names.

using Test
using LMP

@testset "old submodule names" begin
    @test LMP.Layer_sequence._read_raw_gauge_to_gpu! === LMP._read_raw_gauge_to_gpu!
    @test LMP.Layer_sequence._make_gauge_staging === LMP._make_gauge_staging
    @test LMP.gpu.sync_W_to_gpu! === LMP.sync_W_to_gpu!
    @test LMP.activation.WF3x1 === LMP.WF3x1
    @test LMP.Layer_types.LossSumLayer === LMP.LossSumLayer
end

# Ported from the research branch (5845cb1): the configuration index counts
# from the start of training.
@testset "absolute training configuration index" begin
    @test LMP._training_config_index(0, 1, 1, 1, 320) == 1
    @test LMP._training_config_index(4000, 1, 1, 1, 320) == 161
    @test LMP._training_config_index(4000, 2, 1, 1, 320) == 162
    @test LMP._training_config_index(4000, 1, 2, 1, 320) == 1
    @test LMP._training_config_index(4000, 1, 2, 2, 320) == 2
end
