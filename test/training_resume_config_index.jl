using Test, LMP

const LS = LMP.Layer_sequence

@testset "absolute training configuration index" begin
    @test LS._training_config_index(0, 1, 1, 1, 320) == 1
    @test LS._training_config_index(4000, 1, 1, 1, 320) == 161
    @test LS._training_config_index(4000, 2, 1, 1, 320) == 162
    @test LS._training_config_index(4000, 1, 2, 1, 320) == 1
    @test LS._training_config_index(4000, 1, 2, 2, 320) == 2
end
