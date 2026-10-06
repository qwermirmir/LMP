# The path sets the models are built from.

using Test
using LMP

@testset "path sets" begin
    steps(P) = [p[1] for p in P]
    for variant in (:long_only1248, :long_only2468, :corner33_ordered, :short_repeat4)
        P = LMP._default_lmp_paths(5; path_variant=variant)
        @test length(P) == 33
        @test steps(P)[1] == [0]
    end
    @test length(LMP._default_lmp_paths(2)) == 17
    @test length(LMP._default_lmp_paths(4)) == 25
    @test length(LMP._default_lmp_paths(5)) == 33
    @test steps(LMP._default_lmp_paths(5; path_variant=:long_only1248))[end] == fill(-4, 8)
    @test steps(LMP._default_lmp_paths(5; path_variant=:corner33_ordered))[end] == [-3, -4]
    @test_throws ArgumentError LMP._default_lmp_paths(5; path_variant=:unknown)
end
