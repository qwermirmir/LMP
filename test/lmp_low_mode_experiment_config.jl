using Test

include(joinpath(@__DIR__, "..", "research", "compare_lmp_to_low_modes.jl"))

cfg = default_config()
@test cfg.config_ids == [401, 405]
@test cfg.kappa == 0.157
@test cfg.nev == 6
@test cfg.eig_tol == 1e-10
@test cfg.cuda_visible_devices == "1"
@test occursin("16000epoch", cfg.checkpoint)
@test cfg.block_r == 4
