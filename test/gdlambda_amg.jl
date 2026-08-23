using Test
using LinearAlgebra
using LMP

const GD_SRC = normpath(joinpath(@__DIR__, "..", "src", "GDlambdaAMG.jl"))
const CCL_SRC = normpath(joinpath(@__DIR__, "..", "src", "CoarseCorrectionLoss.jl"))
const GD_EXPERIMENT = normpath(joinpath(
    @__DIR__, "..", "..", "scripts", "experiments",
    "evaluate_gdlambda_amg_learned24_vs_random24.jl",
))

@testset "GD-lambda-AMG source integration" begin
    @test isfile(GD_SRC)
    @test isfile(GD_EXPERIMENT)
    if isfile(GD_SRC)
        source = read(GD_SRC, String)
        @test occursin("struct ShiftedGamma5DiracOperator", source)
        @test occursin("struct Gamma5LeftShiftedDiracOperator", source)
        @test occursin("gamma5!(out)", source)
        @test occursin("axpby!(-A.shift", source)
        @test occursin("_coarse_gmres_matrixfree", source)
        @test occursin("restrict_f2c!", source)
        @test occursin("prolongate_c2f!", source)
        @test occursin("_gd_norm(x) = _field_norm(x)", source)
    end
    if isfile(CCL_SRC)
        source = read(CCL_SRC, String)
        @test occursin("coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing", source)
    end
    ddalpha_source = read(normpath(joinpath(@__DIR__, "..", "src", "DDalphaAMG.jl")), String)
    @test occursin("raw_test_vectors_out::Union{Nothing,Vector}=nothing", ddalpha_source)
    @test occursin("raw_test_vectors_out", read(GD_EXPERIMENT, String))
    if isfile(GD_EXPERIMENT)
        source = read(GD_EXPERIMENT, String)
        @test occursin("GD_OUTER_TOL", source)
        @test occursin("require_env", source)
        @test occursin("learned_24", source)
        @test occursin("standard_amg_24", source)
        @test occursin("raw_learned_24", source)
        @test occursin("refined_learned_24", source)
        @test occursin("GD_REFINE_CYCLES", source)
        @test occursin("GD_POST_SMOOTH", source)
        @test occursin("refinement_seconds", source)
        @test occursin("applications.png", source)
        @test occursin("wall_time.png", source)
        @test occursin("shared_random_direction", source)
        @test occursin("gamma5!(shifted_rhs)", source)
        @test occursin("Gamma5LeftShiftedDiracOperator", source)
        @test occursin("smoother=:vcycle", source)
        @test occursin("Refusing to overwrite", source)
        @test occursin("(8000epoch)", source)
        @test occursin("collect(401:410)", source)
    end
end

@testset "harmonic Ritz targets the closest eigenvalue" begin
    @test isdefined(LMP, :harmonic_ritz)
    if isdefined(LMP, :harmonic_ritz)
        A = Diagonal(ComplexF64[-3.0, -0.2, 1.0, 4.0])
        basis = [ComplexF64.(Matrix(I, 4, 4)[:, i]) for i in 1:4]
        Abasis = [A * v for v in basis]
        result = LMP.harmonic_ritz(basis, Abasis; target=0.0)
        @test result.value ≈ -0.2 atol=1e-12
        @test result.residual ≤ 1e-12
        @test abs(result.vector[2]) ≈ 1.0 atol=1e-12
    end
end

@testset "flexible GMRES solves and reports a direct residual" begin
    @test isdefined(LMP, :flexible_gmres!)
    if isdefined(LMP, :flexible_gmres!)
        A = ComplexF64[4 1; 1 3]
        b = ComplexF64[1, 2]
        x = zeros(ComplexF64, 2)
        apply_A! = (out, input) -> (mul!(out, A, input); out)
        apply_M! = (out, input, iteration) -> (copyto!(out, input); out)
        result = LMP.flexible_gmres!(x, apply_A!, b, apply_M!;
                                     tol=1e-12, maxiter=10, restart=2)
        @test result.converged
        @test result.verified_relative_residual ≤ 1e-12
        @test x ≈ A \ b atol=1e-11
        @test result.iterations ≤ 2
    end
end

@testset "paper-default shifted AMG cycle includes GMRES4 post-smoothing" begin
    @test isdefined(LMP, :shifted_amg_vcycle!)
    if isdefined(LMP, :shifted_amg_vcycle!)
        A = Diagonal(ComplexF64[1, 2, 4, 8, 16, 32])
        rhs = ones(ComplexF64, 6)
        out = zeros(ComplexF64, 6)
        operator_calls = Ref(0)
        apply_A! = function(destination, input)
            operator_calls[] += 1
            mul!(destination, A, input)
            return destination
        end
        coarse_correction! = function(destination, input)
            fill!(destination, 0)
            destination[1] = input[1]
            return (; coarse_applications=1)
        end

        result = LMP.shifted_amg_vcycle!(
            out, rhs, apply_A!, coarse_correction!; post_smooth_steps=4)

        @test result.post_iterations == 4
        @test result.coarse_applications == 1
        @test result.post_applications >= 4
        @test operator_calls[] == result.fine_applications
        @test isfinite(result.coarse_relative_residual)
        @test isfinite(result.final_relative_residual)
        @test result.final_relative_residual < result.coarse_relative_residual
        @test norm(rhs - A * out) / norm(rhs) ≈
              result.final_relative_residual atol=1e-13

        @test_throws ArgumentError LMP.shifted_amg_vcycle!(
            out, rhs, apply_A!, coarse_correction!; post_smooth_steps=3)
    end
end

@testset "generalized Davidson converges and restarts deterministically" begin
    @test isdefined(LMP, :generalized_davidson)
    if isdefined(LMP, :generalized_davidson)
        Q = Diagonal(ComplexF64[-3.0, -0.2, 1.0, 4.0])
        apply_Q! = (out, input) -> (mul!(out, Q, input); out)
        initial = [ComplexF64[1, 1, 1, 1], ComplexF64[1, -1, 2, -2]]
        correction_solver! = (out, residual, theta) -> (copyto!(out, residual);
            (; converged=true, iterations=0, applications=0))
        result = LMP.generalized_davidson(
            apply_Q!, initial; correction_solver!, tol=1e-10,
            max_expansions=12, max_subspace=4, retain=2)
        @test result.converged
        @test result.residual ≤ 1e-10
        @test abs(result.value) ≈ 0.2 atol=1e-9

        capped = LMP.generalized_davidson(
            apply_Q!, initial; correction_solver!, tol=1e-30,
            max_expansions=2, max_subspace=3, retain=2)
        @test !capped.converged
        @test capped.capped
        @test capped.restarts ≥ 1
    end
end

@testset "thick restart restores an orthonormal basis" begin
    candidates = [
        (; vector=ComplexF64[1, 0, 0]),
        (; vector=ComplexF64[1, 1e-3, 0]),
    ]
    basis = LMP._gd_restart_basis(candidates, 2)
    gram = [dot(basis[i], basis[j]) for i in 1:2, j in 1:2]
    @test gram ≈ Matrix{ComplexF64}(I, 2, 2) atol=1e-12
end
