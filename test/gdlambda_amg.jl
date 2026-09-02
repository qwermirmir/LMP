using Test
using LinearAlgebra
using Random
using LMP
using Gaugefields
using LatticeDiracOperators

const GD_SRC = normpath(joinpath(@__DIR__, "..", "src", "GDlambdaAMG.jl"))
const GD_ADAPTER_SRC = normpath(joinpath(
    @__DIR__, "..", "src", "GDlambdaAMGDDalphaAdapter.jl"))
const CCL_SRC = normpath(joinpath(@__DIR__, "..", "src", "CoarseCorrectionLoss.jl"))
const GD_EXPERIMENT = normpath(joinpath(
    @__DIR__, "..", "..", "scripts", "experiments",
    "evaluate_gdlambda_amg_learned24_vs_random24.jl",
))

@testset "GD DDalpha adapter owns experiment-only behavior" begin
    @test isfile(GD_ADAPTER_SRC)
    @test isdefined(LMP, :gd_global_orthonormalize!)
    @test isdefined(LMP, :gd_adaptive_amg_setup)
    @test isdefined(LMP, :gd_coarse_gmres)
    @test isdefined(LMP, :gd_adaptive_vcycle)
    adapter_source = read(GD_ADAPTER_SRC, String)
    @test occursin("gd_adaptive_vcycle", adapter_source)
    @test !occursin("DDalphaAMG.vcycle(", adapter_source)
    @test occursin("arnoldi_mode=:auto", adapter_source)
    ddalpha_source = read(
        normpath(joinpath(@__DIR__, "..", "src", "DDalphaAMG.jl")), String)
    @test !occursin("raw_test_vectors_out", ddalpha_source)
    @test !occursin("_amg_global_orthonormalize!", ddalpha_source)
    @test !occursin("coarse_odd_even_relative_residual", ddalpha_source)
end

@testset "GD-lambda-AMG source integration" begin
    @test isfile(GD_SRC)
    @test isfile(GD_EXPERIMENT)
    if isfile(GD_SRC)
        source = read(GD_SRC, String)
        @test occursin("struct ShiftedGamma5DiracOperator", source)
        @test occursin("struct Gamma5LeftShiftedDiracOperator", source)
        @test occursin("gamma5!(out)", source)
        @test occursin("axpby!(-A.shift", source)
        @test occursin("gd_coarse_gmres", source)
        @test occursin("coarse_odd_even=true", source)
        @test !occursin("Coarsest DD-alphaAMG solve did not reach", source)
        @test occursin("restrict_f2c!", source)
        @test occursin("prolongate_c2f!", source)
        @test occursin("_gd_norm(x) = _field_norm(x)", source)
    end
    if isfile(CCL_SRC)
        source = read(CCL_SRC, String)
        @test occursin("history_callback::Union{Nothing,Function}=nothing", source)
        @test occursin("true_relative_residual", source)
        @test occursin("estimated_relative_residual <= tol", source)
        @test occursin("true_relative_residual <= tol", source)
    end
    if isfile(CCL_SRC)
        source = read(CCL_SRC, String)
        @test occursin("coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing", source)
    end
    ddalpha_source = read(normpath(joinpath(@__DIR__, "..", "src", "DDalphaAMG.jl")), String)
    @test !occursin("raw_test_vectors_out", ddalpha_source)
    @test !occursin("raw_test_vectors_out", read(GD_EXPERIMENT, String))
    @test occursin("gd_adaptive_amg_setup", read(GD_EXPERIMENT, String))
    if isfile(GD_EXPERIMENT)
        source = read(GD_EXPERIMENT, String)
        @test occursin("GD_OUTER_TOL", source)
        @test occursin("GD_NEV", source)
        @test occursin("GD_EIGENPAIR_AUDIT", source)
        @test occursin("gd_eigenpair_audit", source)
        @test occursin("q2_identity_error", source)
        @test occursin("locked_count", source)
        @test occursin("GD run did not converge", source)
        @test occursin("Independent locked-pair audit failed", source)
        @test occursin("pair.q2_residual <= GD_OUTER_TOL", source)
        @test occursin("require_env", source)
        @test occursin("standard_amg_24", source)
        @test occursin("learned_refined_amg_24", source)
        @test !occursin("raw_learned_24", source)
        @test !occursin("refined_learned_24", source)
        @test !occursin("for _ in 1:GD_REFINE_CYCLES", source)
        @test occursin("length(rows) == 2 * length(GD_CONFIG_IDS)", source)
        @test occursin("GD_POST_SMOOTH", source)
        @test occursin("refinement_seconds", source)
        @test occursin("applications.png", source)
        @test occursin("wall_time.png", source)
        @test occursin("shared_random_direction", source)
        @test occursin("Gamma5LeftShiftedDiracOperator", source)
        @test occursin("gamma5!(shifted_rhs)", source)
        @test occursin("shifted_amg_vcycle!", source)
        @test occursin("post_smooth_steps=4", source)
        @test occursin("post_smoother_applications", source)
        @test occursin("post_smoother_iterations", source)
        @test occursin("preconditioner_calls", source)
        @test occursin("last_preconditioner_residual_ratio", source)
        @test occursin("gd_adaptive_amg_setup", source)
        @test occursin("use_chirality=true", source)
        @test occursin("setup_smoother=:sap", source)
        @test occursin("setup_sap_iters=3", source)
        @test occursin("setup_block_solver=:mr", source)
        @test occursin("inner_verified_relative_residual", source)
        @test occursin("coarse_verified_relative_residual", source)
        @test occursin("coarse_schur_relative_residual", source)
        @test occursin("GD_COARSE_RESTART == 100", source)
        @test !occursin("GD_COARSE_MAXITER == 0", source)
        @test !occursin("Paper-exact coarse solve must run until", source)
        @test occursin("GD_AMG_NVCYCLE == 6", source)
        @test occursin("gd_warmup!", source)
        @test occursin("max_expansions=1", source)
        @test occursin("warmup_io = IOBuffer()", source)
        @test occursin("hierarchy_update!", source)
        @test occursin("setup_mode=:direct_near_null", source)
        @test occursin("coarse_vectors=vectors", source)
        @test occursin("Refusing to overwrite", source)
        @test occursin("(8000epoch)", source)
        @test occursin("collect(401:410)", source)
        @test occursin("require_convergence::Bool=false", source)
    end
end

@testset "coarse tolerance is not a mandatory convergence gate" begin
    source = read(joinpath(@__DIR__, "..", "src", "DDalphaAMG.jl"), String)
    @test !occursin("safety_max_restarts", source)
    @test !occursin("coarse GMRES failed to reach tolerance", source)
    @test !occursin("maxiter == 0 ?", source)
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

@testset "harmonic Ritz pencil is equation 3.5 without transformation" begin
    @test isdefined(LMP, :_gd_hermitian_harmonic_eigen)
    @test isdefined(LMP, :harmonic_ritz_pencil)
    if isdefined(LMP, :harmonic_ritz_pencil)
        Q = ComplexF64[2 1im 0; -1im -1 0.5; 0 0.5 3]
        raw = [ComplexF64[1, 1, 0], ComplexF64[0, 1, 1]]
        V = LMP._gd_orthonormal_copy(raw)
        QV = [Q * v for v in V]
        C, B = LMP.harmonic_ritz_pencil(V, QV)
        Vmat = hcat(V...)
        QVmat = hcat(QV...)

        @test C ≈ QVmat' * QVmat atol=1e-13
        @test B ≈ QVmat' * Vmat atol=1e-13
        direct_values = eigvals(QVmat' * QVmat, QVmat' * Vmat)
        result = LMP.harmonic_ritz(V, QV)
        actual_values = [candidate.value for candidate in result.candidates]
        @test sort(actual_values; by=abs) ≈
              sort(real.(direct_values); by=abs) atol=1e-12
    end
end

@testset "Hermitian harmonic extraction preserves all finite real modes" begin
    Q = Diagonal(ComplexF64[-3.0, -0.2, 0.1, 2.0])
    V = [ComplexF64[i == j for i in 1:4] for j in 1:4]
    QV = [Q * vector for vector in V]
    result = LMP.harmonic_ritz(V, QV)
    @test length(result.candidates) == 4
    @test sort([candidate.value for candidate in result.candidates]) ≈
          [-3.0, -0.2, 0.1, 2.0] atol=1e-13
end

@testset "harmonic Ritz keeps finite pairs from a singular pencil" begin
    Q = Diagonal(ComplexF64[0, 1])
    V = [ComplexF64[1, 0], ComplexF64[0, 1]]
    QV = [Q * vector for vector in V]
    result = LMP.harmonic_ritz(V, QV)
    @test result.value ≈ 1 atol=1e-13
    @test result.residual ≤ 1e-13
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

@testset "paper correction solve stops at factor 0.1 or five iterations" begin
    rng = MersenneTwister(200408146)
    A = randn(rng, ComplexF64, 9, 9) + 3I
    b = randn(rng, ComplexF64, 9)
    x = zeros(ComplexF64, 9)
    result = LMP.flexible_gmres!(
        x, (out, input) -> (mul!(out, A, input); out), b,
        (out, input, iteration) -> (copyto!(out, input); out);
        tol=0.1, maxiter=5, restart=5)
    direct = norm(b - A * x) / norm(b)
    @test result.iterations <= 5
    @test result.converged || result.iterations == 5
    @test result.verified_relative_residual ≈ direct atol=1e-13
    @test result.converged == (direct <= 0.1)
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

        # Independent explicit form of E=(I-MA)^4(I-CA), with no pre-smooth.
        C = zeros(ComplexF64, 6, 6); C[1, 1] = 1
        coarse_only = C * rhs
        post_rhs = rhs - A * coarse_only
        K = hcat(post_rhs, A * post_rhs, A^2 * post_rhs, A^3 * post_rhs)
        delta_reference = K * ((A * K) \ post_rhs)
        @test out ≈ coarse_only + delta_reference atol=1e-11

        @test_throws ArgumentError LMP.shifted_amg_vcycle!(
            out, rhs, apply_A!, coarse_correction!; post_smooth_steps=3)
    end
end

@testset "shifted chiral coarse stencil has the correct Gamma5c signs" begin
    @test isdefined(LMP, :shifted_chiral_coarse_stencil)
    if isdefined(LMP, :shifted_chiral_coarse_stencil)
        DD = LMP.DDalphaAMG
        Nv = 2
        Nblocks = 16
        NBX = NBY = NBZ = NBT = 2
        self = repeat(reshape(ComplexF64[4 0; 0 5], 2, 2, 1), 1, 1, Nblocks)
        plus = zeros(ComplexF64, Nv, Nv, Nblocks, 4)
        minus = zeros(ComplexF64, Nv, Nv, Nblocks, 4)
        nb_plus = zeros(Int, Nblocks, 4)
        nb_minus = zeros(Int, Nblocks, 4)
        for b in 1:Nblocks, mu in 1:4
            nb_plus[b, mu] = DD._neighbor_block(b, mu, 1, NBX, NBY, NBZ, NBT)
            nb_minus[b, mu] = DD._neighbor_block(b, mu, -1, NBX, NBY, NBZ, NBT)
        end
        op = DD.CoarseStencilOperator(
            Nv, Nblocks, NBX, NBY, NBZ, NBT,
            self, plus, minus, nb_plus, nb_minus)
        plus[1, 2, 1, 1] = 2
        plus[2, 1, 1, 1] = 3
        shifted = LMP.shifted_chiral_coarse_stencil(op, 0.25)

        @test shifted.self[1, 1, :] == fill(3.75 + 0im, Nblocks)
        @test shifted.self[2, 2, :] == fill(5.25 + 0im, Nblocks)
        @test shifted.plus[1, 2, 1, 1] == 2
        @test shifted.plus[2, 1, 1, 1] == 3
        @test op.self[1, 1, 1] == 4.0
        @test op.self[2, 2, 1] == 5.0
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

@testset "diagnostic callback observes extraction lock and restart without changing results" begin
    rng = MersenneTwister(20260830)
    Q = Diagonal(ComplexF64[-4.0, -2.0, -0.7, 0.2, 1.1, 3.0])
    apply_Q! = (out, input) -> (mul!(out, Q, input); out)
    initial = [randn(rng, ComplexF64, 6) for _ in 1:2]
    direction = randn(rng, ComplexF64, 6)
    correction_solver! = (out, residual, theta) -> begin
        copyto!(out, residual)
        return (; iterations=1, applications=1)
    end
    events = NamedTuple[]
    traced = LMP.generalized_davidson(
        apply_Q!, initial; initial_direction=direction, correction_solver!,
        tol=1e-30, max_expansions=3, max_subspace=4, retain=2,
        diagnostic_callback=event -> push!(events, event))
    plain = LMP.generalized_davidson(
        apply_Q!, initial; initial_direction=direction, correction_solver!,
        tol=1e-30, max_expansions=3, max_subspace=4, retain=2)

    @test any(event -> event.stage == :extraction, events)
    @test any(event -> event.stage == :pre_restart, events)
    @test any(event -> event.stage == :post_restart, events)
    @test all(issorted(abs.(getproperty.(event.candidates, :value)))
              for event in events if !isempty(event.candidates))
    @test traced.values == plain.values
    @test traced.residual == plain.residual
    @test traced.expansions == plain.expansions
    @test traced.restarts == plain.restarts
    @test traced.q_applications == plain.q_applications

    lock_events = NamedTuple[]
    exact_initial = [ComplexF64[i == j for i in 1:6] for j in 1:3]
    exact_direction = ComplexF64[0, 0, 0, 1, 0, 0]
    locked = LMP.generalized_davidson(
        apply_Q!, exact_initial; initial_direction=exact_direction,
        correction_solver!, tol=1e-13, max_expansions=2,
        max_subspace=5, retain=3, nev=2,
        diagnostic_callback=event -> push!(lock_events, event))
    @test locked.converged
    @test any(event -> event.stage == :post_lock, lock_events)
    post_lock = first(filter(event -> event.stage == :post_lock, lock_events))
    @test length(post_lock.locked_values) == 1
    @test all(candidate -> hasproperty(candidate, :value) &&
                           hasproperty(candidate, :residual) &&
                           hasproperty(candidate, :coefficient_norm) &&
                           hasproperty(candidate, :locked_overlap),
              post_lock.candidates)
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

@testset "thick restart retains matching V and QV without new Q applies" begin
    rng = MersenneTwister(77)
    Q = Hermitian(randn(rng, ComplexF64, 6, 6))
    raw = LMP._gd_orthonormal_copy(
        [randn(rng, ComplexF64, 6) for _ in 1:4])
    images = [Q * vector for vector in raw]
    candidates = LMP.harmonic_ritz(raw, images).candidates
    basis, Qbasis = LMP._gd_restart_basis_images(candidates, 3)
    retained = hcat((candidate.vector for candidate in candidates[1:3])...)
    expected_basis = Matrix(qr(retained).Q)[:, 1:3]
    actual_basis = hcat(basis...)
    @test actual_basis * actual_basis' ≈
          expected_basis * expected_basis' atol=1e-11
    @test hcat(Qbasis...) ≈ Q * hcat(basis...) atol=1e-11
    @test hcat(basis...)' * hcat(basis...) ≈
          Matrix{ComplexF64}(I, 3, 3) atol=1e-11
end

@testset "Algorithm 2 inserts t separately and restarts 50 to 30" begin
    rng = MersenneTwister(200408146)
    Q = Diagonal(ComplexF64.(range(-4.0, 4.0; length=80)))
    V0 = [randn(rng, ComplexF64, 80) for _ in 1:24]
    t0 = randn(rng, ComplexF64, 80)
    trace = NamedTuple[]
    correction_solver! = (out, residual, theta) -> begin
        copyto!(out, residual)
        return (; iterations=1, applications=1)
    end
    result = LMP.generalized_davidson(
        (out, input) -> (mul!(out, Q, input); out), V0;
        initial_direction=t0,
        correction_solver!, tol=1e-30, max_expansions=27,
        max_subspace=50, retain=30,
        history_callback=record -> push!(trace, record))

    @test first(trace).subspace == 25
    restart_index = findfirst(record -> record.restarts == 1, trace)
    @test restart_index !== nothing
    @test trace[restart_index].subspace == 31
    @test result.restarts == 1
    # Algorithm 2 retains (QV)s together with Vs, so restart performs no
    # additional Q applications: 25 initial + 27 expansions + 28 direct checks.
    @test result.q_applications == 80
end

@testset "Algorithm 2 solves correction before thick restart" begin
    Q = Diagonal(ComplexF64[-4, -2, -0.5, 1, 3, 5])
    q_applications = Ref(0)
    correction_q_counts = Int[]
    apply_Q! = function(out, input)
        q_applications[] += 1
        mul!(out, Q, input)
        return out
    end
    correction_solver! = function(out, residual, theta)
        push!(correction_q_counts, q_applications[])
        copyto!(out, residual)
        return (; iterations=1, applications=1)
    end
    initial = [ComplexF64[1, 1, 1, 1, 1, 1],
               ComplexF64[1, -1, 2, -2, 3, -3]]
    direction = ComplexF64[1, 2, 3, 4, 5, 6]

    LMP.generalized_davidson(
        apply_Q!, initial; initial_direction=direction,
        correction_solver!, tol=1e-30, max_expansions=2,
        max_subspace=4, retain=2)

    # Algorithm 2 line 14 computes t with the current four-vector space;
    # only afterwards do lines 15--21 restart and form the retained images.
    # Each correction follows a fresh direct Q action on the selected Ritz
    # vector; the restart-image refresh still occurs only after correction.
    @test correction_q_counts == [4, 6]
end

@testset "Algorithm 2 locks multiple eigenpairs and excludes them" begin
    rng = MersenneTwister(314159)
    Q = Diagonal(ComplexF64[-3.0, -0.12, 0.31, 1.7, 2.4, 4.0])
    initial = [randn(rng, ComplexF64, 6) for _ in 1:2]
    direction = randn(rng, ComplexF64, 6)
    updates = NamedTuple[]
    correction_solver! = function(out, residual, theta)
        # Deliberately inexact, as in Algorithm 2.  An exact solve gives
        # t = u for r=(Q-theta*I)u and cannot expand the search space.
        copyto!(out, residual)
        return (; iterations=1, applications=1)
    end
    result = LMP.generalized_davidson(
        (out, input) -> (mul!(out, Q, input); out), initial;
        initial_direction=direction, correction_solver!, nev=2,
        ntv=1, tol=1e-8, max_expansions=40, max_subspace=5, retain=2,
        replacement_direction! = (out, attempt) -> begin
            fill!(out, 0)
            out[mod1(attempt, length(out))] = 1
        end,
        hierarchy_update! = (vectors, values, target) ->
            push!(updates, (; vectors=copy(vectors), values=copy(values), target)))

    @test result.converged
    @test length(result.values) == 2
    @test sort(result.values; by=abs) ≈ [-0.12, 0.31] atol=1e-9
    @test abs(dot(result.vectors[1], result.vectors[2])) ≤ 1e-10
    @test all(norm(Q * result.vectors[i] - result.values[i] * result.vectors[i]) ≤ 1e-8
              for i in 1:2)
    @test !isempty(updates)
    @test all(length(update.vectors) == 1 for update in updates)
    @test maximum(record.basis_orthogonality for record in result.history) ≤ 1e-11
    @test maximum(record.locked_orthogonality for record in result.history) ≤ 1e-11
    @test maximum(record.basis_locked_overlap for record in result.history) ≤ 1e-11
    measured = filter(record -> isfinite(record.correction_basis_overlap),
                      result.history)
    @test !isempty(measured)
    @test maximum(record.correction_basis_overlap for record in measured) ≤ 1e-11
    @test maximum(record.correction_locked_overlap for record in measured) ≤ 1e-11
end

@testset "Algorithm 2 rebuild subset is closest to the new target" begin
    vectors = Any[ComplexF64[i] for i in 1:5]
    values = [-0.8, -0.12, 0.05, 0.31, 1.2]
    selected_vectors, selected_values =
        LMP._gd_interpolation_subset(vectors, values, 0.28, 2)
    @test selected_values == [0.31, 0.05]
    @test selected_vectors == Any[ComplexF64[4], ComplexF64[3]]
    @test_throws ArgumentError LMP._gd_interpolation_subset(
        vectors, values, 0.0, 6)
end

@testset "locking distinguishes orthogonal vectors with repeated eigenvalues" begin
    locked = Any[ComplexF64[1, 0, 0]]
    same_pair = (; value=0.2, vector=ComplexF64[1, 0, 0])
    repeated_but_distinct = (; value=0.2, vector=ComplexF64[0, 1, 0])
    @test LMP._gd_candidate_is_locked(same_pair, locked)
    @test !LMP._gd_candidate_is_locked(repeated_but_distinct, locked)
end

function paper_identity_cpu_problem()
    U = Initialize_Gaugefields(3, 0, 4, 4, 4, 4;
                               condition="cold", cuda=false)
    x = Initialize_pseudofermion_fields(U[1], "Wilson", nowing=true)
    params = Dict{String,Any}(
        "Dirac_operator" => "Wilson", "κ" => 0.12,
        "faster version" => true, "verbose_level" => 0)
    return x, Dirac_operator(U, x, params)
end

paper_field_distance(a, b) = sqrt(max(0.0,
    real(inner_product(a, a)) + real(inner_product(b, b)) -
    2real(inner_product(a, b))))

@testset "GD adapter globally orthonormalizes prototype fields" begin
    template, _ = paper_identity_cpu_problem()
    vectors = [similar(template) for _ in 1:3]
    foreach(LMP.DDalphaAMG.random_fermion!, vectors)
    axpby!(1, vectors[1], 1, vectors[2])
    LMP.gd_global_orthonormalize!(vectors)
    gram = [inner_product(vectors[i], vectors[j])
            for i in 1:3, j in 1:3]
    @test gram ≈ Matrix{ComplexF64}(I, 3, 3) atol=1e-11

    duplicate = [LMP.DDalphaAMG._copy_wf(vectors[1]) for _ in 1:2]
    @test_throws ArgumentError LMP.gd_global_orthonormalize!(duplicate)
end

@testset "GD random and supplied prototypes share one adaptive path" begin
    template, D = paper_identity_cpu_problem()
    seed = 20260831
    common = (
        nsmooth=1, nvcycle=1,
        setup_smoother=:sap, setup_sap_iters=1,
        setup_block_solver=:mr, setup_block_mr_steps=1,
        vcycle_update_smoother=:sap, vcycle_post_sap_iters=1,
        vcycle_post_block_mr_steps=1,
        setup_coarse_tol=0.5, setup_coarse_maxiter=1,
        setup_coarse_restart=2,
        use_chirality=false, block_normalize=true, verbose=false,
    )
    Random.seed!(seed)
    generated = LMP.gd_adaptive_amg_setup(D, template, 2, 2; common...)
    Random.seed!(seed)
    supplied = [similar(template) for _ in 1:2]
    foreach(LMP.DDalphaAMG.random_fermion!, supplied)
    supplied_before = [LMP.DDalphaAMG._copy_wf(vector) for vector in supplied]
    injected = LMP.gd_adaptive_amg_setup(
        D, template, 2, 2; prototypes=supplied, common...)
    @test length(generated.prototypes) == length(injected.prototypes) == 2
    @test all(paper_field_distance(
        generated.prototypes[i], injected.prototypes[i]) < 1e-11 for i in 1:2)
    @test all(paper_field_distance(supplied[i], supplied_before[i]) < 1e-13
              for i in 1:2)
    @test generated.setup.coarse_stencil.self ≈
          injected.setup.coarse_stencil.self atol=1e-11
    @test generated.setup.coarse_stencil.plus ≈
          injected.setup.coarse_stencil.plus atol=1e-11
    @test generated.setup.coarse_stencil.minus ≈
          injected.setup.coarse_stencil.minus atol=1e-11
end

@testset "paper fine and coarse operator identities on Wilson fields" begin
    DD = LMP.DDalphaAMG
    template, D = paper_identity_cpu_problem()
    x = similar(template); DD.random_fermion!(x)
    theta = 0.157

    qx_direct = similar(x); mul!(qx_direct, D, x); LMP.gamma5!(qx_direct)
    qx_wrapper = similar(x)
    mul!(qx_wrapper, LMP.ShiftedGamma5DiracOperator(D, 0.0), x)
    @test paper_field_distance(qx_direct, qx_wrapper) ≤ 1e-13

    left = similar(x)
    mul!(left, LMP.Gamma5LeftShiftedDiracOperator(D, theta, template), x)
    right = similar(x); axpby!(1, qx_direct, 0, right)
    axpby!(-theta, x, 1, right); LMP.gamma5!(right)
    @test paper_field_distance(left, right) ≤ 1e-12

    prototypes = [similar(template) for _ in 1:2]
    foreach(DD.random_fermion!, prototypes)
    setup = LMP.setup_DDalphaAMG(D, template, 2, 2;
        setup_mode=:direct_near_null, coarse_vectors=prototypes,
        nsmooth=0, nvcycle=0, use_chirality=true,
        precompute_coarse=true, verbose=false)
    c = DD._coarse_vector(setup)
    c.data .= randn(MersenneTwister(19), ComplexF64, length(c.data))
    px = LMP.prolongate_c2f(setup, c)
    rx = LMP.restrict_f2c(setup, x)
    @test dot(c.data, rx.data) ≈ inner_product(px, x) atol=1e-11

    assembled = similar(c)
    DD.apply_coarse_stencil!(assembled, setup.coarse_stencil, c)
    dpx = similar(px); mul!(dpx, D, px)
    matrix_free = LMP.restrict_f2c(setup, dpx)
    @test norm(assembled.data - matrix_free.data) /
          norm(matrix_free.data) ≤ 1e-11

    shifted = LMP.shifted_chiral_coarse_stencil(setup.coarse_stencil, theta)
    exact = DD._coarse_vector(setup)
    exact.data .= randn(MersenneTwister(23), ComplexF64, length(exact.data))
    shifted_rhs = similar(exact)
    DD.apply_coarse_stencil!(shifted_rhs, shifted, exact)
    original = setup.coarse_stencil
    computed = try
        setup.coarse_stencil = shifted
        LMP.gd_coarse_gmres(setup, nothing, shifted_rhs;
            tol=1e-10, maxiter=10, restart=100, coarse_odd_even=false,
            arnoldi_mode=:scalar)
    finally
        setup.coarse_stencil = original
    end
    shifted_image = similar(exact)
    DD.apply_coarse_stencil!(shifted_image, shifted, computed)
    @test norm(shifted_rhs.data - shifted_image.data) /
          norm(shifted_rhs.data) ≤ 1e-10

    correction_out = similar(x)
    correction = LMP.shifted_coarse_correction!(
        correction_out, x,
        LMP.Gamma5LeftShiftedDiracOperator(D, theta, template), setup;
        coarse_tol=0.1, coarse_maxiter=2, coarse_restart=10)
    @test correction.coarse_verified_relative_residual ≈
          correction.coarse_original_relative_residual atol=1e-14
end

@testset "harmonic vector and residual are independently reconstructed" begin
    rng = MersenneTwister(2718)
    Q = Hermitian(randn(rng, ComplexF64, 7, 7))
    V = LMP._gd_orthonormal_copy(
        [randn(rng, ComplexF64, 7) for _ in 1:4])
    QV = [Q * v for v in V]
    result = LMP.harmonic_ritz(V, QV)
    Vmat, QVmat = hcat(V...), hcat(QV...)
    u = Vmat * result.coefficients
    u ./= norm(u)
    theta = result.value
    @test abs(abs(dot(u, result.vector)) - 1) ≤ 1e-12
    @test result.residual ≈ norm(Q * result.vector - theta * result.vector) atol=1e-12
    @test norm(result.vector) ≈ 1 atol=1e-13
end

@testset "direct Ritz verification rejects stale stored operator images" begin
    Q = Diagonal(ComplexF64[-2.0, 0.25, 3.0])
    vector = normalize(ComplexF64[1, 2, 3])
    stale = zeros(ComplexF64, 3)
    candidate = (; value=0.2, vector, Avector=stale,
                 residual=norm(stale - 0.2vector), coefficients=ComplexF64[1])
    verified = LMP._gd_direct_ritz_candidate(
        (out, input) -> (mul!(out, Q, input); out), candidate)
    @test verified.Avector ≈ Q * vector atol=1e-14
    @test verified.residual ≈ norm(Q * vector - 0.2vector) atol=1e-14
    @test verified.action_mismatch > 0.9
end

@testset "basis image refresh restores the exact paired Q action" begin
    Q = Diagonal(ComplexF64[-2.0, 0.25, 3.0])
    basis = [ComplexF64[1, 0, 0], ComplexF64[0, 1, 0]]
    images = [zeros(ComplexF64, 3), ones(ComplexF64, 3)]
    applications = LMP._gd_refresh_basis_images!(
        images, (out, input) -> (mul!(out, Q, input); out), basis)
    @test applications == 2
    @test images[1] == Q * basis[1]
    @test images[2] == Q * basis[2]
end

@testset "locking removes one Ritz direction in coefficient space" begin
    Q = Diagonal(ComplexF64[-2.0, 0.25, 3.0])
    basis = [ComplexF64[i == j for i in 1:3] for j in 1:3]
    images = [Q * vector for vector in basis]
    coefficients = normalize(ComplexF64[1, 2im, 3])
    locked = hcat(basis...) * coefficients
    reduced_basis, reduced_images = LMP._gd_remove_ritz_direction(
        basis, images, coefficients)
    V = hcat(reduced_basis...)
    QV = hcat(reduced_images...)
    @test size(V) == (3, 2)
    @test V' * V ≈ Matrix{ComplexF64}(I, 2, 2) atol=1e-13
    @test norm(V' * locked) ≤ 1e-13
    @test QV ≈ Q * V atol=1e-13
end

@testset "dependent correction uses a deterministic replacement direction" begin
    Q = Diagonal(ComplexF64[-0.4, -0.1, 0.2, 0.7, 1.3])
    initial = [ComplexF64[1, 0, 0, 1, 0],
               ComplexF64[0, 1, 0, 0, 1]]
    direction = ComplexF64[0, 0, 1, 1, 0]
    replacements = Ref(0)
    result = LMP.generalized_davidson(
        (out, input) -> (mul!(out, Q, input); out), initial;
        initial_direction=direction,
        correction_solver! = (out, residual, theta) ->
            (fill!(out, 0); (; iterations=0, applications=0)),
        replacement_direction! = (out, attempt) -> begin
            replacements[] += 1
            fill!(out, 0)
            out[min(3 + attempt, 5)] = 1
        end,
        tol=1e-12, max_expansions=10, max_subspace=5,
        retain=3, nev=1, ntv=1)
    @test result.converged
    @test minimum(abs.(real.(diag(Q)) .- result.value)) ≤ 1e-13
    @test norm(Q * result.vector - result.value * result.vector) ≤ 1e-12
    @test replacements[] > 0
end

@testset "all already-converged candidates lock before another correction" begin
    Q = Diagonal(ComplexF64[-0.4, -0.1, 0.2, 0.7])
    initial = [ComplexF64[i == j for i in 1:4] for j in 1:3]
    direction = ComplexF64[0, 0, 0, 1]
    corrections = Ref(0)
    result = LMP.generalized_davidson(
        (out, input) -> (mul!(out, Q, input); out), initial;
        initial_direction=direction,
        correction_solver! = (out, residual, theta) -> begin
            corrections[] += 1
            fill!(out, 0)
            return (; iterations=0, applications=0)
        end,
        tol=1e-13, max_expansions=5, max_subspace=5,
        retain=3, nev=4, ntv=1)
    @test result.converged
    @test corrections[] == 0
    # Four initial basis images and four direct Ritz checks. Locking must
    # transform the paired V/QV data without reapplying Q.
    @test result.q_applications == 8
    @test sort(result.values; by=abs) ≈ [-0.1, 0.2, -0.4, 0.7] atol=1e-13
end

@testset "locked eigenpair audit independently verifies Q and Q squared" begin
    Q = Diagonal(ComplexF64[-0.4, -0.1, 0.2, 0.7])
    vectors = [ComplexF64[1, 0, 0, 0], ComplexF64[0, 1, 0, 0]]
    values = [-0.4, -0.1]
    audit = LMP.gd_eigenpair_audit(
        (out, input) -> (mul!(out, Q, input); out), vectors, values)

    @test length(audit.pairs) == 2
    @test maximum(pair.q_residual for pair in audit.pairs) ≤ 1e-14
    @test maximum(pair.q2_residual for pair in audit.pairs) ≤ 1e-14
    @test maximum(pair.q2_identity_error for pair in audit.pairs) ≤ 1e-14
    @test audit.orthogonality_error ≤ 1e-14

    perturbed = [normalize(ComplexF64[1, 0.01, 0, 0]), vectors[2]]
    perturbed_audit = LMP.gd_eigenpair_audit(
        (out, input) -> (mul!(out, Q, input); out), perturbed, values)
    @test perturbed_audit.pairs[1].q_residual > 1e-4
    @test perturbed_audit.pairs[1].q2_identity_error ≤ 1e-14
end
