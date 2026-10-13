#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "CUDA_VISIBLE_DEVICES", "1")

using Dates
using LinearAlgebra
using Printf
using Random
using SHA
using CUDA
using JLD2
using Plots
using Gaugefields
using LatticeDiracOperators
using LMP
using LMP.SubspaceSimilarity

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

Base.@kwdef mutable struct ExperimentConfig
    config_ids::Vector{Int} = [401, 405]
    kappa::Float64 = 0.157
    nev::Int = 6
    eig_tol::Float64 = 1e-10
    subspace_dim::Int = 160
    keep_dim::Int = 48
    max_restarts::Int = 200
    rng_seed::Int = 20260722
    cuda_visible_devices::String = "1"
    checkpoint::String = "/data1/choi/LMP_path33_kappa0158_6eig_scratch_20260614/(16x32)32PTlayer33path(kappa=0.158)(16000epoch)(maxlen5)(xavier_init)(6eig)(320config)(r=2)_final.jld2"
    gauge_dir::String = "/data1/choi/heatbath_ensemble_beta6.0/raw_gauge"
    output_root::String = joinpath(PROJECT_ROOT, "results")
    max_len::Int = 5
    block_r::Int = 4
    nx::Int = 16
    ny::Int = 16
    nz::Int = 16
    nt::Int = 32
    pes::NTuple{4,Int} = (4, 4, 4, 4)
    smoke::Bool = false
    print_settings_only::Bool = false
end

default_config() = ExperimentConfig()

gauge_path(cfg::ExperimentConfig, id::Int) = joinpath(cfg.gauge_dir, "config_$(id).gbin")
eigensystem_path(cfg::ExperimentConfig, id::Int) = joinpath(
    cfg.output_root, "eigenvectors", "lmp_similarity_cfg$(id)_k157_nev$(cfg.nev).jld2")
summary_path(cfg::ExperimentConfig) = joinpath(
    cfg.output_root, "tables", "lmp_low_mode_similarity_cfg401_cfg405_k157.csv")
plot_stem(cfg::ExperimentConfig) = joinpath(
    cfg.output_root, "plots", "lmp_low_mode_similarity_cfg401_cfg405_k157")
settings_path(cfg::ExperimentConfig) = joinpath(
    cfg.output_root, "diagnostics", "lmp_low_mode_similarity_cfg401_cfg405_k157", "settings.txt")

function validate_config(cfg::ExperimentConfig; require_files::Bool=true)
    cfg.config_ids == [401, 405] || cfg.smoke ||
        throw(ArgumentError("full run requires configurations 401 and 405"))
    cfg.kappa == 0.157 || throw(ArgumentError("this comparison is fixed at kappa=0.157"))
    0 < cfg.nev <= cfg.keep_dim < cfg.subspace_dim ||
        throw(ArgumentError("need 0 < nev <= keep_dim < subspace_dim"))
    cfg.eig_tol > 0 || throw(ArgumentError("eig_tol must be positive"))
    cfg.cuda_visible_devices == "1" ||
        throw(ArgumentError("physical GPU 1 is required, got $(cfg.cuda_visible_devices)"))
    if require_files
        isfile(cfg.checkpoint) || error("checkpoint not found: $(cfg.checkpoint)")
        for id in cfg.config_ids
            isfile(gauge_path(cfg, id)) || error("gauge not found: $(gauge_path(cfg, id))")
        end
    end
    return cfg
end

function file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function settings_text(cfg::ExperimentConfig)
    io = IOBuffer()
    println(io, "LMP low-mode comparison settings")
    println(io, "  physical GPU       = ", cfg.cuda_visible_devices)
    println(io, "  configurations     = ", join(cfg.config_ids, ", "))
    println(io, "  lattice            = $(cfg.nx)x$(cfg.ny)x$(cfg.nz)x$(cfg.nt)")
    println(io, "  operator            = D' * D")
    println(io, "  Wilson kappa        = ", cfg.kappa)
    println(io, "  modes               = ", cfg.nev)
    println(io, "  relative eig tol    = ", cfg.eig_tol)
    println(io, "  Krylov dimensions   = subspace $(cfg.subspace_dim), keep $(cfg.keep_dim)")
    println(io, "  maximum restarts    = ", cfg.max_restarts)
    println(io, "  RNG seed            = ", cfg.rng_seed)
    println(io, "  checkpoint training = kappa 0.158, 16000 steps")
    println(io, "  LMP architecture    = max_len $(cfg.max_len), num_eig $(cfg.nev), block_r $(cfg.block_r)")
    println(io, "  checkpoint          = ", cfg.checkpoint)
    for id in cfg.config_ids
        println(io, "  gauge cfg$(id)        = ", gauge_path(cfg, id))
        println(io, "  eigensystem cfg$(id)  = ", eigensystem_path(cfg, id))
    end
    println(io, "  table output        = ", summary_path(cfg))
    println(io, "  plot outputs        = ", plot_stem(cfg), ".png, .pdf")
    return String(take!(io))
end

function checksum_text(cfg::ExperimentConfig)
    io = IOBuffer()
    println(io, "  checkpoint sha256   = ", file_sha256(cfg.checkpoint))
    for id in cfg.config_ids
        println(io, "  gauge cfg$(id) sha256 = ", file_sha256(gauge_path(cfg, id)))
    end
    return String(take!(io))
end

function archive_settings!(cfg::ExperimentConfig, text::AbstractString)
    path = settings_path(cfg)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, text)
    end
    return path
end

field_norm(x) = sqrt(max(real(inner_product(x, x)), 0.0))

function normalize_field!(x)
    nrm = field_norm(x)
    nrm > 0 || error("cannot normalize a zero fermion field")
    axpby!(1 / nrm, x, 0, x)
    return nrm
end

copy_field!(y, x) = (axpby!(1, x, 0, y); y)

function orthogonalize!(w, basis, upto::Int)
    for _ in 1:2, i in 1:upto
        h = inner_product(basis[i], w)
        axpby!(-h, basis[i], 1, w)
    end
    return field_norm(w)
end

function orthonormalize_fields!(basis, upto::Int=length(basis))
    for j in 1:upto
        orthogonalize!(basis[j], basis, j - 1)
        normalize_field!(basis[j])
    end
    return basis
end

function combine!(out, basis, coeffs, n::Int)
    clear_fermion!(out)
    for i in 1:n
        axpby!(coeffs[i], basis[i], 1, out)
    end
    return out
end

function apply_dagdiracdirac!(out, D, tmp, x)
    mul!(tmp, D, x)
    mul!(out, D', tmp)
    return out
end

function eigenpair_stats!(r, Aq, D, tmp, q, lambda)
    apply_dagdiracdirac!(Aq, D, tmp, q)
    aq_norm = field_norm(Aq)
    copy_field!(r, Aq)
    axpby!(-lambda, q, 1, r)
    absolute = field_norm(r)
    relative = absolute / max(aq_norm, eps(Float64))
    return (; absolute, relative, aq_norm)
end

function load_raw_problem(cfg::ExperimentConfig, id::Int)
    U = Initialize_Gaugefields(3, 0, cfg.nx, cfg.ny, cfg.nz, cfg.nt;
                               condition="cold", cuda=true, blocks=cfg.pes)
    path = gauge_path(cfg, id)
    staging = LMP.Layer_sequence._make_gauge_staging(U)
    raw_cpu_buffers = LMP.Layer_sequence._make_raw_cpu_buffers(path, U)
    LMP.Layer_sequence._read_raw_gauge_to_gpu!(U, path, staging, raw_cpu_buffers)
    CUDA.synchronize()

    x = Initialize_pseudofermion_fields(U[1], "Wilson", nowing=true)
    D = Dirac_operator(U, x, Dict(
        "Dirac_operator" => "Wilson",
        "kappa" => cfg.kappa,
        "κ" => cfg.kappa,
        "faster version" => true,
        "verbose_level" => 0,
    ))
    return (; U, x, D)
end

function initialize_eigensolver(problem, cfg::ExperimentConfig)
    x = problem.x
    return (;
        V=[similar(x) for _ in 1:cfg.subspace_dim],
        trial=[similar(x) for _ in 1:cfg.keep_dim],
        best=[similar(x) for _ in 1:cfg.nev],
        w=similar(x), tmp=similar(x), residual=similar(x),
    )
end

function initialize_trials!(state, cfg::ExperimentConfig)
    for v in state.trial
        gauss_distribution_fermion!(v)
    end
    orthonormalize_fields!(state.trial)
    return state.trial
end

function build_krylov_basis!(state, problem, cfg::ExperimentConfig)
    for i in 1:cfg.keep_dim
        copy_field!(state.V[i], state.trial[i])
    end
    orthonormalize_fields!(state.V, cfg.keep_dim)
    actual_dim = cfg.keep_dim
    source = 1
    while actual_dim < cfg.subspace_dim
        apply_dagdiracdirac!(state.w, problem.D, state.tmp, state.V[source])
        beta = orthogonalize!(state.w, state.V, actual_dim)
        if beta < 1e-12
            gauss_distribution_fermion!(state.w)
            beta = orthogonalize!(state.w, state.V, actual_dim)
        end
        beta < 1e-12 && break
        actual_dim += 1
        copy_field!(state.V[actual_dim], state.w)
        axpby!(1 / beta, state.V[actual_dim], 0, state.V[actual_dim])
        source += 1
    end
    return actual_dim
end

function projected_matrix(state, problem, n::Int)
    H = zeros(ComplexF64, n, n)
    for j in 1:n
        apply_dagdiracdirac!(state.w, problem.D, state.tmp, state.V[j])
        for i in 1:n
            H[i, j] = inner_product(state.V[i], state.w)
        end
    end
    return Hermitian((H + H') / 2)
end

function rotate_trials!(state, coeffs, n::Int, cfg::ExperimentConfig)
    for k in 1:cfg.keep_dim
        combine!(state.trial[k], state.V, view(coeffs, :, k), n)
    end
    orthonormalize_fields!(state.trial)
    return state.trial
end

function compute_low_modes!(problem, cfg::ExperimentConfig)
    state = initialize_eigensolver(problem, cfg)
    initialize_trials!(state, cfg)
    values = zeros(Float64, cfg.nev)
    absolute_residuals = fill(Inf, cfg.nev)
    relative_residuals = fill(Inf, cfg.nev)
    best_values = similar(values)
    best_absolute = copy(absolute_residuals)
    best_relative = copy(relative_residuals)
    best_restart = 0
    best_dim = 0
    started = time()

    for restart in 1:cfg.max_restarts
        actual_dim = build_krylov_basis!(state, problem, cfg)
        F = eigen(projected_matrix(state, problem, actual_dim))
        order = sortperm(real.(F.values))
        rotate_trials!(state, F.vectors[:, order[1:cfg.keep_dim]], actual_dim, cfg)

        for k in 1:cfg.nev
            apply_dagdiracdirac!(state.w, problem.D, state.tmp, state.trial[k])
            values[k] = real(inner_product(state.trial[k], state.w))
            stats = eigenpair_stats!(state.residual, state.w, problem.D, state.tmp,
                                     state.trial[k], values[k])
            absolute_residuals[k] = stats.absolute
            relative_residuals[k] = stats.relative
        end
        maxrel = maximum(relative_residuals)
        println(@sprintf("restart %03d  basis=%d  max_relative=%.3e  elapsed=%.1fs  lambdas=%s",
                         restart, actual_dim, maxrel, time() - started,
                         join((@sprintf("%.10e", values[k]) for k in 1:cfg.nev), ", ")))
        flush(stdout)

        if maxrel < maximum(best_relative)
            copyto!(best_values, values)
            copyto!(best_absolute, absolute_residuals)
            copyto!(best_relative, relative_residuals)
            for k in 1:cfg.nev
                copy_field!(state.best[k], state.trial[k])
            end
            best_restart = restart
            best_dim = actual_dim
        end
        maxrel <= cfg.eig_tol && break
    end
    return (;
        vectors=state.best, values=best_values,
        absolute_residuals=best_absolute, relative_residuals=best_relative,
        best_restart, actual_dim=best_dim, seconds=time() - started,
        converged=maximum(best_relative) <= cfg.eig_tol,
    )
end

function fields_to_matrix(fields)
    cpu = LMP.Layer_sequence._to_cpu_fermions(fields)
    n = length(cpu[1].f)
    X = Matrix{ComplexF64}(undef, n, length(cpu))
    for j in eachindex(cpu)
        X[:, j] .= vec(cpu[j].f)
    end
    return X, cpu
end

function load_lmp_vectors(problem, cfg::ExperimentConfig)
    layers = build_lmp_32pt_layers(problem.U, problem.x, problem.D;
                                   max_len=cfg.max_len, num_eig=cfg.nev,
                                   block_r=cfg.block_r, init_gpu=false)
    opt = Adam()
    load_checkpoint!(cfg.checkpoint, layers, opt)
    refresh_model!(layers, problem.D, problem.U)
    vectors = layers[1:end-1]([problem.U])
    length(vectors) == cfg.nev || error(
        "LMP returned $(length(vectors)) vectors, expected $(cfg.nev)")
    return vectors
end

function save_eigensystem!(cfg, id, result, cpu_vectors, gauge_sha, checkpoint_sha)
    result.converged || error(@sprintf(
        "cfg%d did not reach relative tolerance %.3e; best maximum was %.3e",
        id, cfg.eig_tol, maximum(result.relative_residuals)))
    path = eigensystem_path(cfg, id)
    mkpath(dirname(path))
    metadata = Dict{String,Any}(
        "generated_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "config_id" => id, "gauge_path" => gauge_path(cfg, id),
        "gauge_sha256" => gauge_sha, "checkpoint" => cfg.checkpoint,
        "checkpoint_sha256" => checkpoint_sha, "operator" => "DdagD",
        "kappa" => cfg.kappa, "nev" => cfg.nev, "eig_tol" => cfg.eig_tol,
        "subspace_dim" => cfg.subspace_dim, "keep_dim" => cfg.keep_dim,
        "max_restarts" => cfg.max_restarts, "best_restart" => result.best_restart,
        "actual_dim" => result.actual_dim, "seconds" => result.seconds,
        "rng_seed" => cfg.rng_seed + id,
    )
    JLD2.save(path,
        "metadata", metadata,
        "eigenvectors", cpu_vectors,
        "eigenvalues", result.values,
        "absolute_residuals", result.absolute_residuals,
        "relative_residuals", result.relative_residuals)
    println("validated eigensystem saved: ", path)
    flush(stdout)
    return path
end

function compare_one!(cfg, id, checkpoint_sha)
    Random.seed!(cfg.rng_seed + id)
    CUDA.seed!(cfg.rng_seed + id)
    gauge_sha = file_sha256(gauge_path(cfg, id))
    println("\nconfiguration $id  gauge sha256=$gauge_sha")
    problem = load_raw_problem(cfg, id)
    result = compute_low_modes!(problem, cfg)
    result.converged || return (; id, result, exact_matrix=nothing, learned_matrix=nothing,
                                  similarity=nothing, mode_capture=nothing)

    exact_matrix, exact_cpu = fields_to_matrix(result.vectors)
    save_eigensystem!(cfg, id, result, exact_cpu, gauge_sha, checkpoint_sha)
    learned_fields = load_lmp_vectors(problem, cfg)
    learned_matrix, _ = fields_to_matrix(learned_fields)
    similarity = subspace_similarity(exact_matrix, learned_matrix)
    mode_capture = per_mode_capture(exact_matrix, learned_matrix)
    println(@sprintf("cfg%d  mean_capture=%.8f  worst_capture=%.8f  max_angle=%.4f deg",
                     id, similarity.mean_capture, similarity.worst_capture,
                     maximum(similarity.principal_angles_deg)))
    flush(stdout)
    return (; id, result, exact_matrix, learned_matrix, similarity, mode_capture)
end

function csv_escape(x)
    s = string(x)
    return occursin(r"[\",\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

function write_results!(cfg, comparisons, cross_exact)
    path = summary_path(cfg)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "comparison,mode,eigenvalue,absolute_residual,relative_residual,per_mode_capture,singular_value,principal_angle_deg,mean_capture,worst_capture")
        for cmp in comparisons
            cmp.similarity === nothing && continue
            for k in 1:cfg.nev
                row = ("lmp_vs_exact_cfg$(cmp.id)", k, cmp.result.values[k],
                       cmp.result.absolute_residuals[k], cmp.result.relative_residuals[k],
                       cmp.mode_capture[k], cmp.similarity.singular_values[k],
                       cmp.similarity.principal_angles_deg[k],
                       cmp.similarity.mean_capture, cmp.similarity.worst_capture)
                println(io, join(csv_escape.(row), ','))
            end
        end
        if cross_exact !== nothing
            for k in 1:length(cross_exact.singular_values)
                row = ("exact_cfg401_vs_cfg405", k, "", "", "", "",
                       cross_exact.singular_values[k], cross_exact.principal_angles_deg[k],
                       cross_exact.mean_capture, cross_exact.worst_capture)
                println(io, join(csv_escape.(row), ','))
            end
        end
    end

    valid = [cmp for cmp in comparisons if cmp.similarity !== nothing]
    isempty(valid) && return path
    labels = ["cfg$(cmp.id) mode $k" for cmp in valid for k in 1:cfg.nev]
    captures = [cmp.mode_capture[k] for cmp in valid for k in 1:cfg.nev]
    angles = [cmp.similarity.principal_angles_deg[k] for cmp in valid for k in 1:cfg.nev]
    p1 = bar(labels, captures; ylim=(0, 1), ylabel="exact-mode capture",
             title="LMP capture of six lowest D' D modes at kappa=0.157",
             legend=false, xrotation=45, size=(1200, 650), bottom_margin=12Plots.mm)
    p2 = bar(labels, cosd.(angles); ylim=(0, 1), ylabel="cos(principal angle)",
             title="Principal-angle similarity", legend=false, xrotation=45,
             size=(1200, 650), bottom_margin=12Plots.mm)
    p = plot(p1, p2; layout=(2, 1), size=(1200, 1200))
    mkpath(dirname(plot_stem(cfg)))
    savefig(p, plot_stem(cfg) * ".png")
    savefig(p, plot_stem(cfg) * ".pdf")
    return path
end

function parse_config(args)
    cfg = default_config()
    for arg in args
        if arg == "--smoke"
            cfg.smoke = true
        elseif arg == "--print-settings-only"
            cfg.print_settings_only = true
        elseif startswith(arg, "--config-ids=")
            cfg.config_ids = parse.(Int, split(split(arg, "="; limit=2)[2], ','))
        elseif startswith(arg, "--nev=")
            cfg.nev = parse(Int, split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--subspace-dim=")
            cfg.subspace_dim = parse(Int, split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--keep-dim=")
            cfg.keep_dim = parse(Int, split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--max-restarts=")
            cfg.max_restarts = parse(Int, split(arg, "="; limit=2)[2])
        else
            throw(ArgumentError("unknown argument: $arg"))
        end
    end
    cfg.cuda_visible_devices = get(ENV, "CUDA_VISIBLE_DEVICES", "1")
    return cfg
end

function run_experiment(cfg::ExperimentConfig)
    CUDA.functional() || error("CUDA is not functional")
    checkpoint_sha = file_sha256(cfg.checkpoint)
    comparisons = Any[]
    for id in cfg.config_ids
        push!(comparisons, compare_one!(cfg, id, checkpoint_sha))
    end
    cross_exact = if length(comparisons) == 2 && all(c -> c.exact_matrix !== nothing, comparisons)
        subspace_similarity(comparisons[1].exact_matrix, comparisons[2].exact_matrix)
    else
        nothing
    end
    if cfg.smoke
        println("smoke run complete; no scientific result files were written")
        return comparisons
    end
    path = write_results!(cfg, comparisons, cross_exact)
    println("summary written: ", path)
    return comparisons
end

function main(args=ARGS)
    cfg = parse_config(args)
    validate_config(cfg)
    report = settings_text(cfg) * checksum_text(cfg)
    print(report)
    archive_settings!(cfg, report)
    cfg.print_settings_only && return nothing
    return Base.invokelatest(run_experiment, cfg)
end

if !isempty(PROGRAM_FILE) && samefile(PROGRAM_FILE, @__FILE__)
    main()
end
