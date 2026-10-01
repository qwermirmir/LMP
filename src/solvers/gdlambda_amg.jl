using LinearAlgebra

struct ShiftedGamma5DiracOperator{TD}
    D::TD
    shift::Float64
end

ShiftedGamma5DiracOperator(D, shift::Real) =
    ShiftedGamma5DiracOperator{typeof(D)}(D, Float64(shift))

Base.adjoint(A::ShiftedGamma5DiracOperator) = A

function LinearAlgebra.mul!(out, A::ShiftedGamma5DiracOperator, x)
    mul!(out, A.D, x)
    DDalphaAMG.gamma5!(out)
    axpby!(-A.shift, x, 1, out)
    return out
end

struct Gamma5LeftShiftedDiracOperator{TD,TF}
    D::TD
    shift::Float64
    gamma5_work::TF
end


Gamma5LeftShiftedDiracOperator(D, shift::Real, template) =
    Gamma5LeftShiftedDiracOperator{typeof(D),typeof(template)}(
        D, Float64(shift), similar(template))

function LinearAlgebra.mul!(out, A::Gamma5LeftShiftedDiracOperator, x)
    mul!(out, A.D, x)
    _gd_copy!(A.gamma5_work, x)
    DDalphaAMG.gamma5!(A.gamma5_work)
    axpby!(-A.shift, A.gamma5_work, 1, out)
    return out
end

"""
    shifted_chiral_coarse_stencil(op, shift)

Form `Dc - shift * Gamma5c` for a chirally split coarse basis ordered as
`(plus, minus, plus, minus, ...)`, as required by equation (3.7).
"""
function shifted_chiral_coarse_stencil(op::DDalphaAMG.CoarseStencilOperator,
                                       shift::Real)
    iseven(op.Nv) || throw(ArgumentError(
        "chiral coarse dimension must be even, got Nv=$(op.Nv)"))
    shifted_self = copy(op.self)
    for component in 1:op.Nv
        gamma5c = isodd(component) ? 1.0 : -1.0
        @views shifted_self[component, component, :] .-= shift * gamma5c
    end
    return DDalphaAMG.CoarseStencilOperator(
        op.Nv, op.Nblocks, op.NBX, op.NBY, op.NBZ, op.NBT,
        shifted_self, op.plus, op.minus, op.nb_plus, op.nb_minus)
end

"""Apply the shifted Galerkin coarse correction to a fine residual."""
function shifted_coarse_correction!(out, residual,
                                    operator,
                                    setup::DDalphaAMG_Setup;
                                    coarse_tol::Float64,
                                    coarse_maxiter::Int,
                                    coarse_restart::Int,
                                    coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing)
    coarse_rhs = DDalphaAMG._coarse_vector(setup)
    restrict_f2c!(coarse_rhs, setup, residual)
    setup.coarse_stencil isa DDalphaAMG.CoarseStencilOperator ||
        throw(ArgumentError(
            "paper-style shifted correction requires a precomputed coarse stencil"))
    shifted_stencil = shifted_chiral_coarse_stencil(
        setup.coarse_stencil, operator.shift)
    original_stencil = setup.coarse_stencil
    coarse_solution = try
        setup.coarse_stencil = shifted_stencil
        gd_coarse_gmres(
            setup, operator, coarse_rhs;
            tol=coarse_tol, maxiter=coarse_maxiter, restart=coarse_restart,
            coarse_odd_even=true, arnoldi_mode=:scalar,
            coarse_apply_counter=coarse_apply_counter)
    finally
        setup.coarse_stencil = original_stencil
    end
    prolongate_c2f!(out, setup, coarse_solution)
    coarse_image = similar(coarse_rhs)
    DDalphaAMG.apply_coarse_stencil!(
        coarse_image, shifted_stencil, coarse_solution)
    coarse_apply_counter !== nothing && (coarse_apply_counter[] += 1)
    coarse_residual = similar(coarse_rhs)
    coarse_residual.data .= coarse_rhs.data
    DDalphaAMG.coarse_axpby!(-1.0, coarse_image, 1.0, coarse_residual)
    denominator = max(DDalphaAMG.coarse_norm(coarse_rhs), eps(Float64))
    verified_relative_residual =
        DDalphaAMG.coarse_norm(coarse_residual) / denominator
    schur_relative_residual =
        gd_coarse_odd_even_relative_residual(
            shifted_stencil, coarse_solution, coarse_rhs)
    return (; output=out,
            coarse_schur_relative_residual=schur_relative_residual,
            coarse_verified_relative_residual=verified_relative_residual,
            coarse_original_relative_residual=verified_relative_residual)
end

# Small vector interface shared by dense unit tests and Wilson fermion fields.
_gd_copy!(out::AbstractArray, x::AbstractArray) = (copyto!(out, x); out)
_gd_copy!(out, x) = (axpby!(1, x, 0, out); out)
_gd_zero!(out::AbstractArray) = (fill!(out, zero(eltype(out))); out)
_gd_zero!(out) = (zero_wf!(out); out)
_gd_axpy!(a, x::AbstractArray, y::AbstractArray) = (axpy!(a, x, y); y)
_gd_axpy!(a, x, y) = (axpby!(a, x, 1, y); y)
_gd_scale!(a, x::AbstractArray) = (lmul!(a, x); x)
_gd_scale!(a, x) = (axpby!(a, x, 0, x); x)
_gd_dot(x::AbstractArray, y::AbstractArray) = dot(x, y)
_gd_dot(x, y) = inner_product(x, y)
_gd_norm(x::AbstractArray) = norm(x)
_gd_norm(x) = _field_norm(x)

function _gd_clone(x)
    out = similar(x)
    return _gd_copy!(out, x)
end

function _gd_normalize!(x; drop_tol::Float64=1e-14)
    nrm = _gd_norm(x)
    isfinite(nrm) && nrm > drop_tol ||
        throw(ArgumentError("Cannot normalize vector with norm $nrm"))
    _gd_scale!(inv(nrm), x)
    return nrm
end

function _gd_orthogonalize_twice!(x, basis; drop_tol::Float64=1e-14)
    for _ in 1:2, q in basis
        _gd_axpy!(-_gd_dot(q, x), q, x)
    end
    nrm = _gd_norm(x)
    return isfinite(nrm) && nrm > drop_tol ? nrm : 0.0
end

function _gd_orthonormal_copy(block; drop_tol::Float64=1e-14)
    basis = Any[]
    for (column, input) in enumerate(block)
        vector = _gd_clone(input)
        nrm = _gd_orthogonalize_twice!(vector, basis; drop_tol)
        nrm > drop_tol ||
            throw(ArgumentError("Rank-deficient initial block at column $column"))
        _gd_scale!(inv(nrm), vector)
        push!(basis, vector)
    end
    return basis
end

function _gd_restart_basis(candidates, keep::Int; drop_tol::Float64=1e-14)
    1 <= keep <= length(candidates) ||
        throw(ArgumentError("Invalid thick-restart retain count $keep"))
    return _gd_orthonormal_copy(
        Any[candidates[i].vector for i in 1:keep]; drop_tol)
end

function _gd_restart_basis_images(candidates, keep::Int;
                                  locked_vectors=Any[], locked_images=Any[],
                                  drop_tol::Float64=1e-14)
    1 <= keep <= length(candidates) ||
        throw(ArgumentError("Invalid thick-restart retain count $keep"))
    length(locked_vectors) == length(locked_images) ||
        throw(DimensionMismatch("locked vectors and images differ"))
    basis, images = Any[], Any[]
    for candidate in candidates
        vector = _gd_clone(candidate.vector)
        image = _gd_clone(candidate.Avector)
        for _ in 1:2, i in eachindex(locked_vectors)
            coefficient = _gd_dot(locked_vectors[i], vector)
            _gd_axpy!(-coefficient, locked_vectors[i], vector)
            _gd_axpy!(-coefficient, locked_images[i], image)
        end
        for _ in 1:2, i in eachindex(basis)
            coefficient = _gd_dot(basis[i], vector)
            _gd_axpy!(-coefficient, basis[i], vector)
            _gd_axpy!(-coefficient, images[i], image)
        end
        nrm = _gd_norm(vector)
        nrm > drop_tol || continue
        _gd_scale!(inv(nrm), vector)
        _gd_scale!(inv(nrm), image)
        push!(basis, vector); push!(images, image)
        length(basis) == keep && break
    end
    length(basis) == keep || throw(ArgumentError(
        "Insufficient independent thick-restart vectors after locking"))
    return basis, images
end

function _gd_deflate_basis_images(basis, images, locked_vectors, locked_images;
                                  drop_tol::Float64=1e-14)
    length(basis) == length(images) ||
        throw(DimensionMismatch("basis and images differ"))
    length(locked_vectors) == length(locked_images) ||
        throw(DimensionMismatch("locked vectors and images differ"))
    deflated_basis, deflated_images = Any[], Any[]
    for column in eachindex(basis)
        vector = _gd_clone(basis[column])
        image = _gd_clone(images[column])
        for _ in 1:2, i in eachindex(locked_vectors)
            coefficient = _gd_dot(locked_vectors[i], vector)
            _gd_axpy!(-coefficient, locked_vectors[i], vector)
            _gd_axpy!(-coefficient, locked_images[i], image)
        end
        for _ in 1:2, i in eachindex(deflated_basis)
            coefficient = _gd_dot(deflated_basis[i], vector)
            _gd_axpy!(-coefficient, deflated_basis[i], vector)
            _gd_axpy!(-coefficient, deflated_images[i], image)
        end
        nrm = _gd_norm(vector)
        nrm > drop_tol || continue
        _gd_scale!(inv(nrm), vector)
        _gd_scale!(inv(nrm), image)
        push!(deflated_basis, vector); push!(deflated_images, image)
    end
    return deflated_basis, deflated_images
end

function _gd_combine(basis, coefficients)
    length(basis) == length(coefficients) ||
        throw(DimensionMismatch("basis and coefficient lengths differ"))
    out = similar(first(basis))
    _gd_zero!(out)
    for i in eachindex(basis)
        _gd_axpy!(coefficients[i], basis[i], out)
    end
    return out
end

function _gd_remove_ritz_direction(basis, images, coefficients)
    length(basis) == length(images) == length(coefficients) ||
        throw(DimensionMismatch("basis, images, and coefficients differ"))
    length(basis) >= 2 || return Any[], Any[]
    complement = nullspace(adjoint(collect(coefficients)))
    size(complement, 2) == length(basis) - 1 || error(
        "Failed to construct the locked Ritz-vector complement")
    reduced_basis = Any[]
    reduced_images = Any[]
    for column in axes(complement, 2)
        push!(reduced_basis, _gd_combine(basis, view(complement, :, column)))
        push!(reduced_images, _gd_combine(images, view(complement, :, column)))
    end
    return reduced_basis, reduced_images
end

function _gd_direct_ritz_candidate(apply_Q!, candidate)
    direct_image = similar(candidate.vector)
    apply_Q!(direct_image, candidate.vector)
    difference = _gd_clone(direct_image)
    _gd_axpy!(-1, candidate.Avector, difference)
    action_mismatch = _gd_norm(difference) /
                      max(_gd_norm(direct_image), eps(Float64))
    residual_vector = _gd_clone(direct_image)
    _gd_axpy!(-candidate.value, candidate.vector, residual_vector)
    return merge(candidate, (; Avector=direct_image,
                              residual=_gd_norm(residual_vector),
                              action_mismatch))
end

function _gd_refresh_basis_images!(images, apply_Q!, basis)
    length(images) == length(basis) ||
        throw(DimensionMismatch("basis and images differ"))
    for i in eachindex(basis)
        apply_Q!(images[i], basis[i])
    end
    return length(basis)
end

"""
    flexible_gmres!(x, apply_A!, b, apply_M!; tol, maxiter, restart)

Restarted right-preconditioned FGMRES. `apply_A!(out, input)` and
`apply_M!(out, input, iteration)` write into caller-owned vectors.
"""
function flexible_gmres!(x, apply_A!, b, apply_M!;
                          tol::Float64, maxiter::Int, restart::Int,
                          history_callback=(args...)->nothing,
                          drop_tol::Float64=1e-14)
    tol > 0 || throw(ArgumentError("tol must be positive"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    restart > 0 || throw(ArgumentError("restart must be positive"))
    bnorm = max(_gd_norm(b), eps(Float64))
    Ax = similar(b)
    residual = similar(b)
    applications = 0
    iterations = 0
    history = Float64[]

    function direct_residual!()
        apply_A!(Ax, x)
        applications += 1
        _gd_copy!(residual, b)
        _gd_axpy!(-1, Ax, residual)
        return _gd_norm(residual) / bnorm
    end

    relative_residual = direct_residual!()
    push!(history, relative_residual)
    history_callback(iterations, applications, relative_residual)

    while relative_residual > tol && iterations < maxiter
        beta = _gd_norm(residual)
        v1 = _gd_clone(residual)
        _gd_scale!(inv(beta), v1)
        cycle_steps = min(restart, maxiter - iterations)
        V = Any[v1]
        Z = Any[]
        H = zeros(ComplexF64, cycle_steps + 1, cycle_steps)
        used = 0

        for j in 1:cycle_steps
            z = similar(b)
            apply_M!(z, V[j], iterations + j)
            push!(Z, z)
            w = similar(b)
            apply_A!(w, z)
            applications += 1
            for _ in 1:2, i in 1:j
                coefficient = _gd_dot(V[i], w)
                H[i, j] += coefficient
                _gd_axpy!(-coefficient, V[i], w)
            end
            H[j + 1, j] = _gd_norm(w)
            used = j
            if abs(H[j + 1, j]) > drop_tol
                _gd_scale!(inv(real(H[j + 1, j])), w)
                push!(V, w)
            end

            rhs = zeros(ComplexF64, j + 1)
            rhs[1] = beta
            y = H[1:(j + 1), 1:j] \ rhs
            estimate = norm(rhs - H[1:(j + 1), 1:j] * y) / bnorm
            if estimate <= tol || abs(H[j + 1, j]) <= drop_tol
                break
            end
        end

        rhs = zeros(ComplexF64, used + 1)
        rhs[1] = beta
        y = H[1:(used + 1), 1:used] \ rhs
        for i in 1:used
            _gd_axpy!(y[i], Z[i], x)
        end
        iterations += used
        relative_residual = direct_residual!()
        push!(history, relative_residual)
        history_callback(iterations, applications, relative_residual)
    end

    return (; converged=relative_residual <= tol, iterations, applications,
            verified_relative_residual=relative_residual, history)
end

"""
    shifted_amg_vcycle!(out, rhs, apply_A!, coarse_correction!;
                        post_smooth_steps=4)

Apply the paper-default shifted two-level DD-alphaAMG preconditioner: one
shifted Galerkin coarse correction followed by four fine-grid GMRES
post-smoothing steps. Residuals are recomputed with `apply_A!` and reported
relative to the original right-hand side.
"""
function shifted_amg_vcycle!(out, rhs, apply_A!, coarse_correction!;
                             post_smooth_steps::Int=4,
                             drop_tol::Float64=1e-14)
    post_smooth_steps == 4 || throw(ArgumentError(
        "Paper-default shifted AMG requires exactly 4 post-smoothing steps"))
    drop_tol > 0 || throw(ArgumentError("drop_tol must be positive"))

    rhs_norm = _gd_norm(rhs)
    isfinite(rhs_norm) || throw(ArgumentError("Non-finite shifted AMG RHS norm"))
    denominator = max(rhs_norm, eps(Float64))
    _gd_zero!(out)

    coarse_diagnostics = coarse_correction!(out, rhs)
    coarse_applications = coarse_diagnostics isa NamedTuple ?
        get(coarse_diagnostics, :coarse_applications, 0) : 0
    coarse_verified_relative_residual = coarse_diagnostics isa NamedTuple ?
        get(coarse_diagnostics, :coarse_verified_relative_residual, NaN) : NaN

    image = similar(rhs)
    residual = similar(rhs)
    fine_applications = 0
    apply_A!(image, out)
    fine_applications += 1
    _gd_copy!(residual, rhs)
    _gd_axpy!(-1, image, residual)
    coarse_relative_residual = _gd_norm(residual) / denominator
    isfinite(coarse_relative_residual) || throw(ArgumentError(
        "Non-finite residual after shifted coarse correction"))

    delta = similar(rhs)
    _gd_zero!(delta)
    counted_apply_A! = function(destination, input)
        apply_A!(destination, input)
        fine_applications += 1
        return destination
    end
    identity_preconditioner! = function(destination, input, iteration)
        _gd_copy!(destination, input)
        return destination
    end
    post = flexible_gmres!(
        delta, counted_apply_A!, residual, identity_preconditioner!;
        tol=eps(Float64), maxiter=post_smooth_steps,
        restart=post_smooth_steps, drop_tol,
    )
    _gd_axpy!(1, delta, out)

    apply_A!(image, out)
    fine_applications += 1
    _gd_copy!(residual, rhs)
    _gd_axpy!(-1, image, residual)
    final_relative_residual = _gd_norm(residual) / denominator
    isfinite(final_relative_residual) || throw(ArgumentError(
        "Non-finite residual after shifted GMRES post-smoothing"))

    return (; initial_relative_residual=(rhs_norm / denominator),
            coarse_relative_residual, final_relative_residual,
            coarse_applications, coarse_verified_relative_residual,
            post_iterations=post.iterations,
            post_applications=post.applications, fine_applications)
end

function _gd_projected_gram(left, right)
    M = Matrix{ComplexF64}(undef, length(left), length(right))
    for j in eachindex(right), i in eachindex(left)
        M[i, j] = _gd_dot(left[i], right[j])
    end
    return M
end

function _gd_orthogonality_error(vectors)
    isempty(vectors) && return 0.0
    gram = _gd_projected_gram(vectors, vectors)
    return opnorm(gram - Matrix{ComplexF64}(I, size(gram)...))
end

function _gd_cross_orthogonality_error(left, right)
    return isempty(left) || isempty(right) ? 0.0 :
        opnorm(_gd_projected_gram(left, right))
end

function _gd_interpolation_subset(locked_vectors, locked_values,
                                  target::Real, ntv::Int)
    length(locked_vectors) == length(locked_values) ||
        throw(DimensionMismatch("locked vectors and values differ"))
    length(locked_values) >= ntv || throw(ArgumentError(
        "Need at least $ntv locked eigenvectors to rebuild interpolation"))
    indices = sortperm(eachindex(locked_values);
        by=i -> (abs(locked_values[i] - target), abs(locked_values[i]), i))[1:ntv]
    return Any[locked_vectors[i] for i in indices], locked_values[indices]
end

function _gd_candidate_is_locked(candidate, locked_vectors;
                                 overlap_tol::Float64=1e-8)
    isempty(locked_vectors) && return false
    projection_norm2 = sum(abs2(_gd_dot(vector, candidate.vector))
                           for vector in locked_vectors)
    return projection_norm2 >= 1.0 - overlap_tol
end

"""
    gd_eigenpair_audit(apply_Q!, vectors, values)

Independently reapply `Q` to every reported eigenvector and return absolute
`Q` and `Q^2` residuals, the vector-level identity error
`Q(Qv-θv)+θ(Qv-θv) = Q^2v-θ^2v`, and the Gram-matrix
orthogonality error. No operator images produced by the eigensolver are reused.
"""
function gd_eigenpair_audit(apply_Q!, vectors, values)
    length(vectors) == length(values) || throw(DimensionMismatch(
        "vectors and values must have the same length"))
    pairs = NamedTuple[]
    for (vector, value) in zip(vectors, values)
        Qvector = similar(vector)
        apply_Q!(Qvector, vector)

        residual = _gd_clone(Qvector)
        _gd_axpy!(-value, vector, residual)

        Qresidual = similar(vector)
        apply_Q!(Qresidual, residual)
        identity_rhs = _gd_clone(Qresidual)
        _gd_axpy!(value, residual, identity_rhs)

        Q2vector = similar(vector)
        apply_Q!(Q2vector, Qvector)
        normal_residual = _gd_clone(Q2vector)
        _gd_axpy!(-(value^2), vector, normal_residual)

        identity_difference = _gd_clone(normal_residual)
        _gd_axpy!(-1, identity_rhs, identity_difference)
        identity_scale = max(_gd_norm(normal_residual),
                             _gd_norm(identity_rhs), eps(Float64))
        push!(pairs, (; value=Float64(real(value)),
                      vector_norm=_gd_norm(vector),
                      q_residual=_gd_norm(residual),
                      q2_residual=_gd_norm(normal_residual),
                      q2_identity_error=
                          _gd_norm(identity_difference) / identity_scale))
    end
    return (; pairs,
            orthogonality_error=_gd_orthogonality_error(vectors))
end

"""Return the two projected matrices in equation (3.5) of the paper."""
function harmonic_ritz_pencil(basis, Qbasis)
    length(basis) == length(Qbasis) ||
        throw(DimensionMismatch("basis image count differs"))
    return _gd_projected_gram(Qbasis, Qbasis),
           _gd_projected_gram(Qbasis, basis)
end

function _gd_hermitian_harmonic_eigen(C, B;
                                      rank_tol::Float64=1e-13)
    C_h = Hermitian((C + C') / 2)
    B_h = Hermitian((B + B') / 2)
    metric = eigen(C_h)
    scale = maximum(abs, metric.values; init=0.0)
    keep = findall(value -> value > rank_tol * max(scale, 1.0),
                   metric.values)
    isempty(keep) && error("Harmonic Ritz metric has zero numerical rank")
    whitening = metric.vectors[:, keep] *
                Diagonal(inv.(sqrt.(metric.values[keep])))
    inverse_pencil = Hermitian(whitening' * B_h * whitening)
    inverse_decomposition = eigen(inverse_pencil)
    valid = findall(value -> isfinite(value) &&
                             abs(value) > rank_tol,
                    inverse_decomposition.values)
    isempty(valid) && error("No finite real harmonic Ritz candidates")
    values = inv.(inverse_decomposition.values[valid])
    vectors = whitening * inverse_decomposition.vectors[:, valid]
    return (; values, vectors)
end

"""
    harmonic_ritz(basis, Abasis; target=0.0)

Extract harmonic Ritz candidates nearest a real target from an orthonormal
search basis and its operator images.
"""
function harmonic_ritz(basis, Abasis; target::Float64=0.0,
                       rank_tol::Float64=1e-13)
    target == 0.0 || throw(ArgumentError(
        "Algorithm 2 harmonic extraction targets zero; got target=$target"))
    n = length(basis)
    n > 0 || throw(ArgumentError("harmonic Ritz basis is empty"))
    length(Abasis) == n || throw(DimensionMismatch("basis image count differs"))
    C, B = harmonic_ritz_pencil(basis, Abasis)
    decomposition = _gd_hermitian_harmonic_eigen(C, B; rank_tol)
    values = decomposition.values
    order = sort(eachindex(values); by=i -> abs(values[i] - target))

    candidates = NamedTuple[]
    for index in order
        coefficients = decomposition.vectors[:, index]
        vector = _gd_combine(basis, coefficients)
        _gd_normalize!(vector)
        Avector = similar(vector)
        _gd_zero!(Avector)
        for i in 1:n
            _gd_axpy!(coefficients[i], Abasis[i], Avector)
        end
        # Coefficients and vector share the same pre-normalization scale.
        scale = inv(_gd_norm(_gd_combine(basis, coefficients)))
        _gd_scale!(scale, Avector)
        value = values[index]
        residual_vector = _gd_clone(Avector)
        _gd_axpy!(-value, vector, residual_vector)
        push!(candidates, (; value, vector, Avector,
                           residual=_gd_norm(residual_vector), coefficients))
    end
    return merge(first(candidates), (; candidates))
end

"""
    generalized_davidson(apply_Q!, initial_block; correction_solver!, ...)

Single-pair harmonic generalized Davidson targeting the eigenvalue closest to
`target`. The correction callback writes into `out` and may return diagnostics.
"""
function generalized_davidson(apply_Q!, initial_block;
                              correction_solver!, tol::Float64,
                              max_expansions::Int, max_subspace::Int,
                              retain::Int=1, target::Float64=0.0,
                              nev::Int=1, ntv::Int=length(initial_block),
                              hierarchy_update! = (vectors, values, target)->nothing,
                              initial_direction=nothing,
                              replacement_direction! = nothing,
                              max_replacement_attempts::Int=8,
                              history_callback=(args...)->nothing,
                              diagnostic_callback=nothing,
                              normal_residual! = nothing,
                              drop_tol::Float64=1e-14)
    1 <= retain < max_subspace ||
        throw(ArgumentError("Require 1 <= retain < max_subspace"))
    nev >= 1 || throw(ArgumentError("nev must be positive"))
    ntv >= 1 || throw(ArgumentError("ntv must be positive"))
    max_replacement_attempts >= 1 || throw(ArgumentError(
        "max_replacement_attempts must be positive"))
    basis = _gd_orthonormal_copy(initial_block; drop_tol)
    if initial_direction !== nothing
        direction = _gd_clone(initial_direction)
        direction_norm = _gd_orthogonalize_twice!(
            direction, basis; drop_tol)
        direction_norm > drop_tol || throw(ArgumentError(
            "Initial direction is contained in the initial search space"))
        _gd_scale!(inv(direction_norm), direction)
        push!(basis, direction)
    end
    length(basis) <= max_subspace ||
        throw(ArgumentError("Initial block exceeds max_subspace"))
    Qbasis = Any[]
    q_applications = 0
    for vector in basis
        image = similar(vector)
        apply_Q!(image, vector)
        q_applications += 1
        push!(Qbasis, image)
    end
    expansions = 0
    inner_iterations = 0
    inner_applications = 0
    restarts = 0
    history = NamedTuple[]
    final = nothing
    locked_values = Float64[]
    locked_vectors = Any[]
    locked_images = Any[]

    is_locked = candidate ->
        _gd_candidate_is_locked(candidate, locked_vectors)

    function emit_diagnostic(stage, extraction)
        diagnostic_callback === nothing && return nothing
        candidates = [
            (; rank, value=Float64(real(candidate.value)),
             residual=Float64(candidate.residual),
             coefficient_norm=Float64(norm(candidate.coefficients)),
             locked_overlap=Float64(isempty(locked_vectors) ? 0.0 : sqrt(sum(
                 abs2(_gd_dot(vector, candidate.vector))
                 for vector in locked_vectors))))
            for (rank, candidate) in enumerate(extraction.candidates)
        ]
        diagnostic_callback((; stage, expansion=expansions, restarts,
                             locked_values=copy(locked_values), candidates))
        return nothing
    end

    while true
        extraction = harmonic_ritz(basis, Qbasis; target)
        emit_diagnostic(:extraction, extraction)
        available = filter(candidate -> !is_locked(candidate),
                           extraction.candidates)
        isempty(available) && error("No unlocked harmonic Ritz candidate")
        local ritz, normal_residual
        while true
            ritz = _gd_direct_ritz_candidate(apply_Q!, first(available))
            q_applications += 1
            final = ritz
            normal_residual = normal_residual! === nothing ? NaN :
                normal_residual!(ritz.vector, ritz.value)
            record = (; expansion=expansions, subspace=length(basis),
                      value=ritz.value, residual=ritz.residual,
                      normal_residual, q_applications, inner_iterations,
                      inner_applications, restarts,
                      ritz_action_mismatch=ritz.action_mismatch,
                      locked_count=length(locked_vectors),
                      basis_orthogonality=_gd_orthogonality_error(basis),
                      locked_orthogonality=_gd_orthogonality_error(locked_vectors),
                      basis_locked_overlap=_gd_cross_orthogonality_error(
                          basis, locked_vectors),
                      correction_basis_overlap=NaN,
                      correction_locked_overlap=NaN)
            push!(history, record)
            history_callback(record)
            ritz.residual <= tol || break

            push!(locked_values, ritz.value)
            push!(locked_vectors, _gd_clone(ritz.vector))
            push!(locked_images, _gd_clone(ritz.Avector))
            basis, Qbasis = _gd_remove_ritz_direction(
                basis, Qbasis, ritz.coefficients)
            if length(locked_values) == nev
                return (; converged=true, capped=false, rank_lost=false,
                        value=ritz.value, vector=ritz.vector,
                        values=locked_values, vectors=locked_vectors,
                        residual=ritz.residual, normal_residual,
                        expansions, q_applications, inner_iterations,
                        inner_applications, restarts, history)
            end

            extraction = harmonic_ritz(basis, Qbasis; target)
            emit_diagnostic(:post_lock, extraction)
            available = filter(candidate -> !is_locked(candidate),
                               extraction.candidates)
            isempty(available) && error(
                "Search space contains no candidate after locking")
            if length(locked_values) >= ntv
                interpolation_vectors, interpolation_values =
                    _gd_interpolation_subset(
                        locked_vectors, locked_values,
                        first(available).value, ntv)
                hierarchy_update!(interpolation_vectors,
                                  interpolation_values,
                                  first(available).value)
            end
        end
        expansions >= max_expansions && break

        residual_vector = _gd_clone(ritz.Avector)
        _gd_axpy!(-ritz.value, ritz.vector, residual_vector)
        correction = similar(ritz.vector)
        diagnostics = correction_solver!(correction, residual_vector, ritz.value)
        if diagnostics isa NamedTuple
            inner_iterations += get(diagnostics, :iterations, 0)
            inner_applications += get(diagnostics, :applications, 0)
        end

        # Algorithm 2 line 14 computes the correction from the current search
        # space.  Lines 15--21 perform the thick restart only afterwards.
        if length(basis) >= max_subspace
            restart_candidates = filter(candidate -> !is_locked(candidate),
                                        extraction.candidates)
            emit_diagnostic(:pre_restart, extraction)
            keep = min(retain, length(restart_candidates))
            basis, Qbasis = _gd_restart_basis_images(
                restart_candidates, keep;
                locked_vectors, locked_images, drop_tol)
            restarts += 1
            diagnostic_callback === nothing || emit_diagnostic(
                :post_restart, harmonic_ritz(basis, Qbasis; target))
        end

        correction_norm = _gd_orthogonalize_twice!(correction, basis; drop_tol)
        if correction_norm > drop_tol && !isempty(locked_vectors)
            correction_norm = _gd_orthogonalize_twice!(
                correction, locked_vectors; drop_tol)
        end
        if correction_norm > drop_tol
            unit_scale = inv(correction_norm)
            basis_overlap = isempty(basis) ? 0.0 : maximum(
                abs(_gd_dot(vector, correction)) * unit_scale for vector in basis)
            locked_overlap = isempty(locked_vectors) ? 0.0 : maximum(
                abs(_gd_dot(vector, correction)) * unit_scale
                for vector in locked_vectors)
            history[end] = merge(history[end], (;
                correction_basis_overlap=basis_overlap,
                correction_locked_overlap=locked_overlap))
        end
        if correction_norm <= drop_tol
            for attempt in 1:max_replacement_attempts
                replacement_direction! === nothing && break
                replacement_direction!(correction, attempt)
                correction_norm = _gd_orthogonalize_twice!(
                    correction, basis; drop_tol)
                if correction_norm > drop_tol && !isempty(locked_vectors)
                    correction_norm = _gd_orthogonalize_twice!(
                        correction, locked_vectors; drop_tol)
                end
                correction_norm > drop_tol && break
            end
        end
        if correction_norm <= drop_tol
            normal_residual = normal_residual! === nothing ? NaN :
                normal_residual!(ritz.vector, ritz.value)
            return (; converged=false, capped=false, rank_lost=true,
                    value=ritz.value, vector=ritz.vector,
                    values=locked_values, vectors=locked_vectors,
                    residual=ritz.residual, normal_residual,
                    expansions, q_applications, inner_iterations,
                    inner_applications, restarts, history)
        end
        _gd_scale!(inv(correction_norm), correction)
        push!(basis, correction)
        Qcorrection = similar(correction)
        apply_Q!(Qcorrection, correction)
        q_applications += 1
        push!(Qbasis, Qcorrection)
        expansions += 1
    end

    normal_residual = normal_residual! === nothing ? NaN :
        normal_residual!(final.vector, final.value)
    return (; converged=false, capped=true, rank_lost=false,
            value=final.value, vector=final.vector,
            values=locked_values, vectors=locked_vectors,
            residual=final.residual, normal_residual,
            expansions, q_applications, inner_iterations,
            inner_applications, restarts, history)
end
