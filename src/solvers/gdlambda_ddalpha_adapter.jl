using LinearAlgebra

"""Globally orthonormalize GD-owned AMG prototypes in place."""
function gd_global_orthonormalize!(vectors; drop_tol::Float64=1e-14)
    for j in eachindex(vectors)
        for _ in 1:2, i in 1:(j - 1)
            axpby!(-inner_product(vectors[i], vectors[j]), vectors[i], 1,
                   vectors[j])
        end
        norm_j = sqrt(max(0.0, real(inner_product(vectors[j], vectors[j]))))
        isfinite(norm_j) && norm_j > drop_tol || throw(ArgumentError(
            "Rank-deficient GD adaptive setup prototypes at column $j"))
        axpby!(inv(norm_j), vectors[j], 0, vectors[j])
    end
    return vectors
end

"""Return the true relative residual of a GD shifted odd-even Schur system."""
function gd_coarse_odd_even_relative_residual(
        op::DDalphaAMG.CoarseStencilOperator,
        solution::DDalphaAMG.CoarseVector,
        rhs::DDalphaAMG.CoarseVector)
    DDalphaAMG._amg_can_coarse_odd_even(op) || throw(ArgumentError(
        "coarse lattice is not bipartite"))
    self_inv = DDalphaAMG._amg_coarse_self_inverse(op)
    tmp1, tmp2, tmp3 = similar(rhs), similar(rhs), similar(rhs)
    schur_rhs = similar(rhs)
    x_even = similar(solution)
    schur_image = similar(rhs)
    schur_residual = similar(rhs)
    DDalphaAMG._amg_coarse_odd_even_rhs!(
        schur_rhs, op, self_inv, rhs, tmp1, tmp2, tmp3)
    DDalphaAMG._amg_coarse_color_mask!(x_even, op, solution, 0)
    DDalphaAMG._amg_apply_coarse_odd_even_schur!(
        schur_image, op, self_inv, x_even, tmp1, tmp2, tmp3)
    schur_residual.data .= schur_rhs.data
    DDalphaAMG.coarse_axpby!(-1.0, schur_image, 1.0, schur_residual)
    return DDalphaAMG.coarse_norm(schur_residual) /
           max(DDalphaAMG.coarse_norm(schur_rhs), eps(Float64))
end

"""
    gd_coarse_gmres(setup, D, rhs; ...)

GD-local restarted GMRES for the shifted coarse correction. It intentionally
does not redefine `DDalphaAMG.coarse_gmres`; convergence is accepted only after
direct true-residual verification.
"""
function gd_coarse_gmres(
        setup::DDalphaAMG.DDalphaAMG_Setup, D,
        b_c::DDalphaAMG.CoarseVector;
        maxiter::Int=20, restart::Int=20, tol::Float64=1e-2,
        coarse_odd_even::Bool=true,
        arnoldi_mode::Symbol=:auto,
        coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing,
        history_callback::Union{Nothing,Function}=nothing)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    restart > 0 || throw(ArgumentError("restart must be positive"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    arnoldi_mode in (:auto, :scalar, :batched) || throw(ArgumentError(
        "arnoldi_mode must be :auto, :scalar, or :batched"))

    has_stencil = setup.coarse_stencil !== nothing
    op = has_stencil ?
         setup.coarse_stencil::DDalphaAMG.CoarseStencilOperator : nothing
    use_odd_even = coarse_odd_even && has_stencil &&
                   DDalphaAMG._amg_can_coarse_odd_even(op)

    local fine_buf1, fine_buf2
    if !has_stencil
        fine_buf1 = similar(setup.test_vectors[1])
        fine_buf2 = similar(setup.test_vectors[1])
    end

    x_c = similar(b_c)
    DDalphaAMG.zero!(x_c)
    rhs_c = b_c
    local oe_self_inv, oe_tmp1, oe_tmp2, oe_tmp3
    if use_odd_even
        oe_self_inv = DDalphaAMG._amg_coarse_self_inverse!(setup, op)
        oe_tmp1, oe_tmp2, oe_tmp3 = similar(b_c), similar(b_c), similar(b_c)
        oe_rhs = similar(b_c)
        DDalphaAMG._amg_coarse_odd_even_rhs!(
            oe_rhs, op, oe_self_inv, b_c, oe_tmp1, oe_tmp2, oe_tmp3)
        rhs_c = oe_rhs
    end

    b_norm = DDalphaAMG.coarse_norm(rhs_c)
    if b_norm < 1e-30
        if use_odd_even
            result = similar(x_c)
            DDalphaAMG._amg_coarse_odd_even_reconstruct!(
                result, op, oe_self_inv, x_c, b_c,
                oe_tmp1, oe_tmp2, oe_tmp3)
            return result
        end
        return x_c
    end
    history_callback === nothing || history_callback(0, 1.0)

    m = min(restart, length(rhs_c.data))
    ncoarse = length(rhs_c.data)
    V = [similar(rhs_c) for _ in 1:(m + 1)]
    w_c, r_c = similar(rhs_c), similar(rhs_c)
    H = zeros(ComplexF64, m + 1, m)
    gamma = zeros(ComplexF64, m + 1)
    givens_c = zeros(ComplexF64, m)
    givens_s = zeros(ComplexF64, m)
    y = zeros(ComplexF64, m)
    use_batched_arnoldi = rhs_c.data isa DDalphaAMG.CuArray &&
                          arnoldi_mode != :scalar
    if arnoldi_mode == :batched && !(rhs_c.data isa DDalphaAMG.CuArray)
        throw(ArgumentError(
            "arnoldi_mode=:batched requires GPU coarse vectors"))
    end
    local arnoldi_basis, arnoldi_coeff_dev, arnoldi_coeff_host
    if use_batched_arnoldi
        arnoldi_basis, arnoldi_coeff_dev, arnoldi_coeff_host =
            DDalphaAMG._ensure_gpu_coarse_arnoldi_cache!(
                setup, ncoarse, m)
    end

    function apply_Dc!(out, input)
        if use_odd_even
            DDalphaAMG._amg_apply_coarse_odd_even_schur!(
                out, op, oe_self_inv, input,
                oe_tmp1, oe_tmp2, oe_tmp3)
        elseif has_stencil
            DDalphaAMG.apply_coarse_stencil!(out, op, input)
        else
            DDalphaAMG.coarse_apply!(
                out, setup, D, input, fine_buf1, fine_buf2)
        end
        coarse_apply_counter !== nothing && (coarse_apply_counter[] += 1)
        return out
    end

    function finalize_solution(input)
        use_odd_even || return input
        result = similar(input)
        DDalphaAMG._amg_coarse_odd_even_reconstruct!(
            result, op, oe_self_inv, input, b_c,
            oe_tmp1, oe_tmp2, oe_tmp3)
        return result
    end

    function true_relative_residual(input)
        result = finalize_solution(input)
        if use_odd_even
            return result,
                   gd_coarse_odd_even_relative_residual(op, result, b_c)
        end
        residual = similar(b_c)
        image = similar(b_c)
        residual.data .= b_c.data
        if has_stencil
            DDalphaAMG.apply_coarse_stencil!(image, op, result)
        else
            DDalphaAMG.coarse_apply!(
                image, setup, D, result, fine_buf1, fine_buf2)
        end
        coarse_apply_counter !== nothing && (coarse_apply_counter[] += 1)
        DDalphaAMG.coarse_axpby!(-1.0, image, 1.0, residual)
        return result, DDalphaAMG.coarse_norm(residual) /
                       max(DDalphaAMG.coarse_norm(b_c), eps(Float64))
    end

    for cycle in 1:maxiter
        r_c.data .= rhs_c.data
        if cycle > 1
            apply_Dc!(w_c, x_c)
            DDalphaAMG.coarse_axpby!(-1.0, w_c, 1.0, r_c)
        end
        beta = DDalphaAMG.coarse_norm(r_c)
        isfinite(beta) || error(
            "GD coarse GMRES produced a non-finite residual norm at cycle $cycle")
        if beta / b_norm <= tol
            candidate, verified = true_relative_residual(x_c)
            history_callback === nothing ||
                history_callback(cycle - 1, verified)
            verified <= tol && return candidate
        end

        fill!(H, 0)
        fill!(gamma, 0)
        fill!(givens_c, 0)
        fill!(givens_s, 0)
        V[1].data .= r_c.data ./ beta
        if use_batched_arnoldi
            view(arnoldi_basis, :, 1) .= V[1].data
        end
        gamma[1] = beta
        j_final = 0

        for j in 1:m
            apply_Dc!(w_c, V[j])
            if use_batched_arnoldi
                H[j + 1, j] =
                    DDalphaAMG._amg_gpu_batched_orthogonalize!(
                        w_c.data, arnoldi_basis, j,
                        arnoldi_coeff_dev, arnoldi_coeff_host)
                for i in 1:j
                    H[i, j] = arnoldi_coeff_host[i]
                end
            else
                for i in 1:j
                    H[i, j] = DDalphaAMG.coarse_dot(V[i], w_c)
                    DDalphaAMG.coarse_axpby!(
                        -H[i, j], V[i], 1.0, w_c)
                end
                H[j + 1, j] = DDalphaAMG.coarse_norm(w_c)
            end
            j_final = j
            happy_breakdown = abs(H[j + 1, j]) <= eps(Float64)
            if !happy_breakdown
                V[j + 1].data .= w_c.data ./ H[j + 1, j]
                if use_batched_arnoldi
                    view(arnoldi_basis, :, j + 1) .= V[j + 1].data
                end
            end

            for i in 1:(j - 1)
                tmp = -givens_s[i] * H[i, j] +
                      givens_c[i] * H[i + 1, j]
                H[i, j] = conj(givens_c[i]) * H[i, j] +
                          conj(givens_s[i]) * H[i + 1, j]
                H[i + 1, j] = tmp
            end
            denominator = sqrt(abs2(H[j, j]) + abs2(H[j + 1, j]))
            denominator > eps(Float64) || error(
                "GD coarse GMRES encountered a singular projected system at Arnoldi step $j")
            givens_s[j] = H[j + 1, j] / denominator
            givens_c[j] = H[j, j] / denominator
            gamma[j + 1] = -givens_s[j] * gamma[j]
            gamma[j] = conj(givens_c[j]) * gamma[j]
            H[j, j] = denominator
            H[j + 1, j] = 0
            if happy_breakdown || abs(gamma[j + 1]) / b_norm <= tol
                break
            end
        end

        fill!(y, 0)
        for i in j_final:-1:1
            y[i] = gamma[i]
            for k in (i + 1):j_final
                y[i] -= H[i, k] * y[k]
            end
            y[i] /= H[i, i]
        end
        for k in 1:j_final
            DDalphaAMG.coarse_axpby!(y[k], V[k], 1.0, x_c)
        end

        # Preserve the previous GD integration's application accounting: it
        # recomputed the reduced-system residual once at every restart boundary
        # before independently verifying the true reconstructed residual.
        r_c.data .= rhs_c.data
        apply_Dc!(w_c, x_c)
        DDalphaAMG.coarse_axpby!(-1.0, w_c, 1.0, r_c)
        DDalphaAMG.coarse_norm(r_c) / b_norm

        candidate, verified = true_relative_residual(x_c)
        isfinite(verified) || error(
            "GD coarse GMRES produced a non-finite true residual at cycle $cycle")
        history_callback === nothing || history_callback(cycle, verified)
        verified <= tol && return candidate
    end
    return first(true_relative_residual(x_c))
end

"""GD-owned adaptive V-cycle using `gd_coarse_gmres` on the coarse grid."""
function gd_adaptive_vcycle(
        D, rhs, setup::DDalphaAMG.DDalphaAMG_Setup;
        post_smooth::Int,
        sap_block_mr_steps::Int,
        coarse_maxiter::Int,
        coarse_restart::Int,
        coarse_tol::Float64,
        smoother::Symbol,
        coarse_odd_even::Bool=true,
        sap_odd_even::Bool=true,
        sap_schedule::Symbol=:boundary8)
    post_smooth >= 0 || throw(ArgumentError("post_smooth must be nonnegative"))
    smoother in (:sap, :mr) || throw(ArgumentError(
        "GD adaptive V-cycle smoother must be :sap or :mr"))

    solution = similar(rhs)
    DDalphaAMG.zero_wf!(solution)
    fine_residual = similar(rhs)
    fine_image = similar(rhs)
    DDalphaAMG._amg_mul_dirac!(fine_image, D, solution)
    DDalphaAMG.LatticeDiracOperators.substitute_fermion!(fine_residual, rhs)
    DDalphaAMG.axpby!(-1, fine_image, 1, fine_residual)

    coarse_rhs = DDalphaAMG._coarse_vector(setup)
    DDalphaAMG.restrict_f2c!(coarse_rhs, setup, fine_residual)
        coarse_solution = gd_coarse_gmres(
        setup, D, coarse_rhs;
        maxiter=coarse_maxiter,
        restart=coarse_restart,
        tol=coarse_tol,
        coarse_odd_even=coarse_odd_even,
        arnoldi_mode=:auto)
    correction = similar(rhs)
    DDalphaAMG.prolongate_c2f!(correction, setup, coarse_solution)
    DDalphaAMG.axpby!(1, correction, 1, solution)

    if post_smooth > 0 && smoother == :sap
        DDalphaAMG.sap_smooth!(
            solution, D, rhs, setup;
            niter=post_smooth,
            block_solver=:mr,
            nblock_mr=sap_block_mr_steps,
            odd_even=sap_odd_even,
            sap_schedule=sap_schedule)
    elseif post_smooth > 0
        DDalphaAMG.mr_smooth!(solution, D, rhs; niter=post_smooth)
    end
    return solution
end

function _gd_make_setup_shell(template, test_vectors, r::Int, Nv::Int;
                              setup_mode::Symbol=:inverse_sap)
    NX, NY, NZ, NT = template.NX, template.NY, template.NZ, template.NT
    NC, NG = template.NC, template.NG
    NBX, NBY, NBZ, NBT = NX ÷ r, NY ÷ r, NZ ÷ r, NT ÷ r
    Nblocks = NBX * NBY * NBZ * NBT
    return DDalphaAMG.DDalphaAMG_Setup{typeof(template)}(
        test_vectors, r, Nv, NX, NY, NZ, NT, NC, NG,
        NBX, NBY, NBZ, NBT, Nblocks, nothing, nothing,
        DDalphaAMG.CoarseVector[], nothing, nothing,
        zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing, setup_mode)
end

"""
    gd_adaptive_amg_setup(D, template, r, Nv; prototypes=nothing, ...)

Run the experiment-specific common adaptive setup without changing
`DDalphaAMG.setup_DDalphaAMG`. Returns the final setup and unsplit refined
prototypes used as the GD initial block.
"""
function gd_adaptive_amg_setup(
        D, template, r::Int, Nv::Int;
        prototypes=nothing,
        nsmooth::Int=3, nvcycle::Int=6,
        setup_smoother::Symbol=:sap,
        setup_sap_iters::Int=3,
        setup_block_solver::Symbol=:mr,
        setup_block_mr_steps::Int=4,
        setup_block_gmres_steps::Int=setup_block_mr_steps,
        vcycle_update_smoother::Symbol=:sap,
        vcycle_post_sap_iters::Int=4,
        vcycle_post_smooth::Int=4,
        vcycle_post_block_mr_steps::Int=4,
        setup_coarse_tol::Float64=5e-2,
        setup_coarse_maxiter::Int=5,
        setup_coarse_restart::Int=30,
        sap_schedule::Symbol=:boundary8,
        use_chirality::Bool=true,
        block_normalize::Bool=true,
        verbose::Bool=false)
    Nv > 0 || throw(ArgumentError("Nv must be positive"))
    nvcycle >= 0 || throw(ArgumentError("nvcycle must be nonnegative"))
    setup_smoother in (:mr, :sap) || throw(ArgumentError(
        "setup_smoother must be :mr or :sap"))
    vcycle_update_smoother in (:mr, :sap) || throw(ArgumentError(
        "vcycle_update_smoother must be :mr or :sap"))
    setup_block_solver in (:mr, :gmres) || throw(ArgumentError(
        "setup_block_solver must be :mr or :gmres"))
    template.NX % r == 0 && template.NY % r == 0 &&
        template.NZ % r == 0 && template.NT % r == 0 ||
        throw(ArgumentError("fine lattice dimensions must be divisible by r"))

    test_vectors = if prototypes === nothing
        generated = [similar(template) for _ in 1:Nv]
        foreach(DDalphaAMG.random_fermion!, generated)
        generated
    else
        length(prototypes) == Nv || throw(ArgumentError(
            "Expected $Nv GD adaptive prototypes, got $(length(prototypes))"))
        [DDalphaAMG._copy_wf(vector) for vector in prototypes]
    end

    # Validate rank and finiteness without changing the numerical starting
    # vectors. Production refinement orthonormalizes the actual vectors at the
    # same point as the pre-isolation implementation: before each adaptive
    # cycle, after initial inverse smoothing.
    validation_vectors = [DDalphaAMG._copy_wf(vector) for vector in test_vectors]
    gd_global_orthonormalize!(validation_vectors)

    setup_shell = _gd_make_setup_shell(template, test_vectors, r, Nv)
    initial_steps = setup_smoother == :sap ? setup_sap_iters : nsmooth
    DDalphaAMG._amg_initial_inverse_smooth!(
        test_vectors, D, setup_shell;
        smoother=setup_smoother, nstages=initial_steps,
        block_solver=setup_block_solver,
        nblock_mr=setup_block_mr_steps,
        nblock_gmres=setup_block_gmres_steps,
        sap_schedule=sap_schedule)

    for cycle in 1:nvcycle
        gd_global_orthonormalize!(test_vectors)
        coarse_vectors = [DDalphaAMG._copy_wf(vector) for vector in test_vectors]
        if use_chirality
            coarse_vectors = DDalphaAMG._chiral_split_vectors(coarse_vectors)
            DDalphaAMG._block_orthonormalize_chiral_split!(
                coarse_vectors, r; normalize=block_normalize)
        else
            DDalphaAMG.block_orthonormalize!(
                coarse_vectors, r; normalize=block_normalize)
        end
        adaptive_setup = _gd_make_setup_shell(
            template, coarse_vectors, r, length(coarse_vectors))
        DDalphaAMG.precompute_coarse_operator!(
            adaptive_setup, D; mode=:direct_stencil)
        for index in eachindex(test_vectors)
            updated = gd_adaptive_vcycle(
                D, test_vectors[index], adaptive_setup;
                post_smooth=(vcycle_update_smoother == :sap ?
                             vcycle_post_sap_iters : vcycle_post_smooth),
                sap_block_mr_steps=vcycle_post_block_mr_steps,
                coarse_maxiter=setup_coarse_maxiter,
                coarse_restart=setup_coarse_restart,
                coarse_tol=setup_coarse_tol,
                smoother=vcycle_update_smoother,
                coarse_odd_even=true,
                sap_odd_even=true,
                sap_schedule=sap_schedule)
            DDalphaAMG.LatticeDiracOperators.substitute_fermion!(
                test_vectors[index], updated)
            DDalphaAMG.normalize_fermion!(test_vectors[index])
        end
        verbose && println(
            "  [GD inverse-SAP setup] iteration $cycle/$nvcycle done")
    end

    refined = [DDalphaAMG._copy_wf(vector) for vector in test_vectors]
    setup = DDalphaAMG.setup_DDalphaAMG(
        D, template, r, Nv;
        setup_mode=:inverse_sap,
        nsmooth=0,
        verbose=verbose,
        coarse_vectors=refined,
        smoother=:mr,
        nvcycle=0,
        use_chirality=use_chirality,
        precompute_coarse=true,
        block_normalize=block_normalize,
        smooth_external_vectors=false)
    return (; setup, prototypes=refined)
end
