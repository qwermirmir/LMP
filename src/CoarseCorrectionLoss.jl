using LinearAlgebra
using CUDA
using LatticeDiracOperators
using Random

import .Layer_types: Layer, Identity, ActivationFunction
import .activation
import .Layer_sequence
import .DDalphaAMG: DDalphaAMG_Setup, CoarseVector, setup_DDalphaAMG,
                   restrict_f2c!, prolongate_c2f!, restrict_f2c, prolongate_c2f,
                   mr_smooth!
import .gpu: zero_wf!

export CoarseCorrectionLoss_Layer
export coarse_correction_loss_forward!
export refresh_error_vector!

const CCL_WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const CCL_WF3x1GPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}
const CCL_WF3x1 = Union{CCL_WF3x1CPU, CCL_WF3x1GPU}

mutable struct CCL_ErrorCache
    error::Any
    smooth_error::Any
    y::Any
    bc::Any
    Ae::Any
    r::Any
    post_r::Any
    Ar::Any
    tmp::Any
    tmp2::Any
    lambda::Any
    mr_s::Vector{Any}
    mr_t::Vector{Any}
    mr_alpha::Vector{ComplexF64}
    mr_den::Vector{Float64}
    denominator::Float64
    objective_grad_scale::Float64
    loss::Float64
    ratio::Float64
    gain::Float64
    orth_residual::Float64
end

function CCL_ErrorCache(error)
    return CCL_ErrorCache(error, nothing, nothing, nothing, nothing, nothing, nothing,
                          nothing, nothing, nothing, nothing, Any[], Any[], ComplexF64[], Float64[], NaN, 1.0,
                          NaN, NaN, NaN, Inf)
end

mutable struct CoarseCorrectionLoss_Layer <: Layer
    D::Any
    error_vector::Any
    block_size::Int
    nsmooth::Int
    use_chirality::Bool
    coarse_tol::Float64
    coarse_maxiter::Int
    coarse_restart::Int
    operator::Symbol
    objective::Symbol
    gain_eps::Float64
    objective_scale::Float64
    tangent_project::Bool
    orthonormalization_pullback::Bool
    refresh_error::Bool
    error_smooth_steps::Int
    cycle_smoother::Symbol
    cycle_omega::Float64
    cycle_pre_smooth_steps::Int
    cycle_post_smooth_steps::Int
    block_normalize::Bool
    activation::ActivationFunction
    target::Any

    setup::Any
    y::Any
    bc::Any
    orth_residual::Float64
    loss_value::Float64
    ratio_value::Float64
    gain_value::Float64
    objective_grad_scale::Float64
    _Ae::Any
    _r::Any
    _Ar::Any
    _tmp::Any
    _tmp2::Any
    _lambda::Any
    _raw_setup_vectors::Vector{Any}
    _orth_split_grad::Vector{Any}
    _split_grad::Vector{Any}
    _one_split_grad::Vector{Any}
    _delta::Vector{Any}
    _error_caches::Vector{Any}
    _replay_buffer::Vector{Any}
    _replay_next::Int
    _replay_filled::Int
    replay_used::Int
    replay_stored::Int

    function CoarseCorrectionLoss_Layer(D, error_vector;
                                        block_size::Int=4,
                                        nsmooth::Int=0,
                                        use_chirality::Bool=true,
                                        coarse_tol::Real=1e-6,
                                        coarse_maxiter::Int=200,
                                        coarse_restart::Int=50,
                                        operator::Symbol=:dirac,
                                        objective::Symbol=:ratio,
                                        gain_eps::Real=1e-12,
                                        objective_scale::Real=1.0,
                                        tangent_project::Bool=true,
                                        orthonormalization_pullback::Bool=true,
                                        refresh_error::Bool=false,
                                        error_smooth_steps::Int=16,
                                        cycle_smoother::Symbol=:mr,
                                        cycle_omega::Real=0.05,
                                        cycle_pre_smooth_steps::Int=0,
                                        cycle_post_smooth_steps::Int=0,
                                        block_normalize::Bool=true,
                                        activation::ActivationFunction=Identity())
        operator in (:dirac, :normal, :gamma5dirac) ||
            throw(ArgumentError("operator must be :dirac, :normal, or :gamma5dirac, got $operator"))
        objective in (:ratio, :negloggain, :hutchinson, :hutchinson_abs, :frobenius, :cycle_frobenius) ||
            throw(ArgumentError("objective must be :ratio, :negloggain, :hutchinson, :hutchinson_abs, :frobenius, or :cycle_frobenius, got $objective"))
        cycle_smoother in (:mr, :richardson) ||
            throw(ArgumentError("cycle_smoother must be :mr or :richardson, got $cycle_smoother"))
        new(D, error_vector, block_size, nsmooth, use_chirality,
            Float64(coarse_tol), coarse_maxiter, coarse_restart, operator,
            objective, Float64(gain_eps), Float64(objective_scale),
            tangent_project, orthonormalization_pullback,
            refresh_error, error_smooth_steps,
            cycle_smoother, Float64(cycle_omega),
            cycle_pre_smooth_steps, cycle_post_smooth_steps,
            block_normalize,
            activation,
            nothing, nothing, nothing, nothing, Inf, NaN, NaN, NaN, 1.0,
            nothing, nothing, nothing, nothing, nothing, nothing,
            Any[], Any[], Any[], Any[], Any[], Any[],
            Any[], 1, 0, 0, 0)
    end
end

_field_norm2(x) = real(inner_product(x, x))
_field_norm(x) = sqrt(max(_field_norm2(x), 0.0))

function _z4_probe_array!(a::AbstractArray{ComplexF64}, rng::AbstractRNG=Random.default_rng())
    @inbounds for i in eachindex(a)
        r = rand(rng, UInt8) & 0x03
        a[i] = r == 0x00 ? 1.0 + 0.0im :
               r == 0x01 ? -1.0 + 0.0im :
               r == 0x02 ? 0.0 + 1.0im :
                            0.0 - 1.0im
    end
    return a
end

function _z4_probe!(x::CCL_WF3x1, rng::AbstractRNG=Random.default_rng())
    if x.f isa CuArray
        host = Array{ComplexF64}(undef, size(x.f))
        _z4_probe_array!(host, rng)
        copyto!(x.f, host)
    else
        _z4_probe_array!(x.f, rng)
    end
    return x
end

_uses_hutchinson_probe(objective::Symbol) = objective in (:hutchinson, :hutchinson_abs, :cycle_frobenius)

function refresh_error_vector!(layer::CoarseCorrectionLoss_Layer, D_new=layer.D)
    errors = _error_vectors(layer)
    isempty(errors) && return layer

    for e in errors
        if _uses_hutchinson_probe(layer.objective)
            _z4_probe!(e)
        else
            gauss_distribution_fermion!(e)
        end
        if !_uses_hutchinson_probe(layer.objective) && layer.error_smooth_steps > 0
            zero_rhs = similar(e)
            zero_wf!(zero_rhs)
            mr_smooth!(e, D_new, zero_rhs; niter=layer.error_smooth_steps)
        end
        if !_uses_hutchinson_probe(layer.objective)
            nrm = _field_norm(e)
            nrm > eps(Float64) && axpby!(1 / nrm, e, 0, e)
        end
    end
    return layer
end

function _is_error_collection(e)
    e isa AbstractVector || return false
    e isa CCL_WF3x1 && return false
    isempty(e) && return false
    return all(x -> x isa CCL_WF3x1, e)
end

function _error_vectors(layer::CoarseCorrectionLoss_Layer)
    e = layer.error_vector
    e === nothing && return Any[]
    return _is_error_collection(e) ? collect(e) : Any[e]
end

function _ensure_error_caches!(layer::CoarseCorrectionLoss_Layer, errors)
    if length(layer._error_caches) != length(errors)
        layer._error_caches = Any[CCL_ErrorCache(e) for e in errors]
    end
    for i in eachindex(errors)
        cache = layer._error_caches[i]
        if !(cache isa CCL_ErrorCache) || cache.error !== errors[i]
            layer._error_caches[i] = CCL_ErrorCache(errors[i])
        end
    end
    return layer._error_caches
end

function _copyto_wf!(dst, src)
    axpby!(1, src, 0, dst)
    return dst
end

function _copy_wf_ccl(x)
    y = similar(x)
    _copyto_wf!(y, x)
    return y
end

function _raw_setup_vectors_for_pullback(z::AbstractVector{<:CCL_WF3x1}, use_chirality::Bool)
    if use_chirality
        raw = Any[]
        sizehint!(raw, 2length(z))
        for v in z
            plus = _copy_wf_ccl(v)
            minus = _copy_wf_ccl(v)
            chiral_project!(plus, :plus)
            chiral_project!(minus, :minus)
            push!(raw, plus)
            push!(raw, minus)
        end
        return raw
    else
        return Any[_copy_wf_ccl(v) for v in z]
    end
end

function _block_sites_cpu(setup::DDalphaAMG_Setup)
    counts = zeros(Int, setup.Nblocks)
    bmap = Vector{Int}(undef, setup.NX * setup.NY * setup.NZ * setup.NT)
    idx = 0
    @inbounds for it in 1:setup.NT, iz in 1:setup.NZ, iy in 1:setup.NY, ix in 1:setup.NX
        idx += 1
        bx = (ix - 1) ÷ setup.r + 1
        by = (iy - 1) ÷ setup.r + 1
        bz = (iz - 1) ÷ setup.r + 1
        bt = (it - 1) ÷ setup.r + 1
        bid = DDalphaAMG.block_linear_index(bx, by, bz, bt, setup.NBX, setup.NBY, setup.NBZ)
        bmap[idx] = bid
        counts[bid] += 1
    end
    max_count = maximum(counts)
    sites = zeros(Int, max_count, setup.Nblocks)
    pos = zeros(Int, setup.Nblocks)
    @inbounds for site in eachindex(bmap)
        bid = bmap[site]
        pos[bid] += 1
        sites[pos[bid], bid] = site
    end
    return sites, counts
end

function _block_sites_for_pullback(setup::DDalphaAMG_Setup)
    if setup.test_vectors[1] isa CCL_WF3x1GPU
        cache = DDalphaAMG._ensure_gpu_cache!(setup)
        return Array(cache[:block_sites]), Array(cache[:block_counts])
    end
    return _block_sites_cpu(setup)
end

_field_array_cpu(x) = x.f isa CuArray ? Array(x.f) : copy(x.f)

@inline function _site_component_linear(site::Int, a::Int,
                                        NX::Int, NY::Int, NZ::Int, NT::Int,
                                        NC::Int, NG::Int)
    site0 = site - 1
    ic = mod(a - 1, NC) + 1
    ig = (a - 1) ÷ NC + 1
    volume = NX * NY * NZ * NT
    return ic + NC * site0 + NC * volume * (ig - 1)
end

@inline function _field_component_linear(a::AbstractArray, site::Int, comp::Int,
                                         NX::Int, NY::Int, NZ::Int, NT::Int,
                                         NC::Int, NG::Int)
    if ndims(a) == 4
        return (site - 1) * NC * NG + comp
    end
    return _site_component_linear(site, comp, NX, NY, NZ, NT, NC, NG)
end

function _copy_field_array_to_wf!(x, a)
    if x.f isa CuArray
        copyto!(x.f, CuArray(reshape(a, size(x.f))))
    else
        copyto!(x.f, reshape(a, size(x.f)))
    end
    return x
end

function _zero_wf_vector!(xs)
    for x in xs
        zero_wf!(x)
    end
    return xs
end

function _mgs_q_pullback_block(X::Matrix{ComplexF64}, GQ::Matrix{ComplexF64};
                               normalize::Bool=true)
    m, nv = size(X)
    qcols = zeros(ComplexF64, m, nv)
    pre_norm = Vector{Vector{ComplexF64}}(undef, nv)
    norms = zeros(Float64, nv)
    before_proj = [Vector{Vector{ComplexF64}}(undef, max(v - 1, 0)) for v in 1:nv]
    alphas = [Vector{ComplexF64}(undef, max(v - 1, 0)) for v in 1:nv]
    proj_denoms = [Vector{Float64}(undef, max(v - 1, 0)) for v in 1:nv]

    @inbounds for v in 1:nv
        w = copy(view(X, :, v))
        for u in 1:(v - 1)
            before_proj[v][u] = copy(w)
            qu = view(qcols, :, u)
            den = real(dot(qu, qu))
            proj_denoms[v][u] = den
            alpha = normalize || den <= 1e-28 ? dot(qu, w) : dot(qu, w) / den
            alphas[v][u] = alpha
            w .-= alpha .* qu
        end
        if normalize
            pre_norm[v] = copy(w)
            nrm = norm(w)
            norms[v] = nrm
            if nrm > 1e-14
                qcols[:, v] .= w ./ nrm
            else
                qcols[:, v] .= w
            end
        else
            qcols[:, v] .= w
        end
    end

    adj_q = copy(GQ)
    GX = zeros(ComplexF64, m, nv)
    @inbounds for v in nv:-1:1
        qv = view(qcols, :, v)
        gv = copy(view(adj_q, :, v))
        gw = gv
        if normalize
            nrm = norms[v]
            gw = if nrm > 1e-14
                c = real(dot(qv, gv))
                (gv .- c .* qv) ./ nrm
            else
                gv
            end
        end

        for u in (v - 1):-1:1
            qu = view(qcols, :, u)
            w_before = before_proj[v][u]
            alpha = alphas[v][u]
            beta = dot(gw, qu)

            if normalize
                adj_q[:, u] .+= .-conj(alpha) .* gw .- beta .* w_before
                gw .-= conj(beta) .* qu
            else
                den = proj_denoms[v][u]
                if den <= 1e-28
                    adj_q[:, u] .+= .-conj(alpha) .* gw .- beta .* w_before
                    gw .-= conj(beta) .* qu
                else
                    adj_q[:, u] .+= .-conj(alpha) .* gw .-
                                     (beta / den) .* w_before .+
                                     (2real(beta * alpha) / den) .* qu
                    gw .-= (conj(beta) / den) .* qu
                end
            end
        end
        GX[:, v] .+= gw
    end
    return GX
end

function _orthonormalization_pullback_group!(out_grads::Vector, q_grads::Vector,
                                             raw_vectors::Vector, q_vectors::Vector,
                                             setup::DDalphaAMG_Setup,
                                             group::Vector{Int},
                                             block_sites, block_counts,
                                             normalize::Bool)
    isempty(group) && return out_grads
    nv = length(group)
    nc_ng = setup.NC * setup.NG
    raw_arrays = [_field_array_cpu(raw_vectors[g]) for g in group]
    q_arrays = [_field_array_cpu(q_vectors[g]) for g in group]
    g_arrays = [_field_array_cpu(q_grads[g]) for g in group]
    out_arrays = [zeros(ComplexF64, size(raw_arrays[1])) for _ in 1:nv]

    @inbounds for bid in 1:setup.Nblocks
        nsites = Int(block_counts[bid])
        m = nsites * nc_ng
        X = Matrix{ComplexF64}(undef, m, nv)
        Q = Matrix{ComplexF64}(undef, m, nv)
        G = Matrix{ComplexF64}(undef, m, nv)

        row = 0
        for p in 1:nsites
            site = Int(block_sites[p, bid])
            for a in 1:nc_ng
                row += 1
                lin = _field_component_linear(raw_arrays[1], site, a,
                                              setup.NX, setup.NY, setup.NZ,
                                              setup.NT, setup.NC, setup.NG)
                for j in 1:nv
                    X[row, j] = raw_arrays[j][lin]
                    Q[row, j] = q_arrays[j][lin]
                    G[row, j] = conj(g_arrays[j][lin])
                end
            end
        end

        GX = _mgs_q_pullback_block(X, G; normalize=normalize)
        row = 0
        for p in 1:nsites
            site = Int(block_sites[p, bid])
            for a in 1:nc_ng
                row += 1
                lin = _field_component_linear(raw_arrays[1], site, a,
                                              setup.NX, setup.NY, setup.NZ,
                                              setup.NT, setup.NC, setup.NG)
                for j in 1:nv
                    out_arrays[j][lin] = conj(GX[row, j])
                end
            end
        end
    end

    for (j, gidx) in enumerate(group)
        _copy_field_array_to_wf!(out_grads[gidx], out_arrays[j])
    end
    return out_grads
end

function _orthonormalization_pullback!(out_grads::Vector, q_grads::Vector,
                                       raw_vectors::Vector, setup::DDalphaAMG_Setup,
                                       use_chirality::Bool, normalize::Bool)
    length(raw_vectors) == setup.Nv || error("raw setup vector count does not match setup.Nv")
    length(q_grads) == setup.Nv || error("q gradient count does not match setup.Nv")
    _zero_wf_vector!(out_grads)
    block_sites, block_counts = _block_sites_for_pullback(setup)
    if use_chirality
        plus_group = collect(1:2:setup.Nv)
        minus_group = collect(2:2:setup.Nv)
        _orthonormalization_pullback_group!(out_grads, q_grads, raw_vectors,
                                            setup.test_vectors, setup,
                                            plus_group, block_sites, block_counts,
                                            normalize)
        _orthonormalization_pullback_group!(out_grads, q_grads, raw_vectors,
                                            setup.test_vectors, setup,
                                            minus_group, block_sites, block_counts,
                                            normalize)
    else
        _orthonormalization_pullback_group!(out_grads, q_grads, raw_vectors,
                                            setup.test_vectors, setup,
                                            collect(1:setup.Nv), block_sites, block_counts,
                                            normalize)
    end
    return out_grads
end

function _apply_dagdiracdirac!(out, D, tmp, x)
    mul!(tmp, D, x)
    mul!(out, D', tmp)
    return out
end

struct CCL_Gamma5DiracOperator{TD}
    D::TD
end

Base.adjoint(A::CCL_Gamma5DiracOperator) = A

function LinearAlgebra.mul!(out, A::CCL_Gamma5DiracOperator, x)
    mul!(out, A.D, x)
    DDalphaAMG.gamma5!(out)
    return out
end

function _apply_gamma5dirac!(out, D, x)
    mul!(out, D, x)
    DDalphaAMG.gamma5!(out)
    return out
end

function _coarse_apply_A!(out::CoarseVector, setup::DDalphaAMG_Setup, D,
                          x::CoarseVector, fine1, fine2, fine3)
    prolongate_c2f!(fine1, setup, x)
    mul!(fine2, D, fine1)
    mul!(fine3, D', fine2)
    restrict_f2c!(out, setup, fine3)
    return out
end

function _coarse_cg_A(setup::DDalphaAMG_Setup, D, b::CoarseVector;
                      tol::Real=1e-6, maxiter::Int=200)
    x = similar(b)
    r = similar(b)
    p = similar(b)
    Ap = similar(b)
    DDalphaAMG.zero!(x)
    copyto!(r.data, b.data)
    copyto!(p.data, r.data)

    fine1 = similar(setup.test_vectors[1])
    fine2 = similar(setup.test_vectors[1])
    fine3 = similar(setup.test_vectors[1])

    rsold = real(DDalphaAMG.coarse_dot(r, r))
    bnorm = sqrt(max(real(DDalphaAMG.coarse_dot(b, b)), eps(Float64)))
    rsold <= (tol * bnorm)^2 && return x, 0, 0.0

    rel = sqrt(rsold) / bnorm
    steps = 0
    for k in 1:maxiter
        _coarse_apply_A!(Ap, setup, D, p, fine1, fine2, fine3)
        denom = real(DDalphaAMG.coarse_dot(p, Ap))
        abs(denom) <= eps(Float64) && break
        α = rsold / denom
        DDalphaAMG.coarse_axpby!(α, p, 1, x)
        DDalphaAMG.coarse_axpby!(-α, Ap, 1, r)
        rsnew = real(DDalphaAMG.coarse_dot(r, r))
        rel = sqrt(max(rsnew, 0.0)) / bnorm
        steps = k
        rel <= tol && break
        β = rsnew / rsold
        DDalphaAMG.coarse_axpby!(1, r, β, p)
        rsold = rsnew
    end
    return x, steps, rel
end

function _coarse_apply_dirac_matrixfree!(out::CoarseVector, setup::DDalphaAMG_Setup, Dop,
                                         x::CoarseVector, fine1, fine2)
    prolongate_c2f!(fine1, setup, x)
    mul!(fine2, Dop, fine1)
    restrict_f2c!(out, setup, fine2)
    return out
end

function _coarse_gmres_matrixfree(setup::DDalphaAMG_Setup, Dop, b_c::CoarseVector;
                                  maxiter::Int=50, restart::Int=20, tol::Float64=1e-6,
                                  coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                                  history_callback::Union{Nothing,Function}=nothing)
    x_c = similar(b_c)
    DDalphaAMG.zero!(x_c)
    b_norm = DDalphaAMG.coarse_norm(b_c)
    b_norm < 1e-30 && return x_c

    fine_buf1 = similar(setup.test_vectors[1])
    fine_buf2 = similar(setup.test_vectors[1])
    m = min(restart, maxiter)
    V = [similar(b_c) for _ in 1:(m + 1)]
    w_c = similar(b_c)
    r_c = similar(b_c)
    H = zeros(ComplexF64, m + 1, m)
    e1 = zeros(ComplexF64, m + 1)
    trial_x = similar(b_c)
    trial_image = similar(b_c)
    trial_residual = similar(b_c)

    function apply!(w, v)
        _coarse_apply_dirac_matrixfree!(w, setup, Dop, v, fine_buf1, fine_buf2)
        coarse_apply_counter === nothing || (coarse_apply_counter[] += 1)
        return w
    end

    function verified_trial_residual!(iteration, estimated_relative_residual,
                                      base_x, basis, coefficients)
        trial_x.data .= base_x.data
        for k in eachindex(coefficients)
            DDalphaAMG.coarse_axpby!(coefficients[k], basis[k], 1.0, trial_x)
        end
        _coarse_apply_dirac_matrixfree!(
            trial_image, setup, Dop, trial_x, fine_buf1, fine_buf2)
        trial_residual.data .= b_c.data
        DDalphaAMG.coarse_axpby!(-1.0, trial_image, 1.0, trial_residual)
        true_relative_residual = DDalphaAMG.coarse_norm(trial_residual) / b_norm
        history_callback === nothing || history_callback(
            iteration, estimated_relative_residual, true_relative_residual)
        return true_relative_residual
    end

    remaining = maxiter
    completed = 0
    history_callback === nothing || history_callback(0, 1.0, 1.0)
    while remaining > 0
        r_c.data .= b_c.data
        apply!(w_c, x_c)
        DDalphaAMG.coarse_axpby!(-1.0, w_c, 1.0, r_c)
        β = DDalphaAMG.coarse_norm(r_c)
        β / b_norm < tol && return x_c

        V[1].data .= r_c.data ./ β
        fill!(H, 0)
        steps = min(m, remaining)
        for j in 1:steps
            apply!(w_c, V[j])
            for i in 1:j
                H[i, j] = DDalphaAMG.coarse_dot(V[i], w_c)
                DDalphaAMG.coarse_axpby!(-H[i, j], V[i], 1.0, w_c)
            end
            H[j + 1, j] = DDalphaAMG.coarse_norm(w_c)
            trial_rhs = zeros(ComplexF64, j + 1)
            trial_rhs[1] = β
            trial_y = H[1:(j + 1), 1:j] \ trial_rhs
            estimated_relative_residual =
                norm(trial_rhs - H[1:(j + 1), 1:j] * trial_y) / b_norm
            true_relative_residual =
                (history_callback !== nothing || estimated_relative_residual <= tol) ?
                verified_trial_residual!(completed + j,
                    estimated_relative_residual, x_c, V, trial_y) : Inf
            if estimated_relative_residual <= tol &&
               true_relative_residual <= tol
                x_c.data .= trial_x.data
                return x_c
            end
            if abs(H[j + 1, j]) > 1e-14
            V[j + 1].data .= w_c.data ./ H[j + 1, j]
        else
            steps = j
            break
        end
    end
        fill!(e1, 0)
        e1[1] = β
        y = H[1:(steps + 1), 1:steps] \ e1[1:(steps + 1)]
        for k in 1:steps
            DDalphaAMG.coarse_axpby!(y[k], V[k], 1.0, x_c)
        end
        remaining -= steps
        completed += steps
    end
    return x_c
end

function _ensure_cache_buffers!(cache::CCL_ErrorCache, e)
    cache.smooth_error = cache.smooth_error === nothing || typeof(cache.smooth_error) != typeof(e) ? similar(e) : cache.smooth_error
    cache.Ae = cache.Ae === nothing || typeof(cache.Ae) != typeof(e) ? similar(e) : cache.Ae
    cache.r  = cache.r  === nothing || typeof(cache.r)  != typeof(e) ? similar(e) : cache.r
    cache.post_r = cache.post_r === nothing || typeof(cache.post_r) != typeof(e) ? similar(e) : cache.post_r
    cache.Ar = cache.Ar === nothing || typeof(cache.Ar) != typeof(e) ? similar(e) : cache.Ar
    cache.tmp = cache.tmp === nothing || typeof(cache.tmp) != typeof(e) ? similar(e) : cache.tmp
    cache.tmp2 = cache.tmp2 === nothing || typeof(cache.tmp2) != typeof(e) ? similar(e) : cache.tmp2
    return cache
end

function _apply_linear_cycle_smoother!(out, layer::CoarseCorrectionLoss_Layer, input, work1, work2, steps::Int; adjoint::Bool=false)
    _copyto_wf!(out, input)
    steps <= 0 && return out
    ω = layer.cycle_omega
    for _ in 1:steps
        if layer.operator == :dirac
            mul!(work1, adjoint ? layer.D' : layer.D, out)
        elseif layer.operator == :gamma5dirac
            _apply_gamma5dirac!(work1, layer.D, out)
        else
            _apply_dagdiracdirac!(work1, layer.D, work2, out)
        end
        axpby!(-ω, work1, 1, out)
    end
    return out
end

function _apply_post_smoother_adjoint!(out, layer::CoarseCorrectionLoss_Layer, input, work1, work2)
    _copyto_wf!(out, input)
    if layer.cycle_post_smooth_steps <= 0
        return out
    elseif layer.cycle_smoother == :richardson
        ω = layer.cycle_omega
        for _ in 1:layer.cycle_post_smooth_steps
            if layer.operator == :dirac
                mul!(work1, layer.D', out)
            elseif layer.operator == :gamma5dirac
                _apply_gamma5dirac!(work1, layer.D, out)
            else
                _apply_dagdiracdirac!(work1, layer.D, work2, out)
            end
            axpby!(-ω, work1, 1, out)
        end
        return out
    else
        error("exact adjoint for cycle_smoother=$(layer.cycle_smoother) post smoothing is not implemented")
    end
end

function _ensure_mr_records!(cache::CCL_ErrorCache, template, nsteps::Int)
    if length(cache.mr_s) != nsteps || (nsteps > 0 && typeof(cache.mr_s[1]) != typeof(template))
        cache.mr_s = Any[similar(template) for _ in 1:nsteps]
        cache.mr_t = Any[similar(template) for _ in 1:nsteps]
        cache.mr_alpha = zeros(ComplexF64, nsteps)
        cache.mr_den = zeros(Float64, nsteps)
    end
    return cache
end

function _mr_zero_smooth_with_records!(out, layer::CoarseCorrectionLoss_Layer, input, cache::CCL_ErrorCache)
    nsteps = layer.cycle_post_smooth_steps
    _copyto_wf!(out, input)
    _ensure_mr_records!(cache, input, nsteps)
    for k in 1:nsteps
        s = cache.mr_s[k]
        t = cache.mr_t[k]
        mul!(s, layer.D, out)
        axpby!(-1, s, 0, s)       # s = -D*out, i.e. b-Dx with b=0.
        mul!(t, layer.D, s)
        num = inner_product(t, s)
        den = real(inner_product(t, t))
        if abs(den) < 1e-30
            cache.mr_alpha[k] = 0
            cache.mr_den[k] = den
        else
            α = num / den
            cache.mr_alpha[k] = ComplexF64(α)
            cache.mr_den[k] = Float64(den)
            axpby!(α, s, 1, out)
        end
    end
    return out
end

function _apply_mr_post_smoother_adjoint!(out, layer::CoarseCorrectionLoss_Layer,
                                          input_bar, cache::CCL_ErrorCache)
    nsteps = layer.cycle_post_smooth_steps
    _copyto_wf!(out, input_bar)
    nsteps <= 0 && return out

    gs = cache.tmp
    gt = cache.tmp2
    work = cache.Ae
    work2 = cache.smooth_error
    for k in nsteps:-1:1
        den = cache.mr_den[k]
        α = cache.mr_alpha[k]
        if abs(den) < 1e-30
            continue
        end
        s = cache.mr_s[k]
        t = cache.mr_t[k]

        c = inner_product(out, s)
        β = c / den
        γ = -c * α / den

        # gs is the adjoint wrt s in y = x + alpha*s.
        _copyto_wf!(gs, out)
        axpby!(conj(α), gs, 0, gs)
        axpby!(conj(β), t, 1, gs)

        # gt is the adjoint wrt t from alpha=<t,s>/<t,t>.
        axpby!(β, s, 0, gt)
        axpby!(2real(γ), t, 1, gt)

        # xbar = ybar - D' * gs - D' * D' * gt.
        mul!(work, layer.D', gs)
        axpby!(-1, work, 1, out)
        mul!(work, layer.D', gt)
        mul!(work2, layer.D', work)
        axpby!(-1, work2, 1, out)
    end
    return out
end

function _smooth_error_for_cycle!(layer::CoarseCorrectionLoss_Layer, cache::CCL_ErrorCache)
    e = cache.error
    if layer.cycle_pre_smooth_steps <= 0
        _copyto_wf!(cache.smooth_error, e)
    elseif layer.cycle_smoother == :richardson
        _apply_linear_cycle_smoother!(cache.smooth_error, layer, e, cache.tmp, cache.tmp2,
                                      layer.cycle_pre_smooth_steps; adjoint=false)
    else
        _copyto_wf!(cache.smooth_error, e)
        zero_wf!(cache.tmp)
        mr_smooth!(cache.smooth_error, layer.D, cache.tmp; niter=layer.cycle_pre_smooth_steps)
    end
    return cache.smooth_error
end

function _post_smooth_residual_for_cycle!(layer::CoarseCorrectionLoss_Layer, cache::CCL_ErrorCache)
    if layer.cycle_post_smooth_steps <= 0
        _copyto_wf!(cache.post_r, cache.r)
    elseif layer.cycle_smoother == :richardson
        _apply_linear_cycle_smoother!(cache.post_r, layer, cache.r, cache.tmp, cache.tmp2,
                                      layer.cycle_post_smooth_steps; adjoint=false)
    else
        _mr_zero_smooth_with_records!(cache.post_r, layer, cache.r, cache)
    end
    return cache.post_r
end

function _coarse_correction_loss_forward_one!(layer::CoarseCorrectionLoss_Layer,
                                              setup::DDalphaAMG_Setup,
                                              cache::CCL_ErrorCache)
    e0 = cache.error
    _ensure_cache_buffers!(cache, e0)
    cycle_error_objective = layer.objective == :cycle_frobenius
    full_cycle_objective = layer.objective in (:hutchinson, :hutchinson_abs, :frobenius, :cycle_frobenius)

    # Hutchinson/Frobenius mode estimates the full propagated error norm.
    # For gamma5dirac, H = gamma5*D is only the coarse operator; the loss
    # intentionally does not use ||H E e0||^2, because that would suppress
    # the low modes multigrid is meant to capture.
    # The :cycle_frobenius objective estimates the V-cycle error propagator itself:
    #     ||E e0||^2, with ||E e0||^2 / ||e0||^2 kept as the diagnostic ratio.
    #
    # The older :ratio objective keeps the conditional denominator after
    # pre-smoothing:
    #     ||A E e0||^2 / ||A S_pre e0||^2.
    initial_denominator = NaN
    if cycle_error_objective
        initial_denominator = max(real(inner_product(e0, e0)), eps(Float64))
    elseif full_cycle_objective
        if layer.operator == :dirac
            mul!(cache.tmp, layer.D, e0)
            initial_denominator = max(real(inner_product(cache.tmp, cache.tmp)), eps(Float64))
        elseif layer.operator == :gamma5dirac
            initial_denominator = max(real(inner_product(e0, e0)), eps(Float64))
        else
            _apply_dagdiracdirac!(cache.tmp, layer.D, cache.tmp2, e0)
            initial_denominator = max(real(inner_product(e0, cache.tmp)), eps(Float64))
        end
    end

    e = _smooth_error_for_cycle!(layer, cache)
    if layer.operator == :dirac
        mul!(cache.Ae, layer.D, e)  # De: the fine residual induced by error e.
        bc = restrict_f2c(setup, cache.Ae)
        y = DDalphaAMG.coarse_gmres(setup, layer.D, bc;
                                    maxiter=layer.coarse_maxiter,
                                    restart=layer.coarse_restart,
                                    tol=layer.coarse_tol)
    elseif layer.operator == :gamma5dirac
        _apply_gamma5dirac!(cache.Ae, layer.D, e)  # He with H = gamma5 D.
        bc = restrict_f2c(setup, cache.Ae)
        y = _coarse_gmres_matrixfree(setup, CCL_Gamma5DiracOperator(layer.D), bc;
                                     maxiter=layer.coarse_maxiter,
                                     restart=layer.coarse_restart,
                                     tol=layer.coarse_tol)
    else
        _apply_dagdiracdirac!(cache.Ae, layer.D, cache.tmp, e)
        bc = restrict_f2c(setup, cache.Ae)
        y, _, _ = _coarse_cg_A(setup, layer.D, bc;
                               tol=layer.coarse_tol,
                               maxiter=layer.coarse_maxiter)
    end
    cache.bc = bc
    cache.y = y

    prolongate_c2f!(cache.tmp2, setup, y)
    _copyto_wf!(cache.r, e)
    axpby!(-1, cache.tmp2, 1, cache.r)
    q = _post_smooth_residual_for_cycle!(layer, cache)

    local numerator, denominator, opt
    if layer.operator == :dirac
        mul!(cache.tmp, layer.D, q)             # Dq
        mul!(cache.Ar, layer.D', cache.tmp)     # D' D q
        numerator = real(inner_product(cache.tmp, cache.tmp))
        denominator = max(real(inner_product(cache.Ae, cache.Ae)), eps(Float64))
        opt = restrict_f2c(setup, cache.tmp)     # P' D r, should be small after Dc solve.
    elseif layer.operator == :gamma5dirac
        _apply_gamma5dirac!(cache.tmp, layer.D, q)       # Hq, diagnostic only.
        _copyto_wf!(cache.Ar, q)
        numerator = real(inner_product(q, q))
        denominator = max(real(inner_product(e, e)), eps(Float64))
        opt = restrict_f2c(setup, cache.tmp)     # P' H r, should be small after Hc solve.
    else
        _apply_dagdiracdirac!(cache.Ar, layer.D, cache.tmp, q)
        numerator = real(inner_product(q, cache.Ar))
        denominator = max(real(inner_product(e, cache.Ae)), eps(Float64))
        opt = restrict_f2c(setup, cache.Ar)
    end
    if cycle_error_objective
        numerator = real(inner_product(q, q))
        denominator = initial_denominator
        _copyto_wf!(cache.Ar, q)
    end
    if layer.cycle_post_smooth_steps > 0 && layer.cycle_smoother == :richardson
        _copyto_wf!(cache.tmp2, cache.Ar)
        _apply_post_smoother_adjoint!(cache.Ar, layer, cache.tmp2, cache.tmp, cache.Ae)
    elseif layer.cycle_post_smooth_steps > 0 && layer.cycle_smoother == :mr
        _copyto_wf!(cache.tmp2, cache.Ar)
        _apply_mr_post_smoother_adjoint!(cache.Ar, layer, cache.tmp2, cache)
    end
    ratio = numerator / denominator
    gain = 1.0 - ratio
    if layer.objective == :ratio
        loss = ratio
        grad_scale = 1.0
    elseif layer.objective == :negloggain
        effective_gain = max(gain, layer.gain_eps)
        loss = -log(effective_gain)
        grad_scale = 1.0 / effective_gain
    elseif layer.objective == :hutchinson_abs
        denominator = 1.0
        ratio = numerator / initial_denominator
        gain = 1.0 - ratio
        loss = numerator
        grad_scale = 1.0
    elseif cycle_error_objective
        denominator = 1.0
        ratio = numerator / initial_denominator
        gain = 1.0 - ratio
        loss = numerator
        grad_scale = 1.0
    elseif full_cycle_objective
        denominator = initial_denominator
        ratio = numerator / denominator
        gain = 1.0 - ratio
        loss = ratio
        grad_scale = 1.0
    else
        error("unsupported coarse correction objective: $(layer.objective)")
    end
    loss *= layer.objective_scale
    grad_scale *= layer.objective_scale

    orth_residual = sqrt(max(real(DDalphaAMG.coarse_dot(opt, opt)), 0.0)) /
                    max(sqrt(max(real(DDalphaAMG.coarse_dot(bc, bc)), 0.0)), eps(Float64))
    cache.denominator = denominator
    cache.objective_grad_scale = grad_scale
    cache.loss = loss
    cache.ratio = ratio
    cache.gain = gain
    cache.orth_residual = orth_residual
    return loss
end

_replay_buffer_size() = parse(Int, get(ENV, "COARSE_LOSS_REPLAY_BUFFER_SIZE", "0"))
_replay_errors_per_forward() = parse(Int, get(ENV, "COARSE_LOSS_REPLAY_ERRORS_PER_FORWARD", "0"))
_replay_store_per_forward() = parse(Int, get(ENV, "COARSE_LOSS_REPLAY_STORE_PER_FORWARD",
                                            get(ENV, "COARSE_LOSS_REPLAY_ERRORS_PER_FORWARD", "0")))
_allow_approx_post_smooth_backward() =
    lowercase(get(ENV, "COARSE_LOSS_ALLOW_APPROX_POST_SMOOTH_BACKWARD", "false")) in ("1", "true", "yes", "on")

function _ensure_replay_buffer!(layer::CoarseCorrectionLoss_Layer, template)
    nbuf = _replay_buffer_size()
    if nbuf <= 0
        empty!(layer._replay_buffer)
        layer._replay_next = 1
        layer._replay_filled = 0
        return layer._replay_buffer
    end
    if length(layer._replay_buffer) != nbuf ||
       (!isempty(layer._replay_buffer) && typeof(layer._replay_buffer[1]) != typeof(template))
        layer._replay_buffer = Any[similar(template) for _ in 1:nbuf]
        layer._replay_next = 1
        layer._replay_filled = 0
    end
    return layer._replay_buffer
end

function _mix_replay_errors!(layer::CoarseCorrectionLoss_Layer, errors)
    layer.replay_used = 0
    isempty(errors) && return 0
    nbuf = _replay_buffer_size()
    nreq = _replay_errors_per_forward()
    (nbuf <= 0 || nreq <= 0) && return 0
    _ensure_replay_buffer!(layer, errors[1])
    layer._replay_filled <= 0 && return 0

    # Keep slot 1 fresh. Replay slots are chronological:
    #   errors[2] <- residual after one correction,
    #   errors[3] <- residual after two corrections, ...
    # This is an age queue, not hard-negative sampling.
    nuse = min(nreq, layer._replay_filled, max(length(errors) - 1, 0))
    nuse <= 0 && return 0
    for i in 1:nuse
        _copyto_wf!(errors[i + 1], layer._replay_buffer[i])
        layer.replay_used += 1
    end
    return layer.replay_used
end

_is_good_replay_norm(nrm::Real) = isfinite(nrm) && nrm > sqrt(eps(Float64))

function _store_replay_errors!(layer::CoarseCorrectionLoss_Layer, caches)
    layer.replay_stored = 0
    isempty(caches) && return 0
    nbuf = _replay_buffer_size()
    nstore = _replay_store_per_forward()
    (nbuf <= 0 || nstore <= 0) && return 0
    _ensure_replay_buffer!(layer, caches[1].error)

    # Store the remaining error after correction/post-smoothing as detached data.
    # This is intentionally a replay buffer, not backpropagation through past steps.
    # Preserve time/order: cache 1 is the fresh error after one correction,
    # cache 2 is the previous age-1 error after another correction, and so on.
    n = min(nstore, length(caches), nbuf)
    good_stored = 0
    for i in 1:n
        cache = caches[i]
        (isfinite(cache.loss) && isfinite(cache.ratio) && isfinite(cache.gain)) || continue
        src = cache.post_r !== nothing ? cache.post_r : cache.r
        src === nothing && continue
        dst = layer._replay_buffer[i]
        _copyto_wf!(dst, src)
        nrm = _field_norm(dst)
        _is_good_replay_norm(nrm) || continue
        axpby!(1 / nrm, dst, 0, dst)
        layer.replay_stored += 1
        good_stored = i
    end
    layer._replay_filled = max(layer._replay_filled, good_stored)
    return layer.replay_stored
end

function coarse_correction_loss_forward!(layer::CoarseCorrectionLoss_Layer,
                                         z::AbstractVector{<:CCL_WF3x1})
    nv = length(z)
    nv > 0 || error("CoarseCorrectionLoss_Layer needs at least one vector")
    errors = _error_vectors(layer)
    isempty(errors) && error("CoarseCorrectionLoss_Layer.error_vector is not set")

    if layer.orthonormalization_pullback
        layer._raw_setup_vectors = _raw_setup_vectors_for_pullback(z, layer.use_chirality)
    else
        empty!(layer._raw_setup_vectors)
    end

    setup = setup_DDalphaAMG(layer.D, z[1], layer.block_size, nv;
                            nsmooth=layer.nsmooth,
                            verbose=false,
                            coarse_vectors=collect(z),
                            use_chirality=layer.use_chirality,
                            precompute_coarse=false,
                            block_normalize=layer.block_normalize)
    layer.setup = setup

    caches = _ensure_error_caches!(layer, errors)
    _mix_replay_errors!(layer, errors)
    total_loss = 0.0
    total_ratio = 0.0
    total_gain = 0.0
    total_orth = 0.0
    for cache in caches
        total_loss += _coarse_correction_loss_forward_one!(layer, setup, cache)
        total_ratio += cache.ratio
        total_gain += cache.gain
        total_orth += cache.orth_residual
    end
    _store_replay_errors!(layer, caches)

    invn = 1 / length(caches)
    layer.loss_value = total_loss * invn
    layer.ratio_value = total_ratio * invn
    layer.gain_value = total_gain * invn
    layer.orth_residual = total_orth * invn
    layer.objective_grad_scale = sum(c.objective_grad_scale for c in caches) * invn
    layer.bc = caches[1].bc
    layer.y = caches[1].y
    layer._Ae = caches[1].Ae
    layer._r = caches[1].r
    layer._Ar = caches[1].Ar
    layer._tmp = caches[1].tmp
    layer._tmp2 = caches[1].tmp2
    return layer.loss_value
end

function (layer::CoarseCorrectionLoss_Layer)(z::AbstractVector{<:CCL_WF3x1})
    return coarse_correction_loss_forward!(layer, z)
end

function _coarse_data_cpu(cv::CoarseVector)
    return cv.data isa CuArray ? Array(cv.data) : copy(cv.data)
end

_coarse_data_for_kernel(cv::CoarseVector) = cv.data
_coarse_data_for_kernel(data::AbstractVector{ComplexF64}) = data

function _assign_coarse_data!(cv::CoarseVector, data::Vector{ComplexF64})
    if cv.data isa CuArray
        copyto!(cv.data, CuArray(data))
    else
        copyto!(cv.data, data)
    end
    return cv
end

function _ccl_basis_gradient_kernel!(g, Ar, y, bmap, nc_ng::Int32, total::Int32,
                                     Nv::Int32, j::Int32, scale::Float64)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        coeff = y[(bid - Int32(1)) * Nv + j]
        @inbounds g[idx] = scale * conj(Ar[idx]) * coeff
    end
    return nothing
end

function _ccl_basis_gradient_add_kernel!(g, Ar, y, bmap, nc_ng::Int32, total::Int32,
                                         Nv::Int32, j::Int32, scale::Float64)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        coeff = y[(bid - Int32(1)) * Nv + j]
        @inbounds g[idx] += scale * conj(Ar[idx]) * coeff
    end
    return nothing
end

function _fill_basis_gradient!(g, Ar, coeffs, setup::DDalphaAMG_Setup, j::Int, scale::Float64)
    coeff_data = _coarse_data_for_kernel(coeffs)
    if g isa CCL_WF3x1GPU
        cache = DDalphaAMG._ensure_gpu_cache!(setup)
        total = Int32(length(g.f))
        nc_ng = Int32(size(g.f, 1) * size(g.f, 2))
        @cuda threads=256 blocks=cld(Int(total), 256) _ccl_basis_gradient_kernel!(
            reshape(g.f, :), reshape(Ar.f, :), coeff_data, cache[:bmap],
            nc_ng, total, Int32(setup.Nv), Int32(j), scale)
        return g
    end

    coeff_cpu = coeff_data isa CuArray ? Array(coeff_data) : coeff_data
    zero_wf!(g)
    r = setup.r
    Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = DDalphaAMG.block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        c = scale * coeff_cpu[(bidx - 1) * Nv + j]
        for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
            it = (bt - 1) * r + lt
            iz = (bz - 1) * r + lz
            iy = (by - 1) * r + ly
            ix = (bx - 1) * r + lx
            for ig in 1:NG, ic in 1:NC
                g[ic, ix, iy, iz, it, ig] = c * conj(Ar[ic, ix, iy, iz, it, ig])
            end
        end
    end
    return g
end

function _add_basis_gradient!(g, Ar, coeffs, setup::DDalphaAMG_Setup, j::Int, scale::Float64)
    coeff_data = _coarse_data_for_kernel(coeffs)
    if g isa CCL_WF3x1GPU
        cache = DDalphaAMG._ensure_gpu_cache!(setup)
        total = Int32(length(g.f))
        nc_ng = Int32(size(g.f, 1) * size(g.f, 2))
        @cuda threads=256 blocks=cld(Int(total), 256) _ccl_basis_gradient_add_kernel!(
            reshape(g.f, :), reshape(Ar.f, :), coeff_data, cache[:bmap],
            nc_ng, total, Int32(setup.Nv), Int32(j), scale)
        return g
    end

    coeff_cpu = coeff_data isa CuArray ? Array(coeff_data) : coeff_data
    r = setup.r
    Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = DDalphaAMG.block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        c = scale * coeff_cpu[(bidx - 1) * Nv + j]
        for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
            it = (bt - 1) * r + lt
            iz = (bz - 1) * r + lz
            iy = (by - 1) * r + ly
            ix = (bx - 1) * r + lx
            for ig in 1:NG, ic in 1:NC
                g[ic, ix, iy, iz, it, ig] += c * conj(Ar[ic, ix, iy, iz, it, ig])
            end
        end
    end
    return g
end

function _ccl_tangent_col_kernel!(out, cols, n::Int32, nblocks::Int32, j::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    total = n * nblocks
    if idx <= total
        b = (idx - Int32(1)) ÷ n
        i = (idx - Int32(1)) % n + Int32(1)
        mij = cols[(j - Int32(1)) * total + b * n + i]
        mji = cols[(i - Int32(1)) * total + b * n + j]
        out[idx] = 0.5 * (mij + conj(mji))
    end
    return nothing
end

function _project_tangent_gpu!(grads::Vector, setup::DDalphaAMG_Setup)
    n = setup.Nv
    nblocks = setup.Nblocks
    total = n * nblocks
    cols = CUDA.zeros(ComplexF64, total * n)
    tmp_cv = _coarse_vector_like(setup)
    for j in 1:n
        restrict_f2c!(tmp_cv, setup, grads[j])
        copyto!(view(cols, ((j - 1) * total + 1):(j * total)), tmp_cv.data)
    end

    hcol = _coarse_vector_like(setup)
    corr = similar(setup.test_vectors[1])
    threads = 256
    blocks = cld(total, threads)
    for j in 1:n
        @cuda threads=threads blocks=blocks _ccl_tangent_col_kernel!(
            hcol.data, cols, Int32(n), Int32(nblocks), Int32(j))
        prolongate_c2f!(corr, setup, hcol)
        axpby!(-1, corr, 1, grads[j])
    end
    return grads
end

function _project_tangent!(grads::Vector, setup::DDalphaAMG_Setup)
    if setup.test_vectors[1] isa CCL_WF3x1GPU
        return _project_tangent_gpu!(grads, setup)
    end

    n = setup.Nv
    nblocks = setup.Nblocks
    mcols = Vector{Vector{ComplexF64}}(undef, n)
    for j in 1:n
        mcols[j] = _coarse_data_cpu(restrict_f2c(setup, grads[j]))
    end

    hcol = _coarse_vector_like(setup)
    corr = similar(setup.test_vectors[1])
    data = Vector{ComplexF64}(undef, n * nblocks)
    for j in 1:n
        for b in 1:nblocks
            base = (b - 1) * n
            for i in 1:n
                mij = mcols[j][base + i]
                mji = mcols[i][base + j]
                data[base + i] = 0.5 * (mij + conj(mji))
            end
        end
        _assign_coarse_data!(hcol, data)
        prolongate_c2f!(corr, setup, hcol)
        axpby!(-1, corr, 1, grads[j])
    end
    return grads
end

function _coarse_vector_like(setup::DDalphaAMG_Setup)
    return DDalphaAMG._coarse_vector(setup)
end

function _merge_chiral_gradient!(delta::Vector, split_grad::Vector, use_chirality::Bool)
    if use_chirality
        n = length(delta)
        @assert length(split_grad) == 2n
        for i in 1:n
            chiral_project!(split_grad[2i - 1], :plus)
            chiral_project!(split_grad[2i], :minus)
            _copyto_wf!(delta[i], split_grad[2i - 1])
            axpby!(1, split_grad[2i], 1, delta[i])
        end
    else
        for i in eachindex(delta)
            _copyto_wf!(delta[i], split_grad[i])
        end
    end
    return delta
end

function _ensure_delta!(layer::CoarseCorrectionLoss_Layer, z)
    n = length(z)
    if length(layer._delta) != n || (!isempty(layer._delta) && typeof(layer._delta[1]) != typeof(z[1]))
        layer._delta = [similar(z[i]) for i in 1:n]
    end
    return layer._delta
end

function activation.DeltaRule(layer::CoarseCorrectionLoss_Layer, ::Identity,
                              z::Vector{T}, batch::Int) where {T<:CCL_WF3x1}
    delta = _ensure_delta!(layer, z)
    activation.DeltaRule!(layer, layer.activation, z, delta, batch)
    return delta
end

function activation.DeltaRule!(layer::CoarseCorrectionLoss_Layer, ::Identity,
                               z::Vector{T}, delta::Vector, batch::Int) where {T<:CCL_WF3x1}
    setup = layer.setup
    setup === nothing && error("CoarseCorrectionLoss_Layer forward must run before backward")
    caches = layer._error_caches
    isempty(caches) && error("CoarseCorrectionLoss_Layer has no cached error solves")

    nP = setup.Nv
    if length(layer._split_grad) != nP || (!isempty(layer._split_grad) && typeof(layer._split_grad[1]) != typeof(setup.test_vectors[1]))
        layer._split_grad = Any[similar(setup.test_vectors[i]) for i in 1:nP]
    end
    if length(layer._one_split_grad) != nP || (!isempty(layer._one_split_grad) && typeof(layer._one_split_grad[1]) != typeof(setup.test_vectors[1]))
        layer._one_split_grad = Any[similar(setup.test_vectors[i]) for i in 1:nP]
    end
    _zero_wf_vector!(layer._split_grad)

    invn = 1 / length(caches)
    for cache in caches
        y = cache.y
        y === nothing && error("CoarseCorrectionLoss_Layer has no cached coarse solve")
        coeffs = y

        if layer.operator == :dirac
            denom = cache.denominator

            # Explicit term from r = e - P*y: -g_r*y', where g_r = 2D'Dr / ||De||^2.
            for j in 1:nP
                _fill_basis_gradient!(layer._one_split_grad[j], cache.Ar, coeffs, setup, j, -2.0 / denom)
            end

            # Implicit terms from y(P) with D_c*y = P'De.
            # The adjoint equation is D_c' * lambda = -P' * g_r.
            axpby!(2.0 / denom, cache.Ar, 0, cache.tmp2)
            ybar = restrict_f2c(setup, cache.tmp2)
            ybar.data .*= -1
            lambda = _coarse_gmres_matrixfree(setup, layer.D', ybar;
                                              maxiter=layer.coarse_maxiter,
                                              restart=layer.coarse_restart,
                                              tol=layer.coarse_tol)
            cache.lambda = lambda
            layer._lambda = lambda
            lambda_coeffs = lambda

            # +(Dr)*lambda'
            mul!(cache.tmp, layer.D, cache.r)
            for j in 1:nP
                _add_basis_gradient!(layer._one_split_grad[j], cache.tmp, lambda_coeffs, setup, j, 1.0)
            end

            # -(D'P*lambda)*y'
            prolongate_c2f!(cache.tmp2, setup, lambda)
            mul!(cache.Ae, layer.D', cache.tmp2)
            for j in 1:nP
                _add_basis_gradient!(layer._one_split_grad[j], cache.Ae, coeffs, setup, j, -1.0)
            end
        elseif layer.operator == :gamma5dirac
            denom = cache.denominator

            # Explicit term from r = e - P*y: -g_r*y', where g_r is the
            # propagated error gradient. H = gamma5*D is used only in the
            # coarse Galerkin solve, not as an extra loss metric.
            for j in 1:nP
                _fill_basis_gradient!(layer._one_split_grad[j], cache.Ar, coeffs, setup, j, -2.0 / denom)
            end

            # Implicit terms from y(P) with H_c*y = P'He.
            # Since H_c is Hermitian, the adjoint equation uses the same coarse operator.
            axpby!(2.0 / denom, cache.Ar, 0, cache.tmp2)
            ybar = restrict_f2c(setup, cache.tmp2)
            ybar.data .*= -1
            lambda = _coarse_gmres_matrixfree(setup, CCL_Gamma5DiracOperator(layer.D), ybar;
                                              maxiter=layer.coarse_maxiter,
                                              restart=layer.coarse_restart,
                                              tol=layer.coarse_tol)
            cache.lambda = lambda
            layer._lambda = lambda
            lambda_coeffs = lambda

            # +(Hr)*lambda'
            _apply_gamma5dirac!(cache.tmp, layer.D, cache.r)
            for j in 1:nP
                _add_basis_gradient!(layer._one_split_grad[j], cache.tmp, lambda_coeffs, setup, j, 1.0)
            end

            # -(H*P*lambda)*y'
            prolongate_c2f!(cache.tmp2, setup, lambda)
            _apply_gamma5dirac!(cache.Ae, layer.D, cache.tmp2)
            for j in 1:nP
                _add_basis_gradient!(layer._one_split_grad[j], cache.Ae, coeffs, setup, j, -1.0)
            end
        else
            scale = -2.0 / cache.denominator
            for j in 1:nP
                _fill_basis_gradient!(layer._one_split_grad[j], cache.Ar, coeffs, setup, j, scale)
            end
            if layer.cycle_post_smooth_steps > 0
                axpby!(2.0 / cache.denominator, cache.Ar, 0, cache.tmp2)
                ybar = restrict_f2c(setup, cache.tmp2)
                ybar.data .*= -1
                lambda, _, _ = _coarse_cg_A(setup, layer.D, ybar;
                                            tol=layer.coarse_tol,
                                            maxiter=layer.coarse_maxiter)
                cache.lambda = lambda
                layer._lambda = lambda
                lambda_coeffs = lambda

                _apply_dagdiracdirac!(cache.tmp, layer.D, cache.tmp2, cache.r)
                for j in 1:nP
                    _add_basis_gradient!(layer._one_split_grad[j], cache.tmp, lambda_coeffs, setup, j, 1.0)
                end

                prolongate_c2f!(cache.tmp2, setup, lambda)
                _apply_dagdiracdirac!(cache.Ae, layer.D, cache.tmp, cache.tmp2)
                for j in 1:nP
                    _add_basis_gradient!(layer._one_split_grad[j], cache.Ae, coeffs, setup, j, -1.0)
                end
            end
        end

        err_scale = cache.objective_grad_scale * invn
        for j in 1:nP
            axpby!(err_scale, layer._one_split_grad[j], 1, layer._split_grad[j])
        end
    end

    if layer.orthonormalization_pullback
        if length(layer._orth_split_grad) != nP || (!isempty(layer._orth_split_grad) && typeof(layer._orth_split_grad[1]) != typeof(setup.test_vectors[1]))
            layer._orth_split_grad = Any[similar(setup.test_vectors[i]) for i in 1:nP]
        end
        _orthonormalization_pullback!(layer._orth_split_grad, layer._split_grad,
                                      layer._raw_setup_vectors, setup, layer.use_chirality,
                                      layer.block_normalize)
        _merge_chiral_gradient!(delta, layer._orth_split_grad, layer.use_chirality)
    else
        layer.tangent_project && _project_tangent!(layer._split_grad, setup)
        _merge_chiral_gradient!(delta, layer._split_grad, layer.use_chirality)
    end
    scale = 1.0 / max(batch, 1)
    if scale != 1.0
        for d in delta
            axpby!(scale, d, 0, d)
        end
    end
    return delta
end

function Layer_sequence._refresh_layer!(layer::CoarseCorrectionLoss_Layer, D_new, U_new)
    layer.D = D_new
    if layer.refresh_error
        refresh_error_vector!(layer, D_new)
    end
    layer.setup = nothing
    layer.y = nothing
    layer.bc = nothing
    for cache in layer._error_caches
        if cache isa CCL_ErrorCache
            cache.y = nothing
            cache.bc = nothing
            cache.lambda = nothing
        end
    end
    empty!(layer._raw_setup_vectors)
    return layer
end

function Layer_sequence._refresh_forward_error!(layer::CoarseCorrectionLoss_Layer, D_new)
    layer.refresh_error || return nothing
    layer.D = D_new
    refresh_error_vector!(layer, D_new)
    layer.setup = nothing
    layer.y = nothing
    layer.bc = nothing
    for cache in layer._error_caches
        if cache isa CCL_ErrorCache
            cache.y = nothing
            cache.bc = nothing
            cache.lambda = nothing
        end
    end
    empty!(layer._raw_setup_vectors)
    return nothing
end
