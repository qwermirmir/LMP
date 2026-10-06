module DDalphaAMG

using ..LMP: inner_product, fused_wilson_mul_gpu!, zero_wf!
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using CUDA
export DDalphaAMG_Setup, CoarseVector
export setup_DDalphaAMG, solve_DDalphaAMG, vcycle
export restrict_f2c, prolongate_c2f, coarse_apply
export restrict_f2c!, prolongate_c2f!, coarse_apply!
export mr_smooth!, sap_smooth!
export block_orthonormalize!
export deflated_initial_guess
export precompute_coarse_operator!
export gamma5!, chiral_project!


const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

# ==========================================
#  Coarse Grid Vector
# ==========================================

"""
    CoarseVector

Coarse-grid vector storing Nv components on each coarse block.
The entry for block `bidx` and vector component `v` is
`data[(bidx - 1) * Nv + v]`.
"""
mutable struct CoarseVector
    data::AbstractVector{ComplexF64}
    Nv::Int
    Nblocks::Int
end

CoarseVector(Nv::Int, Nblocks::Int) =
    CoarseVector(zeros(ComplexF64, Nv * Nblocks), Nv, Nblocks)
CoarseVector(Nv::Int, Nblocks::Int, ::Val{:gpu}) =
    CoarseVector(CUDA.zeros(ComplexF64, Nv * Nblocks), Nv, Nblocks)
CoarseVector(Nv::Int, Nblocks::Int, ::Val{:cpu}) =
    CoarseVector(Nv, Nblocks)

Base.similar(cv::CoarseVector) =
    CoarseVector(similar(cv.data), cv.Nv, cv.Nblocks)
Base.copy(cv::CoarseVector) =
    CoarseVector(copy(cv.data), cv.Nv, cv.Nblocks)
Base.length(cv::CoarseVector) = length(cv.data)

function zero!(cv::CoarseVector)
    fill!(cv.data, zero(ComplexF64))
    return cv
end

function coarse_dot(a::CoarseVector, b::CoarseVector)
    return dot(a.data, b.data)
end

function coarse_norm(a::CoarseVector)
    return sqrt(real(dot(a.data, a.data)))
end

function coarse_axpby!(α, x::CoarseVector, β, y::CoarseVector)
    if y.data isa CuArray
        y.data .= α .* x.data .+ β .* y.data
        return y
    end
    @inbounds @simd for i in eachindex(y.data)
        y.data[i] = α * x.data[i] + β * y.data[i]
    end
    return y
end


# ==========================================
#  Setup Structure
# ==========================================

"""
    DDalphaAMG_Setup

Container for DD-alphaAMG setup data.

Fields include block-orthonormalized test vectors, block geometry, optional
precomputed coarse operators, reusable coarse GMRES work buffers, and GPU cache
storage.
"""
@inline _amg_device(::WF3x1GPU) = Val(:gpu)
@inline _amg_device(::WF3x1CPU) = Val(:cpu)
@inline _coarse_vector(setup) =
    CoarseVector(setup.Nv, setup.Nblocks, _amg_device(setup.test_vectors[1]))
@inline _coarse_vector(Nv::Int, Nblocks::Int, template::WF3x1) =
    CoarseVector(Nv, Nblocks, _amg_device(template))

function _copy_wf(x::T) where {T<:WF3x1}
    y = similar(x)
    LatticeDiracOperators.substitute_fermion!(y, x)
    return y
end

_copy_wf_vector(xs::AbstractVector{<:WF3x1}) = [_copy_wf(x) for x in xs]

@inline _amg_dirac_kappa(D) = getproperty(D, Symbol("\u03ba"))

@inline function _amg_can_fuse_wilson(D, y::WF3x1GPU, x::WF3x1GPU)
    if !(hasproperty(D, :U) && hasproperty(D, Symbol("\u03ba")) && !hasproperty(D, :D_W))
        return false
    end
    U = getproperty(D, :U)
    return length(U) >= 4 && hasproperty(U[1], :U) && getproperty(U[1], :U) isa CuArray
end

function _amg_mul_dirac!(y::WF3x1, D, x::WF3x1;
                         fine_apply_counter::Union{Nothing,Ref{Int}}=nothing)
    fine_apply_counter !== nothing && (fine_apply_counter[] += 1)
    if y isa WF3x1GPU && x isa WF3x1GPU && _amg_can_fuse_wilson(D, y, x)
        fused_wilson_mul_gpu!(y, x, D.U, _amg_dirac_kappa(D), false)
    else
        mul!(y, D, x)
    end
    return y
end

mutable struct DDalphaAMG_Setup{T<:WF3x1}
    test_vectors::Vector{T}
    r::Int
    Nv::Int
    NX::Int; NY::Int; NZ::Int; NT::Int
    NC::Int; NG::Int
    NBX::Int; NBY::Int; NBZ::Int; NBT::Int
    Nblocks::Int
    Dc_matrix::Union{Nothing, Matrix{ComplexF64}}
    coarse_stencil::Union{Nothing, Any}
    _gmres_V::Vector{CoarseVector}
    _gmres_w::Union{Nothing, CoarseVector}
    _gmres_r::Union{Nothing, CoarseVector}
    _gmres_H::Matrix{ComplexF64}
    _gmres_e1::Vector{ComplexF64}
    _gmres_restart::Int
    _gpu_cache::Any
    setup_mode::Symbol
end

# Preserve the pre-mode positional constructor used by internal helpers and tests.
DDalphaAMG_Setup{T}(args::Vararg{Any,23}) where {T<:WF3x1} =
    DDalphaAMG_Setup{T}(args..., :inverse_sap)

@inline function _amg_validate_setup_mode(mode::Symbol)
    mode in (:inverse_sap, :direct_near_null) && return mode
    throw(ArgumentError(
        "setup_mode must be :inverse_sap or :direct_near_null, got $mode"))
end

@inline function _amg_setup_defaults(mode::Symbol)
    _amg_validate_setup_mode(mode)
    if mode == :direct_near_null
        return (; vcycle_post_sap_iters=1,
                coarse_tol=1e-2, coarse_maxiter=50, coarse_restart=20,
                sap_schedule=:twocolor)
    end
    return (; vcycle_post_sap_iters=2,
            coarse_tol=5e-2, coarse_maxiter=5, coarse_restart=30,
            sap_schedule=:boundary8)
end

@inline function _amg_solve_defaults()
    return (; restart=25, smoother=:sap, pre_smooth=0, post_smooth=2,
            coarse_tol=5e-2, coarse_maxiter=50, coarse_restart=30,
            coarse_odd_even=true, coarse_gmres_mode=:incremental,
            sap_odd_even=true,
            sap_schedule=:boundary8)
end

function _amg_resolve_solve_options(; restart::Union{Nothing,Int}=nothing,
                                    smoother::Union{Nothing,Symbol}=nothing,
                                    pre_smooth::Union{Nothing,Int}=nothing,
                                    post_smooth::Union{Nothing,Int}=nothing,
                                    coarse_tol::Union{Nothing,Float64}=nothing,
                                    coarse_maxiter::Union{Nothing,Int}=nothing,
                                    coarse_restart::Union{Nothing,Int}=nothing,
                                    coarse_odd_even::Union{Nothing,Bool}=nothing,
                                    coarse_gmres_mode::Union{Nothing,Symbol}=nothing,
                                    sap_odd_even::Union{Nothing,Bool}=nothing,
                                    sap_schedule::Union{Nothing,Symbol}=nothing)
    defaults = _amg_solve_defaults()
    return (;
        restart=something(restart, defaults.restart),
        smoother=something(smoother, defaults.smoother),
        pre_smooth=something(pre_smooth, defaults.pre_smooth),
        post_smooth=something(post_smooth, defaults.post_smooth),
        coarse_tol=something(coarse_tol, defaults.coarse_tol),
        coarse_maxiter=something(coarse_maxiter, defaults.coarse_maxiter),
        coarse_restart=something(coarse_restart, defaults.coarse_restart),
        coarse_odd_even=something(coarse_odd_even, defaults.coarse_odd_even),
        coarse_gmres_mode=something(coarse_gmres_mode, defaults.coarse_gmres_mode),
        sap_odd_even=something(sap_odd_even, defaults.sap_odd_even),
        sap_schedule=something(sap_schedule, defaults.sap_schedule),
    )
end

mutable struct AMGProfile
    times::Dict{Symbol,Float64}
    counts::Dict{Symbol,Int}
    synchronize::Bool
end

AMGProfile(; synchronize::Bool=true) =
    AMGProfile(Dict{Symbol,Float64}(), Dict{Symbol,Int}(), synchronize)

function _amg_profile_add!(profile::AMGProfile, key::Symbol, seconds::Real)
    profile.times[key] = get(profile.times, key, 0.0) + Float64(seconds)
    return profile
end

_amg_profile_add!(::Nothing, key::Symbol, seconds::Real) = nothing

function _amg_profile_count!(profile::AMGProfile, key::Symbol, n::Integer=1)
    profile.counts[key] = get(profile.counts, key, 0) + Int(n)
    return profile
end

_amg_profile_count!(::Nothing, key::Symbol, n::Integer=1) = nothing

function _amg_profile_sync(profile::Union{Nothing,AMGProfile})
    if profile !== nothing && profile.synchronize
        try
            CUDA.synchronize()
        catch
        end
    end
    return nothing
end

function _amg_profile_time!(f, profile::Union{Nothing,AMGProfile}, key::Symbol)
    profile === nothing && return f()
    _amg_profile_sync(profile)
    t0 = time_ns()
    result = f()
    _amg_profile_sync(profile)
    _amg_profile_add!(profile, key, (time_ns() - t0) / 1.0e9)
    _amg_profile_count!(profile, key)
    return result
end

function _amg_print_profile(io::IO, profile::AMGProfile)
    total = sum(values(profile.times); init=0.0)
    println(io, "  DD-alphaAMG profile breakdown:")
    for (key, seconds) in sort(collect(profile.times); by=x -> x[2], rev=true)
        pct = total > 0 ? 100 * seconds / total : 0.0
        count = get(profile.counts, key, 0)
        println(io, "    ", rpad(String(key), 30),
                " time=", round(seconds; digits=6), "s",
                "  pct=", round(pct; digits=2), "%",
                "  count=", count)
    end
    return profile
end

function _amg_fine_dirac_flops_per_apply(setup::DDalphaAMG_Setup)
    return 1320 * setup.NX * setup.NY * setup.NZ * setup.NT
end

@inline _amg_complex_matvec_flops(nrows::Int, ncols::Int) =
    nrows * (6 * ncols + 2 * max(ncols - 1, 0))

function _amg_coarse_stencil_couplings_per_block(setup::DDalphaAMG_Setup)
    dims = (setup.NBX, setup.NBY, setup.NBZ, setup.NBT)
    return 1 + sum(d > 2 ? 2 : 1 for d in dims)
end

function _amg_coarse_operator_flops_per_apply(setup::DDalphaAMG_Setup)
    n = setup.Nv * setup.Nblocks
    if setup.Dc_matrix !== nothing
        return _amg_complex_matvec_flops(n, n)
    end
    couplings = _amg_coarse_stencil_couplings_per_block(setup)
    return setup.Nblocks * couplings * _amg_complex_matvec_flops(setup.Nv, setup.Nv)
end

@inline _amg_count_flops!(counter::Nothing, n::Integer) = nothing
@inline _amg_count_flops!(counter::Ref{Int}, n::Integer) = (counter[] += Int(n); nothing)
@inline _amg_complex_axpby_flops(n::Integer) = 14 * Int(n)
@inline _amg_complex_dot_flops(n::Integer) = 8 * Int(n)
@inline _amg_complex_norm_flops(n::Integer) = 3 * Int(n)
@inline _amg_fine_dofs(x::WF3x1) = length(x.f)
@inline _amg_fine_dofs(setup::DDalphaAMG_Setup) =
    setup.NX * setup.NY * setup.NZ * setup.NT * setup.NC * setup.NG
@inline _amg_coarse_dofs(setup::DDalphaAMG_Setup) = setup.Nv * setup.Nblocks
@inline _amg_restrict_flops(setup::DDalphaAMG_Setup) =
    setup.Nv * _amg_fine_dofs(setup) * 8
@inline _amg_prolongate_flops(setup::DDalphaAMG_Setup) =
    setup.Nv * _amg_fine_dofs(setup) * 8

function _amg_flop_stats(setup::DDalphaAMG_Setup,
                         variable_fine_applications::Integer,
                         variable_coarse_applications::Integer,
                         variable_vector_flops::Integer=0;
                         outer_fine_applications::Integer=0,
                         outer_vector_flops::Integer=0)
    fine_per_apply = _amg_fine_dirac_flops_per_apply(setup)
    coarse_per_apply = _amg_coarse_operator_flops_per_apply(setup)
    fine = Int(variable_fine_applications) * fine_per_apply
    coarse = Int(variable_coarse_applications) * coarse_per_apply
    vector = Int(variable_vector_flops)
    preconditioner_total = fine + coarse + vector
    outer_fine = Int(outer_fine_applications) * fine_per_apply
    outer_vector = Int(outer_vector_flops)
    outer_total = outer_fine + outer_vector
    return (; fine_per_apply, coarse_per_apply,
            fine, coarse, vector, preconditioner_total,
            outer_fine, outer_vector, outer_total,
            total=preconditioner_total + outer_total)
end

function _amg_print_solver_counts(io::IO,
                                  setup::DDalphaAMG_Setup,
                                  total_coarse_gmres::Integer,
                                  coarse_apply_counter::Integer,
                                  fine_apply_counter::Integer,
                                  variable_coarse_apply_counter::Integer,
                                  variable_fine_apply_counter::Integer,
                                  variable_vector_flops::Integer=0,
                                  outer_fine_applications::Integer=0,
                                  outer_vector_flops::Integer=0)
    flops = _amg_flop_stats(setup, variable_fine_apply_counter, variable_coarse_apply_counter,
                            variable_vector_flops;
                            outer_fine_applications=outer_fine_applications,
                            outer_vector_flops=outer_vector_flops)
    println(io, "  Total coarse GMRES calls: $total_coarse_gmres")
    println(io, "  Total coarse operator applications: $coarse_apply_counter")
    println(io, "  Total fine Dirac applications: $fine_apply_counter")
    println(io, "  Variable coarse operator applications: $variable_coarse_apply_counter")
    println(io, "  Variable fine Dirac applications: $variable_fine_apply_counter")
    println(io, "  Outer FGMRES fine Dirac applications: $outer_fine_applications")
    println(io, "  Fine Dirac FLOPs per apply: $(flops.fine_per_apply)")
    println(io, "  Coarse operator FLOPs per apply: $(flops.coarse_per_apply)")
    println(io, "  Estimated variable fine Dirac FLOPs: $(flops.fine)")
    println(io, "  Estimated variable coarse operator FLOPs: $(flops.coarse)")
    println(io, "  Estimated variable vector FLOPs: $(flops.vector)")
    println(io, "  Estimated variable preconditioner FLOPs: $(flops.preconditioner_total)")
    println(io, "  Estimated outer fine Dirac FLOPs: $(flops.outer_fine)")
    println(io, "  Estimated outer FGMRES vector FLOPs: $(flops.outer_vector)")
    println(io, "  Estimated outer FGMRES FLOPs: $(flops.outer_total)")
    println(io, "  Estimated variable total FLOPs: $(flops.total)")
    return flops
end

_amg_print_solver_counts(setup::DDalphaAMG_Setup,
                         total_coarse_gmres::Integer,
                         coarse_apply_counter::Integer,
                         fine_apply_counter::Integer,
                         variable_coarse_apply_counter::Integer,
                         variable_fine_apply_counter::Integer,
                         variable_vector_flops::Integer=0,
                         outer_fine_applications::Integer=0,
                         outer_vector_flops::Integer=0) =
    _amg_print_solver_counts(stdout, setup, total_coarse_gmres,
                             coarse_apply_counter, fine_apply_counter,
                             variable_coarse_apply_counter,
                             variable_fine_apply_counter,
                             variable_vector_flops,
                             outer_fine_applications,
                             outer_vector_flops)


"""
    CoarseStencilOperator

Coarse operator stored in block-stencil form on coarse grid DOFs:
- onsite block: self[:, :, b]
- nearest-neighbor links: plus[:, :, b, mu], minus[:, :, b, mu]  (mu=1..4)

Each block matrix has size Nv x Nv, and total coarse size is Nv*Nblocks.
"""
mutable struct CoarseStencilOperator
    Nv::Int
    Nblocks::Int
    NBX::Int
    NBY::Int
    NBZ::Int
    NBT::Int
    self::AbstractArray{ComplexF64,3}   # (Nv, Nv, Nblocks)
    plus::AbstractArray{ComplexF64,4}   # (Nv, Nv, Nblocks, 4)
    minus::AbstractArray{ComplexF64,4}  # (Nv, Nv, Nblocks, 4)
    nb_plus::AbstractMatrix{<:Integer}  # (Nblocks, 4)
    nb_minus::AbstractMatrix{<:Integer} # (Nblocks, 4)
end


@inline function _block_coords(b::Int, NBX::Int, NBY::Int, NBZ::Int)
    t = (b - 1) ÷ (NBX * NBY * NBZ)
    rem1 = (b - 1) - t * (NBX * NBY * NBZ)
    z = rem1 ÷ (NBX * NBY)
    rem2 = rem1 - z * (NBX * NBY)
    y = rem2 ÷ NBX
    x = rem2 - y * NBX
    return x + 1, y + 1, z + 1, t + 1
end

@inline function _periodic_shift(idx::Int, delta::Int, n::Int)
    return mod1(idx + delta, n)
end

@inline function _neighbor_block(b::Int, mu::Int, sgn::Int, NBX::Int, NBY::Int, NBZ::Int, NBT::Int)
    bx, by, bz, bt = _block_coords(b, NBX, NBY, NBZ)
    if mu == 1
        bx = _periodic_shift(bx, sgn, NBX)
    elseif mu == 2
        by = _periodic_shift(by, sgn, NBY)
    elseif mu == 3
        bz = _periodic_shift(bz, sgn, NBZ)
    else
        bt = _periodic_shift(bt, sgn, NBT)
    end
    return block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
end


function apply_coarse_stencil!(out::CoarseVector, op::CoarseStencilOperator, x::CoarseVector)
    if out.data isa CuArray
        fill!(out.data, zero(ComplexF64))
        threads = max(32, min(256, op.Nv))
        @cuda threads=threads blocks=op.Nblocks _amg_coarse_stencil_kernel!(
            out.data, op.self, op.plus, op.minus, op.nb_plus, op.nb_minus, x.data,
            Int32(op.Nv), Int32(op.Nblocks))
        return out
    end
    Nv = op.Nv
    fill!(out.data, zero(ComplexF64))
    # Coefficients were stored as A[dst, src] while fixing a source block basis vector.
    # Therefore apply is most naturally done as source-based scatter.
    @inbounds for bs in 1:op.Nblocks
        src_base = (bs - 1) * Nv

        # onsite: dst = bs
        dst_base = src_base
        for i in 1:Nv
            acc = zero(ComplexF64)
            for j in 1:Nv
                acc += op.self[i, j, bs] * x.data[src_base + j]
            end
            out.data[dst_base + i] += acc
        end

        # +//-nearest neighbors: dst = nb_plus/minus(bs, mu)
        for mu in 1:4
            bp = op.nb_plus[bs, mu]
            bm = op.nb_minus[bs, mu]
            dstp_base = (bp - 1) * Nv
            dstm_base = (bm - 1) * Nv

            for i in 1:Nv
                accp = zero(ComplexF64)
                accm = zero(ComplexF64)
                for j in 1:Nv
                    accp += op.plus[i, j, bs, mu] * x.data[src_base + j]
                    accm += op.minus[i, j, bs, mu] * x.data[src_base + j]
                end
                if bp == bm
                    # On coarse extent 2 along direction mu, +/- neighbors coincide.
                    # precompute currently stores the full response in both slots in this
                    # degenerate case, so adding both would double-count.
                    out.data[dstp_base + i] += accp
                else
                    out.data[dstp_base + i] += accp
                    out.data[dstm_base + i] += accm
                end
            end
        end
    end
    return out
end

function _amg_coarse_self_inverse_gpu(self::CuArray{ComplexF64,3})
    self_lu = copy(self)
    self_inv = similar(self)
    pivot, factor_info = CUDA.CUBLAS.getrf_strided_batched!(self_lu, true)
    factor_status = Array(factor_info)
    failed = findfirst(!iszero, factor_status)
    failed === nothing || throw(SingularException(failed))

    inverse_info = CUDA.CUBLAS.getri_strided_batched!(self_lu, self_inv, pivot)
    inverse_status = Array(inverse_info)
    failed = findfirst(!iszero, inverse_status)
    failed === nothing || throw(SingularException(failed))
    return self_inv
end

function _amg_coarse_self_inverse(op::CoarseStencilOperator)
    if op.self isa CuArray
        return _amg_coarse_self_inverse_gpu(op.self)
    end
    inv_host = similar(op.self)
    @inbounds for b in 1:op.Nblocks
        inv_host[:, :, b] .= inv(Matrix(view(op.self, :, :, b)))
    end
    return inv_host
end

function _amg_coarse_self_inverse!(setup::DDalphaAMG_Setup, op::CoarseStencilOperator)
    cache = setup._gpu_cache isa Dict ? setup._gpu_cache : Dict{Symbol,Any}()
    key = (objectid(op), objectid(op.self), op.Nv, op.Nblocks,
           op.NBX, op.NBY, op.NBZ, op.NBT, op.self isa CuArray)
    if get(cache, :coarse_self_inv_key, nothing) == key
        return cache[:coarse_self_inv]
    end
    self_inv = _amg_coarse_self_inverse(op)
    cache[:coarse_self_inv_key] = key
    cache[:coarse_self_inv] = self_inv
    setup._gpu_cache = cache
    return self_inv
end

function _amg_can_coarse_odd_even(op::CoarseStencilOperator)
    # Periodic nearest-neighbor odd-even splitting is bipartite only on even extents.
    return iseven(op.NBX) && iseven(op.NBY) && iseven(op.NBZ) && iseven(op.NBT)
end

function _amg_coarse_color_mask!(out::CoarseVector, op::CoarseStencilOperator,
                                 x::CoarseVector, color::Integer)
    fill!(out.data, zero(ComplexF64))
    if out.data isa CuArray
        threads = 256
        @cuda threads=threads blocks=cld(length(out.data), threads) _amg_coarse_color_mask_kernel!(
            out.data, x.data, Int32(op.Nv), Int32(op.Nblocks),
            Int32(op.NBX), Int32(op.NBY), Int32(op.NBZ), Int32(color))
        return out
    end
    Nv = op.Nv
    @inbounds for b in 1:op.Nblocks
        _amg_block_color_from_bid(b, op.NBX, op.NBY, op.NBZ) == color || continue
        src_base = (b - 1) * Nv
        for i in 1:Nv
            out.data[src_base + i] = x.data[src_base + i]
        end
    end
    return out
end

function _amg_coarse_color_diff!(out::CoarseVector, op::CoarseStencilOperator,
                                 x::CoarseVector, y::CoarseVector, color::Integer)
    fill!(out.data, zero(ComplexF64))
    if out.data isa CuArray
        threads = 256
        @cuda threads=threads blocks=cld(length(out.data), threads) _amg_coarse_color_diff_kernel!(
            out.data, x.data, y.data, Int32(op.Nv), Int32(op.Nblocks),
            Int32(op.NBX), Int32(op.NBY), Int32(op.NBZ), Int32(color))
        return out
    end
    Nv = op.Nv
    @inbounds for b in 1:op.Nblocks
        _amg_block_color_from_bid(b, op.NBX, op.NBY, op.NBZ) == color || continue
        src_base = (b - 1) * Nv
        for i in 1:Nv
            out.data[src_base + i] = x.data[src_base + i] - y.data[src_base + i]
        end
    end
    return out
end

function _amg_coarse_blockdiag_solve!(out::CoarseVector, op::CoarseStencilOperator,
                                      self_inv, rhs::CoarseVector, color::Integer)
    fill!(out.data, zero(ComplexF64))
    if out.data isa CuArray
        threads = max(32, min(256, op.Nv))
        @cuda threads=threads blocks=op.Nblocks _amg_coarse_blockdiag_solve_kernel!(
            out.data, self_inv, rhs.data, Int32(op.Nv), Int32(op.Nblocks),
            Int32(op.NBX), Int32(op.NBY), Int32(op.NBZ), Int32(color))
        return out
    end
    Nv = op.Nv
    @inbounds for b in 1:op.Nblocks
        _amg_block_color_from_bid(b, op.NBX, op.NBY, op.NBZ) == color || continue
        base = (b - 1) * Nv
        for i in 1:Nv
            acc = zero(ComplexF64)
            for j in 1:Nv
                acc += self_inv[i, j, b] * rhs.data[base + j]
            end
            out.data[base + i] = acc
        end
    end
    return out
end

function _amg_coarse_odd_even_rhs!(out::CoarseVector, op::CoarseStencilOperator,
                                   self_inv, rhs::CoarseVector,
                                   tmp1::CoarseVector, tmp2::CoarseVector,
                                   tmp3::CoarseVector)
    _amg_coarse_color_mask!(tmp1, op, rhs, 1)
    _amg_coarse_blockdiag_solve!(tmp2, op, self_inv, tmp1, 1)
    apply_coarse_stencil!(tmp3, op, tmp2)
    _amg_coarse_color_diff!(out, op, rhs, tmp3, 0)
    return out
end

function _amg_apply_coarse_odd_even_schur!(out::CoarseVector, op::CoarseStencilOperator,
                                           self_inv, x_even::CoarseVector,
                                           tmp1::CoarseVector, tmp2::CoarseVector,
                                           tmp3::CoarseVector)
    apply_coarse_stencil!(tmp1, op, x_even)
    _amg_coarse_blockdiag_solve!(tmp2, op, self_inv, tmp1, 1)
    apply_coarse_stencil!(tmp3, op, tmp2)
    _amg_coarse_color_diff!(out, op, tmp1, tmp3, 0)
    return out
end

function _amg_coarse_odd_even_reconstruct!(out::CoarseVector, op::CoarseStencilOperator,
                                           self_inv, x_even::CoarseVector,
                                           rhs::CoarseVector, tmp1::CoarseVector,
                                           tmp2::CoarseVector, tmp3::CoarseVector)
    apply_coarse_stencil!(tmp1, op, x_even)
    _amg_coarse_color_diff!(tmp2, op, rhs, tmp1, 1)
    _amg_coarse_blockdiag_solve!(tmp3, op, self_inv, tmp2, 1)
    out.data .= x_even.data
    coarse_axpby!(1.0, tmp3, 1.0, out)
    return out
end


function ensure_coarse_gmres_cache!(setup::DDalphaAMG_Setup, template::CoarseVector, restart::Int)
    m = min(restart, length(template.data))
    if setup._gmres_restart != m || length(setup._gmres_V) != m + 1
        setup._gmres_V = [similar(template) for _ in 1:(m + 1)]
        setup._gmres_w = similar(template)
        setup._gmres_r = similar(template)
        setup._gmres_H = zeros(ComplexF64, m + 1, m)
        setup._gmres_e1 = zeros(ComplexF64, m + 1)
        setup._gmres_restart = m
    end
    return m
end

function _amg_gpu_batched_orthogonalize!(w::CuArray{ComplexF64,1},
                                         basis::CuArray{ComplexF64,2},
                                         j::Int,
                                         coeff_dev::CuArray{ComplexF64,1},
                                         coeff_host::Vector{ComplexF64})
    1 <= j <= size(basis, 2) || throw(ArgumentError("invalid Arnoldi column count: $j"))
    length(coeff_dev) >= 2j || throw(ArgumentError("device coefficient buffer is too small"))
    length(coeff_host) >= j || throw(ArgumentError("host coefficient buffer is too small"))

    basis_j = view(basis, :, 1:j)
    coeff_total = view(coeff_dev, 1:j)
    coeff_pass2 = view(coeff_dev, j+1:2j)

    mul!(coeff_total, adjoint(basis_j), w)
    mul!(w, basis_j, coeff_total, ComplexF64(-1), ComplexF64(1))
    mul!(coeff_pass2, adjoint(basis_j), w)
    mul!(w, basis_j, coeff_pass2, ComplexF64(-1), ComplexF64(1))
    coeff_total .+= coeff_pass2
    copyto!(coeff_host, 1, coeff_dev, 1, j)
    return norm(w)
end


# ==========================================
#  Block utilities
# ==========================================

@inline function block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
    return (bt-1)*NBZ*NBY*NBX + (bz-1)*NBY*NBX + (by-1)*NBX + bx
end

function _amg_build_boundary8_affected_sites(bmap::Vector{Int32}, blockinfo, setup)
    schedule_code = _amg_sap_schedule_code(:boundary8)
    affected = [Int32[] for _ in 0:7]
    blocksize = Int(blockinfo.blocksize)

    @inbounds for site in eachindex(bmap)
        b = (site - 1) % blocksize + 1
        r = (site - 1) ÷ blocksize + 1
        neighboring_bids = Int32[bmap[site]]
        for mu in Int32(1):Int32(4)
            bp, rp = _amg_shift_site(b, r, mu, Int32(1), blockinfo)
            bm, rm = _amg_shift_site(b, r, mu, -Int32(1), blockinfo)
            push!(neighboring_bids, bmap[(rp - 1) * blocksize + bp])
            push!(neighboring_bids, bmap[(rm - 1) * blocksize + bm])
        end

        for step in 0:7
            if any(bid -> _amg_sap_block_active_from_bid(
                       bid, step, setup.NBX, setup.NBY, setup.NBZ, setup.NBT,
                       schedule_code), neighboring_bids)
                push!(affected[step + 1], Int32(site))
            end
        end
    end

    return CuArray.(affected)
end

function _ensure_gpu_cache!(setup)
    setup.test_vectors[1] isa WF3x1GPU || return nothing
    cache = hasproperty(setup, :_gpu_cache) ? setup._gpu_cache : nothing
    if cache isa Dict && get(cache, :r, nothing) == setup.r
        if !haskey(cache, :boundary8_affected_sites)
            cache[:boundary8_affected_sites] = _amg_build_boundary8_affected_sites(
                Array(cache[:bmap]), setup.test_vectors[1].blockinfo, setup)
        end
        return cache
    end

    x = setup.test_vectors[1]
    bi = x.blockinfo
    blocksize = bi.blocksize
    rsize = bi.rsize
    nsites = blocksize * rsize
    bmap = Vector{Int32}(undef, nsites)

    idx = 0
    @inbounds for r_idx = 1:rsize
        for b_idx = 1:blocksize
            idx += 1
            ix = 1 + mod(div(b_idx-1, bi.blocks_s[1]), bi.blocks[1]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[1]), bi.blocknumbers[1]) * bi.blocks[1]
            iy = 1 + mod(div(b_idx-1, bi.blocks_s[2]), bi.blocks[2]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[2]), bi.blocknumbers[2]) * bi.blocks[2]
            iz = 1 + mod(div(b_idx-1, bi.blocks_s[3]), bi.blocks[3]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[3]), bi.blocknumbers[3]) * bi.blocks[3]
            it = 1 + mod(div(b_idx-1, bi.blocks_s[4]), bi.blocks[4]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[4]), bi.blocknumbers[4]) * bi.blocks[4]

            bx = (ix - 1) ÷ setup.r + 1
            by = (iy - 1) ÷ setup.r + 1
            bz = (iz - 1) ÷ setup.r + 1
            bt = (it - 1) ÷ setup.r + 1
            bmap[idx] = Int32(block_linear_index(bx, by, bz, bt, setup.NBX, setup.NBY, setup.NBZ))
        end
    end

    counts = zeros(Int32, setup.Nblocks)
    @inbounds for bid in bmap
        if bid < 1 || bid > setup.Nblocks
            error("Invalid AMG GPU block map entry: bid=$bid, expected 1:$(setup.Nblocks)")
        end
        counts[Int(bid)] += Int32(1)
    end
    max_count = maximum(counts)
    sites = zeros(Int32, Int(max_count), setup.Nblocks)
    pos = zeros(Int32, setup.Nblocks)
    @inbounds for site in eachindex(bmap)
        bid = Int(bmap[site])
        pos[bid] += Int32(1)
        sites[Int(pos[bid]), bid] = Int32(site)
    end

    cache = Dict{Symbol,Any}(
        :r => setup.r,
        :bmap => CuArray(bmap),
        :block_sites => CuArray(sites),
        :block_counts => CuArray(counts),
        :max_count => Int(max_count),
        :boundary8_affected_sites => _amg_build_boundary8_affected_sites(bmap, bi, setup),
    )
    if hasproperty(setup, :_gpu_cache)
        setup._gpu_cache = cache
    end
    return cache
end

function _ddamg_fill_random_kernel!(x)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(x)
        x[idx] = ComplexF64(0.0)
    end
    return nothing
end

function random_fermion!(x::WF3x1GPU)
    x.f .= CUDA.randn(Float64, size(x.f)) .+ im .* CUDA.randn(Float64, size(x.f))
    return x
end

function _amg_restrict_kernel!(out, tv, x, block_sites, block_counts, nc_ng::Int32, Nv::Int32)
    bid = Int32(blockIdx().x)
    v = Int32(blockIdx().y)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))
    total = block_counts[bid] * nc_ng
    acc = ComplexF64(0.0)
    p = tid
    while p <= total
        site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
        comp = (p - Int32(1)) % nc_ng + Int32(1)
        site = block_sites[site_pos, bid]
        fidx = (site - Int32(1)) * nc_ng + comp
        @inbounds acc += conj(tv[fidx]) * x[fidx]
        p += bsz
    end
    sh[tid] = acc
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        out[(bid - Int32(1)) * Nv + v] = sh[Int32(1)]
    end
    return nothing
end

function _amg_restrict_one_kernel!(out, tv, x, block_sites, block_counts,
                                   nc::Int32, nc_ng::Int32, Nv::Int32, v::Int32)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    out_len = Int32(length(out))
    field_len = Int32(length(x))
    nsites = field_len ÷ nc_ng
    max_count = Int32(size(block_sites, 1))
    nblocks = Int32(size(block_sites, 2))
    if bid > nblocks || v > Nv
        return nothing
    end
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))
    total = block_counts[bid] * nc_ng
    acc = ComplexF64(0.0)
    p = tid
    while p <= total
        site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
        comp = (p - Int32(1)) % nc_ng + Int32(1)
        if site_pos <= max_count
            site = block_sites[site_pos, bid]
            fidx = (site - Int32(1)) * nc_ng + comp
            if site >= Int32(1) && site <= nsites && fidx >= Int32(1) && fidx <= field_len
                @inbounds acc += conj(tv[fidx]) * x[fidx]
            end
        end
        p += bsz
    end
    sh[tid] = acc
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        out_idx = (bid - Int32(1)) * Nv + v
        if out_idx >= Int32(1) && out_idx <= out_len
            out[out_idx] = sh[Int32(1)]
        end
    end
    return nothing
end

function _amg_prolong_add_kernel!(out, tv, cv, bmap, nc_ng::Int32, total::Int32, Nv::Int32, v::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        nsites = Int32(length(bmap))
        cv_len = Int32(length(cv))
        if site >= Int32(1) && site <= nsites
            bid = bmap[site]
            cv_idx = (bid - Int32(1)) * Nv + v
            if cv_idx >= Int32(1) && cv_idx <= cv_len
                @inbounds out[idx] += cv[cv_idx] * tv[idx]
            end
        end
    end
    return nothing
end

function _amg_block_project_kernel!(x, bmap, nc_ng::Int32, total::Int32, bid_target::Int32, keep_inside::Bool, src)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        inside = bmap[site] == bid_target
        if keep_inside
            if inside
                x[idx] = src === nothing ? ComplexF64(0.0) : src[idx]
            end
        else
            if !inside
                x[idx] = ComplexF64(0.0)
            end
        end
    end
    return nothing
end

@inline function _amg_fine_site_parity(b, r, blockinfo)
    ix = _amg_cvt_x(b, blockinfo.blocks[1], blockinfo.blocks_s[1]) +
         _amg_cvt_x(r, blockinfo.blocknumbers[1], blockinfo.blocknumbers_s[1]) *
         blockinfo.blocks[1]
    iy = _amg_cvt_x(b, blockinfo.blocks[2], blockinfo.blocks_s[2]) +
         _amg_cvt_x(r, blockinfo.blocknumbers[2], blockinfo.blocknumbers_s[2]) *
         blockinfo.blocks[2]
    iz = _amg_cvt_x(b, blockinfo.blocks[3], blockinfo.blocks_s[3]) +
         _amg_cvt_x(r, blockinfo.blocknumbers[3], blockinfo.blocknumbers_s[3]) *
         blockinfo.blocks[3]
    it = _amg_cvt_x(b, blockinfo.blocks[4], blockinfo.blocks_s[4]) +
         _amg_cvt_x(r, blockinfo.blocknumbers[4], blockinfo.blocknumbers_s[4]) *
         blockinfo.blocks[4]
    return (ix + iy + iz + it) & 1
end

function _amg_block_parity_project_kernel!(x, bmap, nc_ng::Int32, total::Int32,
                                           bid_target::Int32, parity::Int32, blockinfo)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
        r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))
        keep = bmap[site] == bid_target && Int32(_amg_fine_site_parity(b, r, blockinfo)) == parity
        if !keep
            x[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_block_axpby_kernel!(y, x, bmap, nc_ng::Int32, total::Int32, bid_target::Int32, α, β)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        if bmap[site] == bid_target
            @inbounds y[idx] = α * x[idx] + β * y[idx]
        end
    end
    return nothing
end

function _amg_block_inner_kernel!(out, x, y, block_sites, block_counts, nc_ng::Int32, bid::Int32)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))
    total = block_counts[bid] * nc_ng
    acc = ComplexF64(0.0)
    p = tid
    while p <= total
        site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
        comp = (p - Int32(1)) % nc_ng + Int32(1)
        site = block_sites[site_pos, bid]
        fidx = (site - Int32(1)) * nc_ng + comp
        @inbounds acc += conj(x[fidx]) * y[fidx]
        p += bsz
    end
    sh[tid] = acc
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        out[1] = sh[Int32(1)]
    end
    return nothing
end

function _amg_orthogonalize_kernel!(v, u, block_sites, block_counts, coeffs, nc_ng::Int32,
                                    normalize::Bool)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, 2Int(bsz))
    total = block_counts[bid] * nc_ng
    acc = ComplexF64(0.0)
    den = 0.0
    p = tid
    while p <= total
        site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
        comp = (p - Int32(1)) % nc_ng + Int32(1)
        site = block_sites[site_pos, bid]
        fidx = (site - Int32(1)) * nc_ng + comp
        @inbounds begin
            acc += conj(u[fidx]) * v[fidx]
            den += abs2(u[fidx])
        end
        p += bsz
    end
    sh[tid] = acc
    sh[tid + bsz] = ComplexF64(den)
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh[tid] += sh[tid + half]
            sh[tid + bsz] += sh[tid + bsz + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    den_block = real(sh[Int32(1) + bsz])
    coeff = normalize || den_block <= 1e-28 ? sh[Int32(1)] : sh[Int32(1)] / den_block
    if tid == Int32(1)
        coeffs[bid] = coeff
    end
    sync_threads()
    p = tid
    while p <= total
        site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
        comp = (p - Int32(1)) % nc_ng + Int32(1)
        site = block_sites[site_pos, bid]
        fidx = (site - Int32(1)) * nc_ng + comp
        @inbounds v[fidx] -= coeff * u[fidx]
        p += bsz
    end
    return nothing
end

function _amg_normalize_kernel!(v, block_sites, block_counts, norms, nc_ng::Int32)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(Float64, Int(bsz))
    total = block_counts[bid] * nc_ng
    acc = 0.0
    p = tid
    while p <= total
        site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
        comp = (p - Int32(1)) % nc_ng + Int32(1)
        site = block_sites[site_pos, bid]
        fidx = (site - Int32(1)) * nc_ng + comp
        @inbounds acc += abs2(v[fidx])
        p += bsz
    end
    sh[tid] = acc
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    nrm = sqrt(sh[Int32(1)])
    if tid == Int32(1)
        norms[bid] = nrm
    end
    sync_threads()
    if nrm > 1e-14
        p = tid
        while p <= total
            site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
            comp = (p - Int32(1)) % nc_ng + Int32(1)
            site = block_sites[site_pos, bid]
            fidx = (site - Int32(1)) * nc_ng + comp
            @inbounds v[fidx] /= nrm
            p += bsz
        end
    end
    return nothing
end

function _amg_set_basis_kernel!(cv, j::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= length(cv)
        cv[idx] = idx == j ? ComplexF64(1.0) : ComplexF64(0.0)
    end
    return nothing
end

function _amg_copy_stencil_column_kernel!(self, plus, minus, result, nb_plus, nb_minus,
                                          Nv::Int32, Nblocks::Int32, bs::Int32, v::Int32)
    i = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if i <= Nv
        self[i, v, bs] = result[(bs - Int32(1)) * Nv + i]
        for mu in Int32(1):Int32(4)
            bp = nb_plus[bs, mu]
            bm = nb_minus[bs, mu]
            plus[i, v, bs, mu] = result[(bp - Int32(1)) * Nv + i]
            minus[i, v, bs, mu] = result[(bm - Int32(1)) * Nv + i]
        end
    end
    return nothing
end

function _amg_coarse_stencil_kernel!(out, self, plus, minus, nb_plus, nb_minus, x,
                                     Nv::Int32, Nblocks::Int32)
    bid = Int32(blockIdx().x)
    i = Int32(threadIdx().x)
    if bid <= Nblocks && i <= Nv
        dst = (bid - Int32(1)) * Nv + i
        acc = ComplexF64(0.0)
        for j in Int32(1):Nv
            acc += self[i, j, bid] * x[(bid - Int32(1)) * Nv + j]
        end
        for mu in Int32(1):Int32(4)
            sm = nb_minus[bid, mu]
            sp = nb_plus[bid, mu]
            for j in Int32(1):Nv
                acc += plus[i, j, sm, mu] * x[(sm - Int32(1)) * Nv + j]
                if sp != sm
                    acc += minus[i, j, sp, mu] * x[(sp - Int32(1)) * Nv + j]
                end
            end
        end
        out[dst] = acc
    end
    return nothing
end

function _amg_coarse_color_mask_kernel!(out, x, Nv::Int32, Nblocks::Int32,
                                        NBX::Int32, NBY::Int32, NBZ::Int32,
                                        color::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= length(out)
        bid = (idx - Int32(1)) ÷ Nv + Int32(1)
        if bid <= Nblocks && Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color
            out[idx] = x[idx]
        else
            out[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_coarse_color_diff_kernel!(out, x, y, Nv::Int32, Nblocks::Int32,
                                        NBX::Int32, NBY::Int32, NBZ::Int32,
                                        color::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= length(out)
        bid = (idx - Int32(1)) ÷ Nv + Int32(1)
        if bid <= Nblocks && Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color
            out[idx] = x[idx] - y[idx]
        else
            out[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_coarse_blockdiag_solve_kernel!(out, self_inv, rhs, Nv::Int32, Nblocks::Int32,
                                             NBX::Int32, NBY::Int32, NBZ::Int32,
                                             color::Int32)
    bid = Int32(blockIdx().x)
    i = Int32(threadIdx().x)
    if bid <= Nblocks && i <= Nv
        dst = (bid - Int32(1)) * Nv + i
        if Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color
            acc = zero(ComplexF64)
            for j in Int32(1):Nv
                acc += self_inv[i, j, bid] * rhs[(bid - Int32(1)) * Nv + j]
            end
            out[dst] = acc
        else
            out[dst] = zero(ComplexF64)
        end
    end
    return nothing
end

@inline _amg_cvt_x(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)

@inline function _amg_si_plus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bs = b
    rs = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == blocks - 1
        bs = bs - (blocks - 1) * blocks_s
        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == blocknumbers - 1
            rs = rs - (blocknumbers - 1) * blocknumbers_s
        else
            rs += blocknumbers_s
        end
    else
        bs += blocks_s
    end
    return bs, rs
end

@inline function _amg_si_minus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bs = b
    rs = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == 0
        bs = bs + (blocks - 1) * blocks_s
        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == 0
            rs = rs + (blocknumbers - 1) * blocknumbers_s
        else
            rs -= blocknumbers_s
        end
    else
        bs -= blocks_s
    end
    return bs, rs
end

@inline function _amg_shift_site(b, r, mu::Int32, sgn::Int32, blockinfo)
    blocks = blockinfo.blocks[mu]
    blocks_s = blockinfo.blocks_s[mu]
    blocknumbers = blockinfo.blocknumbers[mu]
    blocknumbers_s = blockinfo.blocknumbers_s[mu]
    if sgn > 0
        return _amg_si_plus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    else
        return _amg_si_minus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    end
end

@inline function _amg_bc_factor_site(b, r, mu::Int32, sgn::Int32, blockinfo)
    if mu != Int32(4)
        return 1.0
    end
    it = 1 + _amg_cvt_x(b, blockinfo.blocks[4], blockinfo.blocks_s[4]) +
             _amg_cvt_x(r, blockinfo.blocknumbers[4], blockinfo.blocknumbers_s[4]) *
             blockinfo.blocks[4]
    nt = blockinfo.blocks[4] * blockinfo.blocknumbers[4]
    crosses = (sgn > 0 && it == nt) || (sgn < 0 && it == 1)
    return crosses ? -1.0 : 1.0
end

@inline _amg_proj_m1(s1,s2,s3,s4) = (s1 + im*s4, s2 + im*s3, s3 - im*s2, s4 - im*s1)
@inline _amg_proj_p1(s1,s2,s3,s4) = (s1 - im*s4, s2 - im*s3, s3 + im*s2, s4 + im*s1)
@inline _amg_proj_m2(s1,s2,s3,s4) = (s1 + s4, s2 - s3, s3 - s2, s4 + s1)
@inline _amg_proj_p2(s1,s2,s3,s4) = (s1 - s4, s2 + s3, s3 + s2, s4 - s1)
@inline _amg_proj_m3(s1,s2,s3,s4) = (s1 + im*s3, s2 - im*s4, s3 - im*s1, s4 + im*s2)
@inline _amg_proj_p3(s1,s2,s3,s4) = (s1 - im*s3, s2 + im*s4, s3 + im*s1, s4 - im*s2)
@inline _amg_proj_m4(s1,s2,s3,s4) = (s1 + s3, s2 + s4, s3 + s1, s4 + s2)
@inline _amg_proj_p4(s1,s2,s3,s4) = (s1 - s3, s2 - s4, s3 - s1, s4 - s2)

@inline function _amg_project_minus(mu::Int32, s1,s2,s3,s4)
    if mu == Int32(1)
        return _amg_proj_m1(s1,s2,s3,s4)
    elseif mu == Int32(2)
        return _amg_proj_m2(s1,s2,s3,s4)
    elseif mu == Int32(3)
        return _amg_proj_m3(s1,s2,s3,s4)
    else
        return _amg_proj_m4(s1,s2,s3,s4)
    end
end

@inline function _amg_project_plus(mu::Int32, s1,s2,s3,s4)
    if mu == Int32(1)
        return _amg_proj_p1(s1,s2,s3,s4)
    elseif mu == Int32(2)
        return _amg_proj_p2(s1,s2,s3,s4)
    elseif mu == Int32(3)
        return _amg_proj_p3(s1,s2,s3,s4)
    else
        return _amg_proj_p4(s1,s2,s3,s4)
    end
end

function _amg_pack_test_vectors_kernel!(packed, src, v::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    n = Int32(length(src))
    if idx <= n
        packed[idx + (v - Int32(1)) * n] = src[idx]
    end
    return nothing
end

function _amg_direct_wilson_stencil_kernel!(self, plus, minus, packed, U1, U2, U3, U4,
                                            bmap, block_sites, block_counts, nb_plus, nb_minus,
                                            κ::Float64, blockinfo, Nv::Int32)
    bs_amg = Int32(blockIdx().x)
    vi = Int32(blockIdx().y)
    vj = Int32(blockIdx().z)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)

    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz) * 9)
    sh_self = sh
    sh_p1 = @view sh[Int(bsz)+1:2*Int(bsz)]
    sh_p2 = @view sh[2*Int(bsz)+1:3*Int(bsz)]
    sh_p3 = @view sh[3*Int(bsz)+1:4*Int(bsz)]
    sh_p4 = @view sh[4*Int(bsz)+1:5*Int(bsz)]
    sh_m1 = @view sh[5*Int(bsz)+1:6*Int(bsz)]
    sh_m2 = @view sh[6*Int(bsz)+1:7*Int(bsz)]
    sh_m3 = @view sh[7*Int(bsz)+1:8*Int(bsz)]
    sh_m4 = @view sh[8*Int(bsz)+1:9*Int(bsz)]

    accs = zero(ComplexF64)
    accp1 = zero(ComplexF64); accp2 = zero(ComplexF64); accp3 = zero(ComplexF64); accp4 = zero(ComplexF64)
    accm1 = zero(ComplexF64); accm2 = zero(ComplexF64); accm3 = zero(ComplexF64); accm4 = zero(ComplexF64)

    total = block_counts[bs_amg]
    p = tid
    while p <= total
        site = block_sites[p, bs_amg]
        b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
        r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))

        @inbounds begin
            x11 = packed[1,1,b,r,vj]; x21 = packed[2,1,b,r,vj]; x31 = packed[3,1,b,r,vj]
            x12 = packed[1,2,b,r,vj]; x22 = packed[2,2,b,r,vj]; x32 = packed[3,2,b,r,vj]
            x13 = packed[1,3,b,r,vj]; x23 = packed[2,3,b,r,vj]; x33 = packed[3,3,b,r,vj]
            x14 = packed[1,4,b,r,vj]; x24 = packed[2,4,b,r,vj]; x34 = packed[3,4,b,r,vj]
        end

        @inbounds begin
            accs += conj(packed[1,1,b,r,vi]) * x11 + conj(packed[2,1,b,r,vi]) * x21 + conj(packed[3,1,b,r,vi]) * x31
            accs += conj(packed[1,2,b,r,vi]) * x12 + conj(packed[2,2,b,r,vi]) * x22 + conj(packed[3,2,b,r,vi]) * x32
            accs += conj(packed[1,3,b,r,vi]) * x13 + conj(packed[2,3,b,r,vi]) * x23 + conj(packed[3,3,b,r,vi]) * x33
            accs += conj(packed[1,4,b,r,vi]) * x14 + conj(packed[2,4,b,r,vi]) * x24 + conj(packed[3,4,b,r,vi]) * x34
        end

        for mu in Int32(1):Int32(4)
            U = mu == Int32(1) ? U1 : (mu == Int32(2) ? U2 : (mu == Int32(3) ? U3 : U4))

            # Source site q contributes to y(q-mu) through the forward hop of y(q-mu).
            bm_b, bm_r = _amg_shift_site(b, r, mu, -Int32(1), blockinfo)
            facm = _amg_bc_factor_site(bm_b, bm_r, mu, Int32(1), blockinfo)
            @inbounds begin
                u11=U[1,1,bm_b,bm_r]; u12=U[1,2,bm_b,bm_r]; u13=U[1,3,bm_b,bm_r]
                u21=U[2,1,bm_b,bm_r]; u22=U[2,2,bm_b,bm_r]; u23=U[2,3,bm_b,bm_r]
                u31=U[3,1,bm_b,bm_r]; u32=U[3,2,bm_b,bm_r]; u33=U[3,3,bm_b,bm_r]
            end
            t11 = u11*x11 + u12*x21 + u13*x31; t21 = u21*x11 + u22*x21 + u23*x31; t31 = u31*x11 + u32*x21 + u33*x31
            t12 = u11*x12 + u12*x22 + u13*x32; t22 = u21*x12 + u22*x22 + u23*x32; t32 = u31*x12 + u32*x22 + u33*x32
            t13 = u11*x13 + u12*x23 + u13*x33; t23 = u21*x13 + u22*x23 + u23*x33; t33 = u31*x13 + u32*x23 + u33*x33
            t14 = u11*x14 + u12*x24 + u13*x34; t24 = u21*x14 + u22*x24 + u23*x34; t34 = u31*x14 + u32*x24 + u33*x34

            p11,p12,p13,p14 = _amg_project_minus(mu, t11,t12,t13,t14)
            p21,p22,p23,p24 = _amg_project_minus(mu, t21,t22,t23,t24)
            p31,p32,p33,p34 = _amg_project_minus(mu, t31,t32,t33,t34)
            cm = -κ * facm
            destm = bmap[(Int32(bm_r) - Int32(1)) * Int32(blockinfo.blocksize) + Int32(bm_b)]
            valm = zero(ComplexF64)
            @inbounds begin
                valm += conj(packed[1,1,bm_b,bm_r,vi]) * (cm*p11) + conj(packed[2,1,bm_b,bm_r,vi]) * (cm*p21) + conj(packed[3,1,bm_b,bm_r,vi]) * (cm*p31)
                valm += conj(packed[1,2,bm_b,bm_r,vi]) * (cm*p12) + conj(packed[2,2,bm_b,bm_r,vi]) * (cm*p22) + conj(packed[3,2,bm_b,bm_r,vi]) * (cm*p32)
                valm += conj(packed[1,3,bm_b,bm_r,vi]) * (cm*p13) + conj(packed[2,3,bm_b,bm_r,vi]) * (cm*p23) + conj(packed[3,3,bm_b,bm_r,vi]) * (cm*p33)
                valm += conj(packed[1,4,bm_b,bm_r,vi]) * (cm*p14) + conj(packed[2,4,bm_b,bm_r,vi]) * (cm*p24) + conj(packed[3,4,bm_b,bm_r,vi]) * (cm*p34)
            end
            if destm == bs_amg
                accs += valm
            elseif destm == nb_minus[bs_amg, mu]
                if mu == Int32(1)
                    accm1 += valm
                elseif mu == Int32(2)
                    accm2 += valm
                elseif mu == Int32(3)
                    accm3 += valm
                else
                    accm4 += valm
                end
            end

            # Source site q contributes to y(q+mu) through the backward hop of y(q+mu).
            bp_b, bp_r = _amg_shift_site(b, r, mu, Int32(1), blockinfo)
            facp = _amg_bc_factor_site(bp_b, bp_r, mu, -Int32(1), blockinfo)
            @inbounds begin
                v11=conj(U[1,1,b,r]); v12=conj(U[2,1,b,r]); v13=conj(U[3,1,b,r])
                v21=conj(U[1,2,b,r]); v22=conj(U[2,2,b,r]); v23=conj(U[3,2,b,r])
                v31=conj(U[1,3,b,r]); v32=conj(U[2,3,b,r]); v33=conj(U[3,3,b,r])
            end
            q11 = v11*x11 + v12*x21 + v13*x31; q21 = v21*x11 + v22*x21 + v23*x31; q31 = v31*x11 + v32*x21 + v33*x31
            q12 = v11*x12 + v12*x22 + v13*x32; q22 = v21*x12 + v22*x22 + v23*x32; q32 = v31*x12 + v32*x22 + v33*x32
            q13 = v11*x13 + v12*x23 + v13*x33; q23 = v21*x13 + v22*x23 + v23*x33; q33 = v31*x13 + v32*x23 + v33*x33
            q14 = v11*x14 + v12*x24 + v13*x34; q24 = v21*x14 + v22*x24 + v23*x34; q34 = v31*x14 + v32*x24 + v33*x34

            r11,r12,r13,r14 = _amg_project_plus(mu, q11,q12,q13,q14)
            r21,r22,r23,r24 = _amg_project_plus(mu, q21,q22,q23,q24)
            r31,r32,r33,r34 = _amg_project_plus(mu, q31,q32,q33,q34)
            cp = -κ * facp
            destp = bmap[(Int32(bp_r) - Int32(1)) * Int32(blockinfo.blocksize) + Int32(bp_b)]
            valp = zero(ComplexF64)
            @inbounds begin
                valp += conj(packed[1,1,bp_b,bp_r,vi]) * (cp*r11) + conj(packed[2,1,bp_b,bp_r,vi]) * (cp*r21) + conj(packed[3,1,bp_b,bp_r,vi]) * (cp*r31)
                valp += conj(packed[1,2,bp_b,bp_r,vi]) * (cp*r12) + conj(packed[2,2,bp_b,bp_r,vi]) * (cp*r22) + conj(packed[3,2,bp_b,bp_r,vi]) * (cp*r32)
                valp += conj(packed[1,3,bp_b,bp_r,vi]) * (cp*r13) + conj(packed[2,3,bp_b,bp_r,vi]) * (cp*r23) + conj(packed[3,3,bp_b,bp_r,vi]) * (cp*r33)
                valp += conj(packed[1,4,bp_b,bp_r,vi]) * (cp*r14) + conj(packed[2,4,bp_b,bp_r,vi]) * (cp*r24) + conj(packed[3,4,bp_b,bp_r,vi]) * (cp*r34)
            end
            if destp == bs_amg
                accs += valp
            elseif destp == nb_plus[bs_amg, mu]
                if mu == Int32(1)
                    accp1 += valp
                elseif mu == Int32(2)
                    accp2 += valp
                elseif mu == Int32(3)
                    accp3 += valp
                else
                    accp4 += valp
                end
            end
        end

        p += bsz
    end

    sh_self[tid] = accs
    sh_p1[tid] = accp1; sh_p2[tid] = accp2; sh_p3[tid] = accp3; sh_p4[tid] = accp4
    sh_m1[tid] = accm1; sh_m2[tid] = accm2; sh_m3[tid] = accm3; sh_m4[tid] = accm4
    sync_threads()

    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh_self[tid] += sh_self[tid + half]
            sh_p1[tid] += sh_p1[tid + half]; sh_p2[tid] += sh_p2[tid + half]
            sh_p3[tid] += sh_p3[tid + half]; sh_p4[tid] += sh_p4[tid + half]
            sh_m1[tid] += sh_m1[tid + half]; sh_m2[tid] += sh_m2[tid + half]
            sh_m3[tid] += sh_m3[tid + half]; sh_m4[tid] += sh_m4[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end

    if tid == Int32(1)
        self[vi, vj, bs_amg] = sh_self[1]
        plus[vi, vj, bs_amg, 1] = sh_p1[1]; plus[vi, vj, bs_amg, 2] = sh_p2[1]
        plus[vi, vj, bs_amg, 3] = sh_p3[1]; plus[vi, vj, bs_amg, 4] = sh_p4[1]
        minus[vi, vj, bs_amg, 1] = sh_m1[1]; minus[vi, vj, bs_amg, 2] = sh_m2[1]
        minus[vi, vj, bs_amg, 3] = sh_m3[1]; minus[vi, vj, bs_amg, 4] = sh_m4[1]
    end
    return nothing
end

@inline function _amg_block_color_from_bid(bid::Integer, NBX::Integer, NBY::Integer, NBZ::Integer)
    b0 = bid - 1
    bx = b0 % NBX + 1
    by = (b0 ÷ NBX) % NBY + 1
    bz = (b0 ÷ (NBX * NBY)) % NBZ + 1
    bt = b0 ÷ (NBX * NBY * NBZ) + 1
    return (bx + by + bz + bt) & 1
end

@inline _amg_sap_schedule_code(::Val{:twocolor}) = Int32(0)
@inline _amg_sap_schedule_code(::Val{:boundary8}) = Int32(1)
@inline function _amg_sap_schedule_code(schedule::Symbol)
    schedule == :twocolor && return Int32(0)
    schedule == :boundary8 && return Int32(1)
    throw(ArgumentError("sap_schedule must be :boundary8 or :twocolor, got $schedule"))
end

@inline function _amg_sap_schedule_steps(schedule::Symbol)
    schedule == :twocolor && return 0:1
    schedule == :boundary8 && return 0:7
    throw(ArgumentError("sap_schedule must be :boundary8 or :twocolor, got $schedule"))
end

@inline function _amg_sap_color(bx::Integer, by::Integer, bz::Integer, bt::Integer)
    return (bx + by + bz + bt) & 1
end

@inline function _amg_sap_block_step_code(bx::Integer, by::Integer, bz::Integer, bt::Integer,
                                          NBX::Integer, NBY::Integer, NBZ::Integer, NBT::Integer,
                                          schedule_code::Integer)
    color = _amg_sap_color(bx, by, bz, bt)
    schedule_code == 0 && return color

    count_plus = (bx == NBX ? 1 : 0) + (by == NBY ? 1 : 0) +
                 (bz == NBZ ? 1 : 0) + (bt == NBT ? 1 : 0)
    count_minus = (bx == 1 ? 1 : 0) + (by == 1 ? 1 : 0) +
                  (bz == 1 ? 1 : 0) + (bt == 1 ? 1 : 0)
    count_inner = (1 < bx < NBX ? 1 : 0) + (1 < by < NBY ? 1 : 0) +
                  (1 < bz < NBZ ? 1 : 0) + (1 < bt < NBT ? 1 : 0)

    if count_inner == 4
        return 4 * color
    elseif count_minus == 0
        return color == 0 ? 1 : 7
    elseif count_plus == 0
        return color == 0 ? 3 : 5
    else
        return 2 + 4 * color
    end
end

@inline function _amg_sap_block_step(bx::Integer, by::Integer, bz::Integer, bt::Integer,
                                     NBX::Integer, NBY::Integer, NBZ::Integer, NBT::Integer,
                                     schedule::Symbol)
    return _amg_sap_block_step_code(bx, by, bz, bt, NBX, NBY, NBZ, NBT,
                                    _amg_sap_schedule_code(schedule))
end

@inline function _amg_sap_block_step_from_bid(bid::Integer, NBX::Integer, NBY::Integer,
                                             NBZ::Integer, NBT::Integer,
                                             schedule_code::Integer)
    b0 = bid - 1
    bx = b0 % NBX + 1
    by = (b0 ÷ NBX) % NBY + 1
    bz = (b0 ÷ (NBX * NBY)) % NBZ + 1
    bt = b0 ÷ (NBX * NBY * NBZ) + 1
    return _amg_sap_block_step_code(bx, by, bz, bt, NBX, NBY, NBZ, NBT, schedule_code)
end

@inline function _amg_sap_block_active_from_bid(bid::Integer, step::Integer,
                                               NBX::Integer, NBY::Integer,
                                               NBZ::Integer, NBT::Integer,
                                               schedule_code::Integer)
    return _amg_sap_block_step_from_bid(bid, NBX, NBY, NBZ, NBT, schedule_code) == step
end

function _amg_init_color_fields_kernel!(delta, local_r, res, bmap,
                                        nc_ng::Int32, total::Int32, step::Int32,
                                        NBX::Int32, NBY::Int32, NBZ::Int32,
                                        NBT::Int32, schedule_code::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        use = _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        delta[idx] = zero(ComplexF64)
        local_r[idx] = use ? res[idx] : zero(ComplexF64)
    end
    return nothing
end

function _amg_add_color_field_kernel!(x, delta, bmap,
                                      nc_ng::Int32, total::Int32, step::Int32,
                                      NBX::Int32, NBY::Int32, NBZ::Int32,
                                      NBT::Int32, schedule_code::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
            x[idx] += delta[idx]
        end
    end
    return nothing
end

function _amg_sub_field_kernel!(y, x)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= length(y)
        y[idx] -= x[idx]
    end
    return nothing
end

function _amg_oe_init_fields_kernel!(delta, schur_r, b_odd, res, bmap,
                                     nc_ng::Int32, total::Int32, step::Int32,
                                     NBX::Int32, NBY::Int32, NBZ::Int32,
                                     NBT::Int32, schedule_code::Int32,
                                     blockinfo)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
        r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))
        bid = bmap[site]
        active = _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        parity = Int32(_amg_fine_site_parity(b, r, blockinfo))
        delta[idx] = zero(ComplexF64)
        schur_r[idx] = active && parity == Int32(0) ? res[idx] : zero(ComplexF64)
        b_odd[idx] = active && parity == Int32(1) ? res[idx] : zero(ComplexF64)
    end
    return nothing
end

function _amg_oe_sub_even_kernel!(schur_r, A_b_odd, bmap,
                                  nc_ng::Int32, total::Int32, step::Int32,
                                  NBX::Int32, NBY::Int32, NBZ::Int32,
                                  NBT::Int32, schedule_code::Int32,
                                  blockinfo)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
        r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))
        bid = bmap[site]
        active = _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        parity = Int32(_amg_fine_site_parity(b, r, blockinfo))
        if active && parity == Int32(0)
            schur_r[idx] -= A_b_odd[idx]
        else
            schur_r[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_oe_extract_odd_kernel!(odd_part, schur_Ax, bmap,
                                     nc_ng::Int32, total::Int32, step::Int32,
                                     NBX::Int32, NBY::Int32, NBZ::Int32,
                                     NBT::Int32, schedule_code::Int32,
                                     blockinfo)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
        r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))
        bid = bmap[site]
        active = _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        parity = Int32(_amg_fine_site_parity(b, r, blockinfo))
        odd_part[idx] = active && parity == Int32(1) ? schur_Ax[idx] : zero(ComplexF64)
    end
    return nothing
end

function _amg_oe_finish_schur_kernel!(schur_Ax, A_odd_part, bmap,
                                      nc_ng::Int32, total::Int32, step::Int32,
                                      NBX::Int32, NBY::Int32, NBZ::Int32,
                                      NBT::Int32, schedule_code::Int32,
                                      blockinfo)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
        r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))
        bid = bmap[site]
        active = _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        parity = Int32(_amg_fine_site_parity(b, r, blockinfo))
        if active && parity == Int32(0)
            schur_Ax[idx] -= A_odd_part[idx]
        else
            schur_Ax[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_oe_reconstruct_odd_kernel!(delta, b_odd, A_delta, bmap,
                                         nc_ng::Int32, total::Int32, step::Int32,
                                         NBX::Int32, NBY::Int32, NBZ::Int32,
                                         NBT::Int32, schedule_code::Int32,
                                         blockinfo)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
        r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))
        bid = bmap[site]
        active = _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        parity = Int32(_amg_fine_site_parity(b, r, blockinfo))
        if active && parity == Int32(1)
            delta[idx] = b_odd[idx] - A_delta[idx]
        elseif !active
            delta[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_block_mr_dot_kernel!(num, den, Alr, local_r,
                                   block_sites, block_counts,
                                   nc_ng::Int32, step::Int32,
                                   NBX::Int32, NBY::Int32, NBZ::Int32,
                                   NBT::Int32, schedule_code::Int32)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh_num = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))
    sh_den = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz), Int(bsz) * sizeof(ComplexF64))
    acc_num = zero(ComplexF64)
    acc_den = zero(ComplexF64)
    if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        total = block_counts[bid] * nc_ng
        p = tid
        while p <= total
            site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
            comp = (p - Int32(1)) % nc_ng + Int32(1)
            site = block_sites[site_pos, bid]
            idx = (site - Int32(1)) * nc_ng + comp
            a = Alr[idx]
            r = local_r[idx]
            acc_num += conj(a) * r
            acc_den += conj(a) * a
            p += bsz
        end
    end
    sh_num[tid] = acc_num
    sh_den[tid] = acc_den
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh_num[tid] += sh_num[tid + half]
            sh_den[tid] += sh_den[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        num[bid] = sh_num[1]
        den[bid] = sh_den[1]
    end
    return nothing
end

function _amg_block_mr_update_kernel!(delta, local_r, Alr, num, den, bmap,
                                      nc_ng::Int32, total::Int32, step::Int32,
                                      NBX::Int32, NBY::Int32, NBZ::Int32,
                                      NBT::Int32, schedule_code::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
            d = den[bid]
            if abs(d) > 1.0e-30
                alpha = num[bid] / d
                delta[idx] += alpha * local_r[idx]
                local_r[idx] -= alpha * Alr[idx]
            end
        end
    end
    return nothing
end

function _amg_color_block_dot_kernel!(out, x, y, block_sites, block_counts,
                                      nc_ng::Int32, step::Int32,
                                      NBX::Int32, NBY::Int32, NBZ::Int32,
                                      NBT::Int32, schedule_code::Int32)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))
    acc = zero(ComplexF64)
    if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        total = block_counts[bid] * nc_ng
        p = tid
        while p <= total
            site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
            comp = (p - Int32(1)) % nc_ng + Int32(1)
            site = block_sites[site_pos, bid]
            idx = (site - Int32(1)) * nc_ng + comp
            acc += conj(x[idx]) * y[idx]
            p += bsz
        end
    end
    sh[tid] = acc
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        out[bid] = sh[1]
    end
    return nothing
end

@inline function _amg_gmres_h_index(row::Int32, col::Int32, bid::Int32,
                                    mplus::Int32, m::Int32)
    return row + (col - Int32(1)) * mplus + (bid - Int32(1)) * mplus * m
end

function _amg_color_block_dot_h_kernel!(H, row::Int32, col::Int32,
                                        x, y, block_sites, block_counts,
                                        nc_ng::Int32, step::Int32,
                                        NBX::Int32, NBY::Int32, NBZ::Int32,
                                        NBT::Int32, schedule_code::Int32,
                                        mplus::Int32, m::Int32)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))
    acc = zero(ComplexF64)
    if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
        total = block_counts[bid] * nc_ng
        p = tid
        while p <= total
            site_pos = (p - Int32(1)) ÷ nc_ng + Int32(1)
            comp = (p - Int32(1)) % nc_ng + Int32(1)
            site = block_sites[site_pos, bid]
            idx = (site - Int32(1)) * nc_ng + comp
            acc += conj(x[idx]) * y[idx]
            p += bsz
        end
    end
    sh[tid] = acc
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        H[_amg_gmres_h_index(row, col, bid, mplus, m)] = sh[1]
    end
    return nothing
end

function _amg_color_invnorm_from_vec_kernel!(coeff, norm_out, dotvals,
                                             step::Int32,
                                             NBX::Int32, NBY::Int32, NBZ::Int32,
                                             NBT::Int32, schedule_code::Int32)
    bid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if bid <= length(coeff)
        if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
            nrm = sqrt(max(real(dotvals[bid]), 0.0))
            norm_out[bid] = nrm
            coeff[bid] = nrm > 1.0e-30 ? ComplexF64(1.0 / nrm) : zero(ComplexF64)
        else
            norm_out[bid] = 0.0
            coeff[bid] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_color_invnorm_from_h_kernel!(coeff, H, row::Int32, col::Int32,
                                           step::Int32,
                                           NBX::Int32, NBY::Int32, NBZ::Int32,
                                           NBT::Int32, schedule_code::Int32,
                                           mplus::Int32, m::Int32)
    bid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if bid <= length(coeff)
        if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
            idx = _amg_gmres_h_index(row, col, bid, mplus, m)
            nrm = sqrt(max(real(H[idx]), 0.0))
            H[idx] = ComplexF64(nrm)
            coeff[bid] = nrm > 1.0e-30 ? ComplexF64(1.0 / nrm) : zero(ComplexF64)
        else
            coeff[bid] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_color_block_axpby_coeff_kernel!(y, x, coeff, bmap,
                                              nc_ng::Int32, total::Int32,
                                              step::Int32,
                                              NBX::Int32, NBY::Int32, NBZ::Int32,
                                              NBT::Int32, schedule_code::Int32,
                                              α, β)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
            y[idx] = α * coeff[bid] * x[idx] + β * y[idx]
        elseif β == zero(β)
            y[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_color_block_axpby_h_kernel!(y, x, H, row::Int32, col::Int32,
                                          bmap, nc_ng::Int32, total::Int32,
                                          step::Int32,
                                          NBX::Int32, NBY::Int32, NBZ::Int32,
                                          NBT::Int32, schedule_code::Int32,
                                          mplus::Int32, m::Int32,
                                          α, β)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
            c = H[_amg_gmres_h_index(row, col, bid, mplus, m)]
            y[idx] = α * c * x[idx] + β * y[idx]
        elseif β == zero(β)
            y[idx] = zero(ComplexF64)
        end
    end
    return nothing
end

function _amg_color_gmres_update_kernel!(delta, v, ycoeff, k::Int32,
                                         bmap, nc_ng::Int32, total::Int32,
                                         step::Int32,
                                         NBX::Int32, NBY::Int32, NBZ::Int32,
                                         NBT::Int32, schedule_code::Int32,
                                         m::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        if _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)
            delta[idx] += ycoeff[k + (bid - Int32(1)) * m] * v[idx]
        end
    end
    return nothing
end

function _amg_local_wilson_block_kernel!(yf, xf, U1, U2, U3, U4, bmap,
                                         κ::Float64, blockinfo,
                                         step::Int32, NBX::Int32, NBY::Int32, NBZ::Int32,
                                         NBT::Int32, schedule_code::Int32)
    b = Int64(threadIdx().x)
    r = Int64(blockIdx().x)
    site = Int32((r - 1) * blockinfo.blocksize + b)
    bid = bmap[site]
    active = _amg_sap_block_active_from_bid(bid, step, NBX, NBY, NBZ, NBT, schedule_code)

    if !active
        @inbounds begin
            for ig in 1:4, ic in 1:3
                yf[ic, ig, b, r] = zero(ComplexF64)
            end
        end
        return nothing
    end

    @inbounds begin
        x11 = xf[1,1,b,r]; x21 = xf[2,1,b,r]; x31 = xf[3,1,b,r]
        x12 = xf[1,2,b,r]; x22 = xf[2,2,b,r]; x32 = xf[3,2,b,r]
        x13 = xf[1,3,b,r]; x23 = xf[2,3,b,r]; x33 = xf[3,3,b,r]
        x14 = xf[1,4,b,r]; x24 = xf[2,4,b,r]; x34 = xf[3,4,b,r]
    end

    a11=zero(ComplexF64); a21=zero(ComplexF64); a31=zero(ComplexF64)
    a12=zero(ComplexF64); a22=zero(ComplexF64); a32=zero(ComplexF64)
    a13=zero(ComplexF64); a23=zero(ComplexF64); a33=zero(ComplexF64)
    a14=zero(ComplexF64); a24=zero(ComplexF64); a34=zero(ComplexF64)

    for mu in Int32(1):Int32(4)
        U = mu == Int32(1) ? U1 : (mu == Int32(2) ? U2 : (mu == Int32(3) ? U3 : U4))

        bf, rf = _amg_shift_site(b, r, mu, Int32(1), blockinfo)
        site_f = Int32((rf - 1) * blockinfo.blocksize + bf)
        if bmap[site_f] == bid
            fac = _amg_bc_factor_site(b, r, mu, Int32(1), blockinfo)
            @inbounds begin
                xp11 = xf[1,1,bf,rf]*fac; xp21 = xf[2,1,bf,rf]*fac; xp31 = xf[3,1,bf,rf]*fac
                xp12 = xf[1,2,bf,rf]*fac; xp22 = xf[2,2,bf,rf]*fac; xp32 = xf[3,2,bf,rf]*fac
                xp13 = xf[1,3,bf,rf]*fac; xp23 = xf[2,3,bf,rf]*fac; xp33 = xf[3,3,bf,rf]*fac
                xp14 = xf[1,4,bf,rf]*fac; xp24 = xf[2,4,bf,rf]*fac; xp34 = xf[3,4,bf,rf]*fac
                u11=U[1,1,b,r]; u12=U[1,2,b,r]; u13=U[1,3,b,r]
                u21=U[2,1,b,r]; u22=U[2,2,b,r]; u23=U[2,3,b,r]
                u31=U[3,1,b,r]; u32=U[3,2,b,r]; u33=U[3,3,b,r]
            end
            t11 = u11*xp11 + u12*xp21 + u13*xp31; t21 = u21*xp11 + u22*xp21 + u23*xp31; t31 = u31*xp11 + u32*xp21 + u33*xp31
            t12 = u11*xp12 + u12*xp22 + u13*xp32; t22 = u21*xp12 + u22*xp22 + u23*xp32; t32 = u31*xp12 + u32*xp22 + u33*xp32
            t13 = u11*xp13 + u12*xp23 + u13*xp33; t23 = u21*xp13 + u22*xp23 + u23*xp33; t33 = u31*xp13 + u32*xp23 + u33*xp33
            t14 = u11*xp14 + u12*xp24 + u13*xp34; t24 = u21*xp14 + u22*xp24 + u23*xp34; t34 = u31*xp14 + u32*xp24 + u33*xp34
            p11,p12,p13,p14 = _amg_project_minus(mu, t11,t12,t13,t14)
            p21,p22,p23,p24 = _amg_project_minus(mu, t21,t22,t23,t24)
            p31,p32,p33,p34 = _amg_project_minus(mu, t31,t32,t33,t34)
            a11 += κ*p11; a21 += κ*p21; a31 += κ*p31
            a12 += κ*p12; a22 += κ*p22; a32 += κ*p32
            a13 += κ*p13; a23 += κ*p23; a33 += κ*p33
            a14 += κ*p14; a24 += κ*p24; a34 += κ*p34
        end

        bb, rb = _amg_shift_site(b, r, mu, -Int32(1), blockinfo)
        site_b = Int32((rb - 1) * blockinfo.blocksize + bb)
        if bmap[site_b] == bid
            fac = _amg_bc_factor_site(b, r, mu, -Int32(1), blockinfo)
            @inbounds begin
                xm11 = xf[1,1,bb,rb]*fac; xm21 = xf[2,1,bb,rb]*fac; xm31 = xf[3,1,bb,rb]*fac
                xm12 = xf[1,2,bb,rb]*fac; xm22 = xf[2,2,bb,rb]*fac; xm32 = xf[3,2,bb,rb]*fac
                xm13 = xf[1,3,bb,rb]*fac; xm23 = xf[2,3,bb,rb]*fac; xm33 = xf[3,3,bb,rb]*fac
                xm14 = xf[1,4,bb,rb]*fac; xm24 = xf[2,4,bb,rb]*fac; xm34 = xf[3,4,bb,rb]*fac
                v11=conj(U[1,1,bb,rb]); v12=conj(U[2,1,bb,rb]); v13=conj(U[3,1,bb,rb])
                v21=conj(U[1,2,bb,rb]); v22=conj(U[2,2,bb,rb]); v23=conj(U[3,2,bb,rb])
                v31=conj(U[1,3,bb,rb]); v32=conj(U[2,3,bb,rb]); v33=conj(U[3,3,bb,rb])
            end
            q11 = v11*xm11 + v12*xm21 + v13*xm31; q21 = v21*xm11 + v22*xm21 + v23*xm31; q31 = v31*xm11 + v32*xm21 + v33*xm31
            q12 = v11*xm12 + v12*xm22 + v13*xm32; q22 = v21*xm12 + v22*xm22 + v23*xm32; q32 = v31*xm12 + v32*xm22 + v33*xm32
            q13 = v11*xm13 + v12*xm23 + v13*xm33; q23 = v21*xm13 + v22*xm23 + v23*xm33; q33 = v31*xm13 + v32*xm23 + v33*xm33
            q14 = v11*xm14 + v12*xm24 + v13*xm34; q24 = v21*xm14 + v22*xm24 + v23*xm34; q34 = v31*xm14 + v32*xm24 + v33*xm34
            p11,p12,p13,p14 = _amg_project_plus(mu, q11,q12,q13,q14)
            p21,p22,p23,p24 = _amg_project_plus(mu, q21,q22,q23,q24)
            p31,p32,p33,p34 = _amg_project_plus(mu, q31,q32,q33,q34)
            a11 += κ*p11; a21 += κ*p21; a31 += κ*p31
            a12 += κ*p12; a22 += κ*p22; a32 += κ*p32
            a13 += κ*p13; a23 += κ*p23; a33 += κ*p33
            a14 += κ*p14; a24 += κ*p24; a34 += κ*p34
        end
    end

    @inbounds begin
        yf[1,1,b,r] = x11 - a11; yf[2,1,b,r] = x21 - a21; yf[3,1,b,r] = x31 - a31
        yf[1,2,b,r] = x12 - a12; yf[2,2,b,r] = x22 - a22; yf[3,2,b,r] = x32 - a32
        yf[1,3,b,r] = x13 - a13; yf[2,3,b,r] = x23 - a23; yf[3,3,b,r] = x33 - a33
        yf[1,4,b,r] = x14 - a14; yf[2,4,b,r] = x24 - a24; yf[3,4,b,r] = x34 - a34
    end
    return nothing
end

function _amg_boundary8_residual_update_kernel!(res, delta, affected_sites,
                                                 U1, U2, U3, U4,
                                                 κ::Float64, blockinfo,
                                                 naffected::Int32)
    affected_idx = Int32(threadIdx().x) +
                   (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if affected_idx > naffected
        return nothing
    end

    site = affected_sites[affected_idx]
    b = Int64((site - Int32(1)) % Int32(blockinfo.blocksize) + Int32(1))
    r = Int64((site - Int32(1)) ÷ Int32(blockinfo.blocksize) + Int32(1))

    @inbounds begin
        x11 = delta[1,1,b,r]; x21 = delta[2,1,b,r]; x31 = delta[3,1,b,r]
        x12 = delta[1,2,b,r]; x22 = delta[2,2,b,r]; x32 = delta[3,2,b,r]
        x13 = delta[1,3,b,r]; x23 = delta[2,3,b,r]; x33 = delta[3,3,b,r]
        x14 = delta[1,4,b,r]; x24 = delta[2,4,b,r]; x34 = delta[3,4,b,r]
    end

    a11=zero(ComplexF64); a21=zero(ComplexF64); a31=zero(ComplexF64)
    a12=zero(ComplexF64); a22=zero(ComplexF64); a32=zero(ComplexF64)
    a13=zero(ComplexF64); a23=zero(ComplexF64); a33=zero(ComplexF64)
    a14=zero(ComplexF64); a24=zero(ComplexF64); a34=zero(ComplexF64)

    for mu in Int32(1):Int32(4)
        U = mu == Int32(1) ? U1 : (mu == Int32(2) ? U2 : (mu == Int32(3) ? U3 : U4))

        bf, rf = _amg_shift_site(b, r, mu, Int32(1), blockinfo)
        fac = _amg_bc_factor_site(b, r, mu, Int32(1), blockinfo)
        @inbounds begin
            xp11 = delta[1,1,bf,rf]*fac; xp21 = delta[2,1,bf,rf]*fac; xp31 = delta[3,1,bf,rf]*fac
            xp12 = delta[1,2,bf,rf]*fac; xp22 = delta[2,2,bf,rf]*fac; xp32 = delta[3,2,bf,rf]*fac
            xp13 = delta[1,3,bf,rf]*fac; xp23 = delta[2,3,bf,rf]*fac; xp33 = delta[3,3,bf,rf]*fac
            xp14 = delta[1,4,bf,rf]*fac; xp24 = delta[2,4,bf,rf]*fac; xp34 = delta[3,4,bf,rf]*fac
            u11=U[1,1,b,r]; u12=U[1,2,b,r]; u13=U[1,3,b,r]
            u21=U[2,1,b,r]; u22=U[2,2,b,r]; u23=U[2,3,b,r]
            u31=U[3,1,b,r]; u32=U[3,2,b,r]; u33=U[3,3,b,r]
        end
        t11 = u11*xp11 + u12*xp21 + u13*xp31; t21 = u21*xp11 + u22*xp21 + u23*xp31; t31 = u31*xp11 + u32*xp21 + u33*xp31
        t12 = u11*xp12 + u12*xp22 + u13*xp32; t22 = u21*xp12 + u22*xp22 + u23*xp32; t32 = u31*xp12 + u32*xp22 + u33*xp32
        t13 = u11*xp13 + u12*xp23 + u13*xp33; t23 = u21*xp13 + u22*xp23 + u23*xp33; t33 = u31*xp13 + u32*xp23 + u33*xp33
        t14 = u11*xp14 + u12*xp24 + u13*xp34; t24 = u21*xp14 + u22*xp24 + u23*xp34; t34 = u31*xp14 + u32*xp24 + u33*xp34
        p11,p12,p13,p14 = _amg_project_minus(mu, t11,t12,t13,t14)
        p21,p22,p23,p24 = _amg_project_minus(mu, t21,t22,t23,t24)
        p31,p32,p33,p34 = _amg_project_minus(mu, t31,t32,t33,t34)
        a11 += κ*p11; a21 += κ*p21; a31 += κ*p31
        a12 += κ*p12; a22 += κ*p22; a32 += κ*p32
        a13 += κ*p13; a23 += κ*p23; a33 += κ*p33
        a14 += κ*p14; a24 += κ*p24; a34 += κ*p34

        bb, rb = _amg_shift_site(b, r, mu, -Int32(1), blockinfo)
        fac = _amg_bc_factor_site(b, r, mu, -Int32(1), blockinfo)
        @inbounds begin
            xm11 = delta[1,1,bb,rb]*fac; xm21 = delta[2,1,bb,rb]*fac; xm31 = delta[3,1,bb,rb]*fac
            xm12 = delta[1,2,bb,rb]*fac; xm22 = delta[2,2,bb,rb]*fac; xm32 = delta[3,2,bb,rb]*fac
            xm13 = delta[1,3,bb,rb]*fac; xm23 = delta[2,3,bb,rb]*fac; xm33 = delta[3,3,bb,rb]*fac
            xm14 = delta[1,4,bb,rb]*fac; xm24 = delta[2,4,bb,rb]*fac; xm34 = delta[3,4,bb,rb]*fac
            v11=conj(U[1,1,bb,rb]); v12=conj(U[2,1,bb,rb]); v13=conj(U[3,1,bb,rb])
            v21=conj(U[1,2,bb,rb]); v22=conj(U[2,2,bb,rb]); v23=conj(U[3,2,bb,rb])
            v31=conj(U[1,3,bb,rb]); v32=conj(U[2,3,bb,rb]); v33=conj(U[3,3,bb,rb])
        end
        q11 = v11*xm11 + v12*xm21 + v13*xm31; q21 = v21*xm11 + v22*xm21 + v23*xm31; q31 = v31*xm11 + v32*xm21 + v33*xm31
        q12 = v11*xm12 + v12*xm22 + v13*xm32; q22 = v21*xm12 + v22*xm22 + v23*xm32; q32 = v31*xm12 + v32*xm22 + v33*xm32
        q13 = v11*xm13 + v12*xm23 + v13*xm33; q23 = v21*xm13 + v22*xm23 + v23*xm33; q33 = v31*xm13 + v32*xm23 + v33*xm33
        q14 = v11*xm14 + v12*xm24 + v13*xm34; q24 = v21*xm14 + v22*xm24 + v23*xm34; q34 = v31*xm14 + v32*xm24 + v33*xm34
        p11,p12,p13,p14 = _amg_project_plus(mu, q11,q12,q13,q14)
        p21,p22,p23,p24 = _amg_project_plus(mu, q21,q22,q23,q24)
        p31,p32,p33,p34 = _amg_project_plus(mu, q31,q32,q33,q34)
        a11 += κ*p11; a21 += κ*p21; a31 += κ*p31
        a12 += κ*p12; a22 += κ*p22; a32 += κ*p32
        a13 += κ*p13; a23 += κ*p23; a33 += κ*p33
        a14 += κ*p14; a24 += κ*p24; a34 += κ*p34
    end

    @inbounds begin
        res[1,1,b,r] -= x11 - a11; res[2,1,b,r] -= x21 - a21; res[3,1,b,r] -= x31 - a31
        res[1,2,b,r] -= x12 - a12; res[2,2,b,r] -= x22 - a22; res[3,2,b,r] -= x32 - a32
        res[1,3,b,r] -= x13 - a13; res[2,3,b,r] -= x23 - a23; res[3,3,b,r] -= x33 - a33
        res[1,4,b,r] -= x14 - a14; res[2,4,b,r] -= x24 - a24; res[3,4,b,r] -= x34 - a34
    end
    return nothing
end

function _amg_boundary8_residual_update!(res::WF3x1, delta::WF3x1, D,
                                         setup::DDalphaAMG_Setup, step::Integer)
    if !(res isa WF3x1GPU && delta isa WF3x1GPU && 0 <= step <= 7 &&
         _amg_can_fuse_wilson(D, res, delta))
        return false
    end

    affected_sites = _ensure_gpu_cache!(setup)[:boundary8_affected_sites][step + 1]
    naffected = length(affected_sites)
    naffected == 0 && return true
    threads = 256
    @cuda threads=threads blocks=cld(naffected, threads) _amg_boundary8_residual_update_kernel!(
        res.f, delta.f, affected_sites,
        D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
        Float64(_amg_dirac_kappa(D)), res.blockinfo, Int32(naffected))
    return true
end

function _amg_gamma5_kernel!(x, total_sites::Int32, nc::Int32, ng::Int32)
    site = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if site <= total_sites
        base = (site - Int32(1)) * nc * ng
        for ic in Int32(1):nc
            x[base + ic] = -x[base + ic]
            x[base + nc + ic] = -x[base + nc + ic]
        end
    end
    return nothing
end

function _amg_chiral_project_kernel!(x, total_sites::Int32, nc::Int32, ng::Int32, plus::Bool)
    site = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if site <= total_sites
        base = (site - Int32(1)) * nc * ng
        if plus
            for ic in Int32(1):nc
                x[base + ic] = ComplexF64(0.0)
                x[base + nc + ic] = ComplexF64(0.0)
            end
        else
            for ic in Int32(1):nc
                x[base + Int32(2) * nc + ic] = ComplexF64(0.0)
                x[base + Int32(3) * nc + ic] = ComplexF64(0.0)
            end
        end
    end
    return nothing
end

"""
    block_project!(f, bx, by, bz, bt, r)

Keep only the selected block `(bx, by, bz, bt)` and zero all sites outside it.
"""
function block_project!(f::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NX, NY, NZ, NT = f.NX, f.NY, f.NZ, f.NT
    NC, NG = f.NC, f.NG

    ix_lo = (bx-1)*r + 1; ix_hi = bx*r
    iy_lo = (by-1)*r + 1; iy_hi = by*r
    iz_lo = (bz-1)*r + 1; iz_hi = bz*r
    it_lo = (bt-1)*r + 1; it_hi = bt*r

    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    if !(ix_lo <= ix <= ix_hi && iy_lo <= iy <= iy_hi &&
                         iz_lo <= iz <= iz_hi && it_lo <= it <= it_hi)
                        for ig = 1:NG
                            @simd for ic = 1:NC
                                f[ic, ix, iy, iz, it, ig] = zero(ComplexF64)
                            end
                        end
                    end
                end
            end
        end
    end
    return f
end

"""
    block_parity_project!(f, bx, by, bz, bt, r, parity)

Keep only fine-grid sites of the selected parity inside the selected block.
The parity is `(x + y + z + t) & 1` in zero-based fine coordinates.
"""
function block_parity_project!(f::WF3x1, bx::Int, by::Int, bz::Int, bt::Int,
                               r::Int, parity::Int)
    NX, NY, NZ, NT = f.NX, f.NY, f.NZ, f.NT
    NC, NG = f.NC, f.NG

    ix_lo = (bx-1)*r + 1; ix_hi = bx*r
    iy_lo = (by-1)*r + 1; iy_hi = by*r
    iz_lo = (bz-1)*r + 1; iz_hi = bz*r
    it_lo = (bt-1)*r + 1; it_hi = bt*r

    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    keep = ix_lo <= ix <= ix_hi && iy_lo <= iy <= iy_hi &&
                           iz_lo <= iz <= iz_hi && it_lo <= it <= it_hi &&
                           (((ix - 1) + (iy - 1) + (iz - 1) + (it - 1)) & 1) == parity
                    if !keep
                        for ig = 1:NG
                            @simd for ic = 1:NC
                                f[ic, ix, iy, iz, it, ig] = zero(ComplexF64)
                            end
                        end
                    end
                end
            end
        end
    end
    return f
end

"""
    block_copy!(dest, src, bx, by, bz, bt, r)

Copy the selected block from `src` into the same block location of `dest`.
Entries outside the selected block of `dest` are left unchanged.
"""
function block_copy!(dest::WF3x1, src::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = src.NC, src.NG
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            dest[ic, ix, iy, iz, it, ig] = src[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return dest
end

"""
    block_zero!(f, bx, by, bz, bt, r)

Zero the selected block of `f` in place.
"""
function block_zero!(f::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = f.NC, f.NG
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            f[ic, ix, iy, iz, it, ig] = zero(ComplexF64)
                        end
                    end
                end
            end
        end
    end
    return f
end

"""
    block_axpby!(α, x, β, y, bx, by, bz, bt, r)

Apply `y[block] = α * x[block] + β * y[block]` only inside the selected block.
"""
function block_axpby!(α, x::WF3x1, β, y::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = x.NC, x.NG
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            y[ic, ix, iy, iz, it, ig] =
                                α * x[ic, ix, iy, iz, it, ig] + β * y[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return y
end

"""
    block_inner_product(x, y, bx, by, bz, bt, r) -> ComplexF64

Return the inner product restricted to the selected block.
"""
function block_inner_product(x::WF3x1, y::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    NC, NG = x.NC, x.NG
    val = zero(ComplexF64)
    @inbounds for lt = 1:r
        it = (bt-1)*r + lt
        for lz = 1:r
            iz = (bz-1)*r + lz
            for ly = 1:r
                iy = (by-1)*r + ly
                for lx = 1:r
                    ix = (bx-1)*r + lx
                    for ig = 1:NG
                        @simd for ic = 1:NC
                            val += conj(x[ic, ix, iy, iz, it, ig]) * y[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return val
end

@inline function _gpu_bid(setup_like, bx::Int, by::Int, bz::Int, bt::Int)
    return Int32(block_linear_index(bx, by, bz, bt, setup_like.NBX, setup_like.NBY, setup_like.NBZ))
end

function _gpu_block_context_from_field(f::WF3x1GPU, r::Int)
    NX, NY, NZ, NT = f.NX, f.NY, f.NZ, f.NT
    setup_like = (; r=r, NBX=NX ÷ r, NBY=NY ÷ r, NBZ=NZ ÷ r, NBT=NT ÷ r,
                  Nblocks=(NX ÷ r) * (NY ÷ r) * (NZ ÷ r) * (NT ÷ r),
                  test_vectors=[f])
    cache = _ensure_gpu_cache!(setup_like)
    return setup_like, cache
end

function block_project!(f::WF3x1GPU, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    setup_like, cache = _gpu_block_context_from_field(f, r)
    nc_ng = Int32(size(f.f, 1) * size(f.f, 2))
    total = Int32(length(f.f))
    bid = _gpu_bid(setup_like, bx, by, bz, bt)
    @cuda threads=256 blocks=cld(Int(total), 256) _amg_block_project_kernel!(
        reshape(f.f, :), cache[:bmap], nc_ng, total, bid, false, nothing)
    return f
end

function block_parity_project!(f::WF3x1GPU, bx::Int, by::Int, bz::Int, bt::Int,
                               r::Int, parity::Int)
    setup_like, cache = _gpu_block_context_from_field(f, r)
    nc_ng = Int32(size(f.f, 1) * size(f.f, 2))
    total = Int32(length(f.f))
    bid = _gpu_bid(setup_like, bx, by, bz, bt)
    @cuda threads=256 blocks=cld(Int(total), 256) _amg_block_parity_project_kernel!(
        reshape(f.f, :), cache[:bmap], nc_ng, total, bid, Int32(parity), f.blockinfo)
    return f
end

function block_copy!(dest::WF3x1GPU, src::WF3x1GPU, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    setup_like, cache = _gpu_block_context_from_field(dest, r)
    nc_ng = Int32(size(dest.f, 1) * size(dest.f, 2))
    total = Int32(length(dest.f))
    bid = _gpu_bid(setup_like, bx, by, bz, bt)
    @cuda threads=256 blocks=cld(Int(total), 256) _amg_block_project_kernel!(
        reshape(dest.f, :), cache[:bmap], nc_ng, total, bid, true, reshape(src.f, :))
    return dest
end

function block_zero!(f::WF3x1GPU, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    setup_like, cache = _gpu_block_context_from_field(f, r)
    nc_ng = Int32(size(f.f, 1) * size(f.f, 2))
    total = Int32(length(f.f))
    bid = _gpu_bid(setup_like, bx, by, bz, bt)
    @cuda threads=256 blocks=cld(Int(total), 256) _amg_block_project_kernel!(
        reshape(f.f, :), cache[:bmap], nc_ng, total, bid, true, nothing)
    return f
end

function block_axpby!(α, x::WF3x1GPU, β, y::WF3x1GPU, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    setup_like, cache = _gpu_block_context_from_field(y, r)
    nc_ng = Int32(size(y.f, 1) * size(y.f, 2))
    total = Int32(length(y.f))
    bid = _gpu_bid(setup_like, bx, by, bz, bt)
    @cuda threads=256 blocks=cld(Int(total), 256) _amg_block_axpby_kernel!(
        reshape(y.f, :), reshape(x.f, :), cache[:bmap], nc_ng, total, bid, α, β)
    return y
end

function block_inner_product(x::WF3x1GPU, y::WF3x1GPU, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    setup_like, cache = _gpu_block_context_from_field(x, r)
    out = CUDA.zeros(ComplexF64, 1)
    nc_ng = Int32(size(x.f, 1) * size(x.f, 2))
    bid = _gpu_bid(setup_like, bx, by, bz, bt)
    @cuda threads=256 blocks=1 shmem=(256 * sizeof(ComplexF64)) _amg_block_inner_kernel!(
        out, reshape(x.f, :), reshape(y.f, :), cache[:block_sites], cache[:block_counts], nc_ng, bid)
    return Array(out)[1]
end


# ==========================================
#  Chiral Splitting
# ==========================================

"""
    gamma5_sign(ig)

Return the gamma5 sign for spin component `ig` using the convention
`diag(-1, -1, +1, +1)`.
"""
@inline gamma5_sign(ig::Int) = ig <= 2 ? -1 : 1

"""
    gamma5!(x::WF3x1)

Apply gamma5 in place using the convention `diag(-1, -1, +1, +1)` for spin
components `ig=1:4`.
"""
function gamma5!(x::WF3x1)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC, NG = x.NC, x.NG
    @inbounds for it=1:NT, iz=1:NZ, iy=1:NY, ix=1:NX
        for ic=1:NC
            # ig=1,2 have gamma5 eigenvalue -1
            x[ic,ix,iy,iz,it,1] = -x[ic,ix,iy,iz,it,1]
            x[ic,ix,iy,iz,it,2] = -x[ic,ix,iy,iz,it,2]
            # ig=3,4 have gamma5 eigenvalue +1 and are unchanged
        end
    end
    return x
end

function gamma5!(x::WF3x1GPU)
    total_sites = Int32(size(x.f, 3) * size(x.f, 4))
    @cuda threads=256 blocks=cld(Int(total_sites), 256) _amg_gamma5_kernel!(
        reshape(x.f, :), total_sites, Int32(x.NC), Int32(x.NG))
    return x
end

"""
    chiral_project!(x::WF3x1, chirality::Symbol)

Project `x` in place onto one chiral sector.

- `:plus` keeps `ig=3,4` and zeros `ig=1,2`.
- `:minus` keeps `ig=1,2` and zeros `ig=3,4`.
"""
function chiral_project!(x::WF3x1, chirality::Symbol)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC = x.NC
    @inbounds for it=1:NT, iz=1:NZ, iy=1:NY, ix=1:NX
        for ic=1:NC
            if chirality == :plus
                # P+ zeros ig=1,2 (gamma5 eigenvalue -1)
                x[ic,ix,iy,iz,it,1] = zero(ComplexF64)
                x[ic,ix,iy,iz,it,2] = zero(ComplexF64)
            else
                # P- zeros ig=3,4 (gamma5 eigenvalue +1)
                x[ic,ix,iy,iz,it,3] = zero(ComplexF64)
                x[ic,ix,iy,iz,it,4] = zero(ComplexF64)
            end
        end
    end
    return x
end

function chiral_project!(x::WF3x1GPU, chirality::Symbol)
    total_sites = Int32(size(x.f, 3) * size(x.f, 4))
    @cuda threads=256 blocks=cld(Int(total_sites), 256) _amg_chiral_project_kernel!(
        reshape(x.f, :), total_sites, Int32(x.NC), Int32(x.NG), chirality == :plus)
    return x
end

function _chiral_split_vectors(vectors::Vector{T}) where {T<:WF3x1}
    split_vectors = Vector{T}(undef, 2 * length(vectors))
    for i in eachindex(vectors)
        plus = _copy_wf(vectors[i])
        minus = _copy_wf(vectors[i])
        chiral_project!(plus, :plus)
        chiral_project!(minus, :minus)
        split_vectors[2 * i - 1] = plus
        split_vectors[2 * i] = minus
    end
    return split_vectors
end


# ==========================================
#  Block Orthonormalization
# ==========================================

"""
    block_orthonormalize!(vectors, r; normalize=true)

Block-orthogonalize the test vectors with modified Gram-Schmidt.
For each block B, enforce `<phi_i | phi_j>_B = 0` for `i != j`.
When `normalize=true`, also enforce `<phi_i | phi_i>_B = 1`.
"""
function block_orthonormalize!(vectors::Vector{T}, r::Int; normalize::Bool=true) where {T<:WF3x1}
    if vectors[1] isa WF3x1GPU
        v1 = vectors[1]
        setup_like = (; r=r, NBX=v1.NX ÷ r, NBY=v1.NY ÷ r, NBZ=v1.NZ ÷ r, NBT=v1.NT ÷ r,
                      Nblocks=(v1.NX ÷ r) * (v1.NY ÷ r) * (v1.NZ ÷ r) * (v1.NT ÷ r),
                      test_vectors=vectors)
        cache = _ensure_gpu_cache!(setup_like)
        nc_ng = Int32(size(v1.f, 1) * size(v1.f, 2))
        coeffs = CUDA.zeros(ComplexF64, setup_like.Nblocks)
        norms = CUDA.zeros(Float64, setup_like.Nblocks)
        for v in 1:length(vectors)
            for u in 1:(v-1)
                @cuda threads=256 blocks=setup_like.Nblocks shmem=(2 * 256 * sizeof(ComplexF64)) _amg_orthogonalize_kernel!(
                    reshape(vectors[v].f, :), reshape(vectors[u].f, :),
                    cache[:block_sites], cache[:block_counts], coeffs, nc_ng, normalize)
            end
            if normalize
                @cuda threads=256 blocks=setup_like.Nblocks shmem=(256 * sizeof(Float64)) _amg_normalize_kernel!(
                    reshape(vectors[v].f, :), cache[:block_sites], cache[:block_counts], norms, nc_ng)
            end
        end
        return vectors
    end
    Nv = length(vectors)
    v1 = vectors[1]
    NX, NY, NZ, NT = v1.NX, v1.NY, v1.NZ, v1.NT
    NC, NG = v1.NC, v1.NG
    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r
    block_dof = r^4 * NC * NG

    # Preallocate local buffers reused for each block.
    local_vecs = [Vector{ComplexF64}(undef, block_dof) for _ in 1:Nv]

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        # Extract block-local data.
        for v = 1:Nv
            idx = 0
            for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
                it = (bt-1)*r + lt; iz = (bz-1)*r + lz
                iy = (by-1)*r + ly; ix = (bx-1)*r + lx
                for ig = 1:NG
                    for ic = 1:NC
                        idx += 1
                        local_vecs[v][idx] = vectors[v][ic, ix, iy, iz, it, ig]
                    end
                end
            end
        end

        # Modified Gram-Schmidt
        for v = 1:Nv
            for u = 1:(v-1)
                den = dot(local_vecs[u], local_vecs[u])
                proj = normalize || real(den) <= 1e-28 ?
                       dot(local_vecs[u], local_vecs[v]) :
                       dot(local_vecs[u], local_vecs[v]) / den
                local_vecs[v] .-= proj .* local_vecs[u]
            end
            if normalize
                nrm = norm(local_vecs[v])
                if nrm > 1e-14
                    local_vecs[v] ./= nrm
                end
            end
        end

        # Write back.
        for v = 1:Nv
            idx = 0
            for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
                it = (bt-1)*r + lt; iz = (bz-1)*r + lz
                iy = (by-1)*r + ly; ix = (bx-1)*r + lx
                for ig = 1:NG
                    for ic = 1:NC
                        idx += 1
                        vectors[v][ic, ix, iy, iz, it, ig] = local_vecs[v][idx]
                    end
                end
            end
        end
    end

    return vectors
end

function _block_orthonormalize_chiral_split!(vectors::Vector{T}, r::Int; normalize::Bool=true) where {T<:WF3x1}
    @assert iseven(length(vectors)) "chiral split vectors must come in plus/minus pairs"
    plus_vectors = vectors[1:2:end]
    minus_vectors = vectors[2:2:end]
    block_orthonormalize!(plus_vectors, r; normalize=normalize)
    block_orthonormalize!(minus_vectors, r; normalize=normalize)
    return vectors
end


# ==========================================
#  Restriction: fine -> coarse
# ==========================================

"""
    restrict_f2c(setup, x) -> CoarseVector

Restrict a fine-grid vector to the coarse grid:
`v_coarse[B, i] = <phi_i | x>_B`.
"""
function restrict_f2c(setup::DDalphaAMG_Setup, x::WF3x1)
    if x isa WF3x1GPU
        cv = _coarse_vector(setup)
        return restrict_f2c!(cv, setup, x)
    end
    r = setup.r; Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    cv = _coarse_vector(setup)

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        for v = 1:Nv
            val = zero(ComplexF64)
            tv = setup.test_vectors[v]
            for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
                it = (bt-1)*r + lt; iz = (bz-1)*r + lz
                iy = (by-1)*r + ly; ix = (bx-1)*r + lx
                for ig = 1:NG
                    for ic = 1:NC
                        val += conj(tv[ic, ix, iy, iz, it, ig]) *
                               x[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            cv.data[(bidx-1)*Nv + v] = val
        end
    end

    return cv
end

"""
    restrict_f2c!(cv, setup, x) -> CoarseVector

In-place version of `restrict_f2c`; writes the result into `cv`.
"""
function restrict_f2c!(cv::CoarseVector, setup::DDalphaAMG_Setup, x::WF3x1)
    if x isa WF3x1GPU
        cache = _ensure_gpu_cache!(setup)
        nc = Int32(size(x.f, 1))
        nc_ng = Int32(size(x.f, 1) * size(x.f, 2))
        fill!(cv.data, zero(ComplexF64))
        for v in 1:setup.Nv
            @cuda threads=256 blocks=setup.Nblocks shmem=(256 * sizeof(ComplexF64)) _amg_restrict_one_kernel!(
                cv.data, reshape(setup.test_vectors[v].f, :), reshape(x.f, :),
                cache[:block_sites], cache[:block_counts], nc, nc_ng, Int32(setup.Nv), Int32(v))
        end
        return cv
    end
    r = setup.r; Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        for v = 1:Nv
            val = zero(ComplexF64)
            tv = setup.test_vectors[v]
            for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
                it = (bt-1)*r + lt; iz = (bz-1)*r + lz
                iy = (by-1)*r + ly; ix = (bx-1)*r + lx
                for ig = 1:NG
                    for ic = 1:NC
                        val += conj(tv[ic, ix, iy, iz, it, ig]) *
                               x[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            cv.data[(bidx-1)*Nv + v] = val
        end
    end

    return cv
end


# ==========================================
#  Prolongation: coarse -> fine
# ==========================================

"""
    prolongate_c2f(setup, cv) -> WF3x1

Prolongate a coarse-grid vector to the fine grid:
`x(site) = sum_i c[B(site), i] * phi_i(site)`.
"""
function prolongate_c2f(setup::DDalphaAMG_Setup, cv::CoarseVector)
    if setup.test_vectors[1] isa WF3x1GPU
        f = similar(setup.test_vectors[1])
        return prolongate_c2f!(f, setup, cv)
    end
    r = setup.r; Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    f = similar(setup.test_vectors[1])

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        base = (bidx-1)*Nv
        for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
            it = (bt-1)*r + lt; iz = (bz-1)*r + lz
            iy = (by-1)*r + ly; ix = (bx-1)*r + lx
            for ig = 1:NG
                for ic = 1:NC
                    val = zero(ComplexF64)
                    for v = 1:Nv
                        val += cv.data[base + v] * setup.test_vectors[v][ic, ix, iy, iz, it, ig]
                    end
                    f[ic, ix, iy, iz, it, ig] = val
                end
            end
        end
    end

    return f
end

"""
    prolongate_c2f!(f, setup, cv) -> WF3x1

In-place version of `prolongate_c2f`; writes the result into `f`.
"""
function prolongate_c2f!(f::WF3x1, setup::DDalphaAMG_Setup, cv::CoarseVector)
    if f isa WF3x1GPU
        cache = _ensure_gpu_cache!(setup)
        zero_wf!(f)
        nc_ng = Int32(size(f.f, 1) * size(f.f, 2))
        total = Int32(length(f.f))
        for v in 1:setup.Nv
            @cuda threads=256 blocks=cld(Int(total), 256) _amg_prolong_add_kernel!(
                reshape(f.f, :), reshape(setup.test_vectors[v].f, :), cv.data,
                cache[:bmap], nc_ng, total, Int32(setup.Nv), Int32(v))
        end
        return f
    end
    r = setup.r; Nv = setup.Nv
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT
    NC, NG = setup.NC, setup.NG

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        bidx = block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
        base = (bidx-1)*Nv
        for lt = 1:r, lz = 1:r, ly = 1:r, lx = 1:r
            it = (bt-1)*r + lt; iz = (bz-1)*r + lz
            iy = (by-1)*r + ly; ix = (bx-1)*r + lx
            for ig = 1:NG
                for ic = 1:NC
                    val = zero(ComplexF64)
                    for v = 1:Nv
                        val += cv.data[base + v] * setup.test_vectors[v][ic, ix, iy, iz, it, ig]
                    end
                    f[ic, ix, iy, iz, it, ig] = val
                end
            end
        end
    end

    return f
end


# ==========================================
#  Coarse Grid Operator, matrix-free
# ==========================================

"""
    coarse_apply(setup, D, cv) -> CoarseVector

Apply the matrix-free coarse operator `Dc = R * D * P`.
The restriction is `R = P'`. For paper chiral coarsening, the prolongator
already contains separately coarsened chiral sectors.
"""
function coarse_apply(setup::DDalphaAMG_Setup, D, cv::CoarseVector)
    fine_v = prolongate_c2f(setup, cv)
    Dfine_v = similar(fine_v)
    _amg_mul_dirac!(Dfine_v, D, fine_v)
    return restrict_f2c(setup, Dfine_v)
end

"""
    coarse_apply!(result, setup, D, cv, fine_buf1, fine_buf2) -> CoarseVector

In-place coarse-operator application. `fine_buf1` stores the prolongated vector
and `fine_buf2` stores the fine-grid Dirac application before restriction.
"""
function coarse_apply!(result::CoarseVector, setup::DDalphaAMG_Setup, D, cv::CoarseVector,
                       fine_buf1::WF3x1, fine_buf2::WF3x1)
    prolongate_c2f!(fine_buf1, setup, cv)
    _amg_mul_dirac!(fine_buf2, D, fine_buf1)
    restrict_f2c!(result, setup, fine_buf2)
    return result
end


# ==========================================
#  Coarse operator precomputation
# ==========================================

"""
    precompute_coarse_operator!(setup, D; verbose=false, mode=:stencil)

Build the coarse operator in one of two forms:

- `mode=:stencil` (default): block-stencil operator with Nv x Nv couplings per block/neighbor.
- `mode=:dense`: full dense matrix Dc (legacy path).
"""
function precompute_coarse_operator!(setup::DDalphaAMG_Setup, D;
                                     verbose::Bool=false,
                                     mode::Symbol=:stencil)
    Nv = setup.Nv
    Nblocks = setup.Nblocks
    n = Nv * Nblocks

    setup.Dc_matrix = nothing
    setup.coarse_stencil = nothing

    if mode == :direct_stencil
        use_gpu = setup.test_vectors[1] isa WF3x1GPU
        if !(use_gpu && _amg_can_fuse_wilson(D, setup.test_vectors[1], setup.test_vectors[1]))
            verbose && println("  direct_stencil unavailable for this operator/layout; falling back to stencil probing")
            return precompute_coarse_operator!(setup, D; verbose=verbose, mode=:stencil)
        end
        if min(setup.NBX, setup.NBY, setup.NBZ, setup.NBT) <= 2
            verbose && println("  direct_stencil requires at least 3 coarse blocks per direction; falling back to stencil probing")
            return precompute_coarse_operator!(setup, D; verbose=verbose, mode=:stencil)
        end

        cache = _ensure_gpu_cache!(setup)
        x = setup.test_vectors[1]
        bi = x.blockinfo
        self = CUDA.zeros(ComplexF64, Nv, Nv, Nblocks)
        plus = CUDA.zeros(ComplexF64, Nv, Nv, Nblocks, 4)
        minus = CUDA.zeros(ComplexF64, Nv, Nv, Nblocks, 4)
        nb_plus = Matrix{Int}(undef, Nblocks, 4)
        nb_minus = Matrix{Int}(undef, Nblocks, 4)

        for b in 1:Nblocks, mu in 1:4
            nb_plus[b, mu] = _neighbor_block(b, mu, +1, setup.NBX, setup.NBY, setup.NBZ, setup.NBT)
            nb_minus[b, mu] = _neighbor_block(b, mu, -1, setup.NBX, setup.NBY, setup.NBZ, setup.NBT)
        end
        nb_plus_dev = CuArray(Int32.(nb_plus))
        nb_minus_dev = CuArray(Int32.(nb_minus))

        packed = CUDA.zeros(ComplexF64, setup.NC, setup.NG, bi.blocksize, bi.rsize, Nv)
        nfield = length(x.f)
        for v in 1:Nv
            @cuda threads=256 blocks=cld(nfield, 256) _amg_pack_test_vectors_kernel!(
                packed, reshape(setup.test_vectors[v].f, :), Int32(v))
        end

        if verbose
            println("  coarse operator direct GPU stencil assembly... blocks=$(Nblocks), Nv=$(Nv), R=P'")
        end
        threads = 128
        @cuda threads=threads blocks=(Nblocks, Nv, Nv) shmem=(9 * threads * sizeof(ComplexF64)) _amg_direct_wilson_stencil_kernel!(
            self, plus, minus, packed, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
            cache[:bmap], cache[:block_sites], cache[:block_counts], nb_plus_dev, nb_minus_dev,
            Float64(_amg_dirac_kappa(D)), bi, Int32(Nv))

        setup.coarse_stencil = CoarseStencilOperator(
            Nv, Nblocks, setup.NBX, setup.NBY, setup.NBZ, setup.NBT,
            self, plus, minus, nb_plus_dev, nb_minus_dev
        )

        if verbose
            println("  coarse operator direct stencil assembly done")
        end
        return setup
    elseif mode == :stencil
        if verbose
            nnz_blocks = Nblocks * (1 + 8)
            bytes_est = nnz_blocks * Nv * Nv * 16
            println("  precomputing coarse operator stencil... blocks=$(Nblocks), Nv=$(Nv), approx $(round(bytes_est/1024/1024, digits=1)) MB")
        end

        use_gpu = setup.test_vectors[1] isa WF3x1GPU
        self = use_gpu ? CUDA.zeros(ComplexF64, Nv, Nv, Nblocks) : zeros(ComplexF64, Nv, Nv, Nblocks)
        plus = use_gpu ? CUDA.zeros(ComplexF64, Nv, Nv, Nblocks, 4) : zeros(ComplexF64, Nv, Nv, Nblocks, 4)
        minus = use_gpu ? CUDA.zeros(ComplexF64, Nv, Nv, Nblocks, 4) : zeros(ComplexF64, Nv, Nv, Nblocks, 4)
        nb_plus = Matrix{Int}(undef, Nblocks, 4)
        nb_minus = Matrix{Int}(undef, Nblocks, 4)

        for b in 1:Nblocks, mu in 1:4
            nb_plus[b, mu] = _neighbor_block(b, mu, +1, setup.NBX, setup.NBY, setup.NBZ, setup.NBT)
            nb_minus[b, mu] = _neighbor_block(b, mu, -1, setup.NBX, setup.NBY, setup.NBZ, setup.NBT)
        end

        e_c = _coarse_vector(Nv, Nblocks, setup.test_vectors[1])
        fine_buf1 = similar(setup.test_vectors[1])
        fine_buf2 = similar(setup.test_vectors[1])
        result_c = _coarse_vector(Nv, Nblocks, setup.test_vectors[1])
        nb_plus_dev = use_gpu ? CuArray(Int32.(nb_plus)) : nb_plus
        nb_minus_dev = use_gpu ? CuArray(Int32.(nb_minus)) : nb_minus

        for bs in 1:Nblocks
            src_base = (bs - 1) * Nv
            for v in 1:Nv
                if use_gpu
                    @cuda threads=256 blocks=cld(length(e_c.data), 256) _amg_set_basis_kernel!(
                        e_c.data, Int32(src_base + v))
                else
                    fill!(e_c.data, zero(ComplexF64))
                    e_c.data[src_base + v] = one(ComplexF64)
                end
                coarse_apply!(result_c, setup, D, e_c, fine_buf1, fine_buf2)

                if use_gpu
                    @cuda threads=256 blocks=cld(Nv, 256) _amg_copy_stencil_column_kernel!(
                        self, plus, minus, result_c.data, nb_plus_dev, nb_minus_dev,
                        Int32(Nv), Int32(Nblocks), Int32(bs), Int32(v))
                else
                    # onsite
                    dst_base = (bs - 1) * Nv
                    @inbounds for i in 1:Nv
                        self[i, v, bs] = result_c.data[dst_base + i]
                    end

                    # plus / minus neighbors
                    for mu in 1:4
                        bp = nb_plus[bs, mu]
                        bm = nb_minus[bs, mu]
                        basep = (bp - 1) * Nv
                        basem = (bm - 1) * Nv
                        @inbounds for i in 1:Nv
                            plus[i, v, bs, mu] = result_c.data[basep + i]
                            minus[i, v, bs, mu] = result_c.data[basem + i]
                        end
                    end
                end
            end
        end

        setup.coarse_stencil = CoarseStencilOperator(
            Nv, Nblocks, setup.NBX, setup.NBY, setup.NBZ, setup.NBT,
            self, plus, minus, nb_plus_dev, nb_minus_dev
        )

        if verbose
            println("  coarse operator stencil precompute done")
        end
        return setup
    elseif mode != :dense
        throw(ArgumentError("unknown coarse operator mode: $mode (use :stencil or :dense)"))
    end
    if setup.test_vectors[1] isa WF3x1GPU
        throw(ArgumentError("mode=:dense is CPU-only; use mode=:stencil for GPU DDalphaAMG"))
    end

    if verbose
        println("  dense coarse operator precompute done")
    end

    Dc = zeros(ComplexF64, n, n)
    e_c = _coarse_vector(Nv, Nblocks, setup.test_vectors[1])
    fine_buf1 = similar(setup.test_vectors[1])
    fine_buf2 = similar(setup.test_vectors[1])
    result_c = _coarse_vector(Nv, Nblocks, setup.test_vectors[1])

    for j in 1:n
        fill!(e_c.data, zero(ComplexF64))
        e_c.data[j] = one(ComplexF64)
        coarse_apply!(result_c, setup, D, e_c, fine_buf1, fine_buf2)
        @inbounds for i in 1:n
            Dc[i, j] = result_c.data[i]
        end
    end

    setup.Dc_matrix = Dc
    if verbose
        println("  dense coarse operator precompute done")
    end
    return setup
end


# ==========================================
#  Random fermion generation for setup
# ==========================================

function random_fermion!(x::WF3x1)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC, NG = x.NC, x.NG
    @inbounds for it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX
        for ig = 1:NG
            for ic = 1:NC
                x[ic, ix, iy, iz, it, ig] = randn() + im * randn()
            end
        end
    end
    return x
end

function normalize_fermion!(x::WF3x1)
    nrm2 = real(inner_product(x, x))
    if nrm2 > 1e-30
        axpby!(1 / sqrt(nrm2), x, 0, x)
    end
    return x
end


# ==========================================
#  Smoothers
# ==========================================

"""
    mr_smooth!(x, D, b; niter)

Minimal Residual (MR) smoother. Each iteration computes
`r = b - D*x`, `alpha = <D*r, r> / <D*r, D*r>`, then updates `x += alpha*r`.
"""
function mr_smooth!(x::WF3x1, D, b::WF3x1; niter::Int=4,
                    fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                    flop_counter::Union{Nothing,Ref{Int}}=nothing)

    Dr = similar(x)  # D*r work buffer
    r = similar(x)   # b - D*x work buffer
    zero_wf!(Dr)
    zero_wf!(r)
    for _ = 1:niter
        # r = b - D * x
        _amg_mul_dirac!(r, D, x; fine_apply_counter=fine_apply_counter)
        axpby!(1, b, -1, r)  # r = b - D*x
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))
        # Dr = D * r
        _amg_mul_dirac!(Dr, D, r; fine_apply_counter=fine_apply_counter)
        num = inner_product(Dr, r)
        den = inner_product(Dr, Dr)
        _amg_count_flops!(flop_counter, 2 * _amg_complex_dot_flops(_amg_fine_dofs(x)))
        if abs(den) < 1e-30
            break
        end
        α = num / den
        axpby!(α, r, 1, x)
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))
    end
    return x
end

function _amg_local_gmres_block_solve!(
        δ::WF3x1, D, rhs::WF3x1,
        bx::Int, by::Int, bz::Int, bt::Int, r_blk::Int, restart::Int,
        V::Vector{<:WF3x1}, w::WF3x1,
        H::Matrix{ComplexF64}, e1::Vector{ComplexF64};
        fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
        flop_counter::Union{Nothing,Ref{Int}}=nothing,
        block_dofs::Int)
    block_zero!(δ, bx, by, bz, bt, r_blk)
    fill!(H, zero(ComplexF64))
    fill!(e1, zero(ComplexF64))

    block_project!(V[1], bx, by, bz, bt, r_blk)
    block_copy!(V[1], rhs, bx, by, bz, bt, r_blk)
    β = sqrt(real(block_inner_product(V[1], V[1], bx, by, bz, bt, r_blk)))
    _amg_count_flops!(flop_counter, _amg_complex_norm_flops(block_dofs))
    β < 1e-30 && return δ

    block_axpby!(1 / β, V[1], 0, V[1], bx, by, bz, bt, r_blk)
    _amg_count_flops!(flop_counter, 2 * block_dofs)

    j_final = restart
    for j = 1:restart
        _amg_mul_dirac!(w, D, V[j]; fine_apply_counter=fine_apply_counter)
        block_project!(w, bx, by, bz, bt, r_blk)

        for i = 1:j
            H[i,j] = block_inner_product(V[i], w, bx, by, bz, bt, r_blk)
            block_axpby!(-H[i,j], V[i], 1, w, bx, by, bz, bt, r_blk)
            _amg_count_flops!(flop_counter,
                _amg_complex_dot_flops(block_dofs) + _amg_complex_axpby_flops(block_dofs))
        end

        H[j+1,j] = sqrt(real(block_inner_product(w, w, bx, by, bz, bt, r_blk)))
        _amg_count_flops!(flop_counter, _amg_complex_norm_flops(block_dofs))
        if abs(H[j+1,j]) < 1e-14
            j_final = j
            break
        end

        block_project!(V[j+1], bx, by, bz, bt, r_blk)
        block_copy!(V[j+1], w, bx, by, bz, bt, r_blk)
        block_axpby!(1 / H[j+1,j], V[j+1], 0, V[j+1], bx, by, bz, bt, r_blk)
        _amg_count_flops!(flop_counter, 2 * block_dofs)
    end

    e1[1] = β
    y = H[1:j_final+1, 1:j_final] \ view(e1, 1:j_final+1)
    _amg_count_flops!(flop_counter, 16 * j_final^3)

    for k = 1:j_final
        block_axpby!(y[k], V[k], 1, δ, bx, by, bz, bt, r_blk)
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(block_dofs))
    end
    return δ
end

function _amg_apply_block_odd_even_schur!(
        out::WF3x1, D, x_even::WF3x1,
        bx::Int, by::Int, bz::Int, bt::Int, r_blk::Int,
        odd_part::WF3x1, A_odd_part::WF3x1;
        fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
        flop_counter::Union{Nothing,Ref{Int}}=nothing,
        block_dofs::Int)
    _amg_mul_dirac!(out, D, x_even; fine_apply_counter=fine_apply_counter)
    block_project!(out, bx, by, bz, bt, r_blk)

    block_copy!(odd_part, out, bx, by, bz, bt, r_blk)
    block_parity_project!(odd_part, bx, by, bz, bt, r_blk, 1)

    _amg_mul_dirac!(A_odd_part, D, odd_part; fine_apply_counter=fine_apply_counter)
    block_project!(A_odd_part, bx, by, bz, bt, r_blk)

    block_axpby!(-1, A_odd_part, 1, out, bx, by, bz, bt, r_blk)
    block_parity_project!(out, bx, by, bz, bt, r_blk, 0)
    _amg_count_flops!(flop_counter, 2 * _amg_complex_axpby_flops(block_dofs))
    return out
end

function _amg_local_odd_even_mr_block_solve!(
        δ::WF3x1, D, rhs::WF3x1,
        bx::Int, by::Int, bz::Int, bt::Int, r_blk::Int, nblock_mr::Int,
        local_b::WF3x1, b_odd::WF3x1, A_b_odd::WF3x1,
        schur_r::WF3x1, schur_Ar::WF3x1,
        odd_part::WF3x1, A_odd_part::WF3x1, delta_odd::WF3x1;
        fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
        flop_counter::Union{Nothing,Ref{Int}}=nothing,
        block_dofs::Int)
    block_zero!(δ, bx, by, bz, bt, r_blk)

    block_project!(local_b, bx, by, bz, bt, r_blk)
    block_copy!(local_b, rhs, bx, by, bz, bt, r_blk)

    # Schur RHS on even sites: b_e - A_eo A_oo^{-1} b_o.
    block_copy!(b_odd, local_b, bx, by, bz, bt, r_blk)
    block_parity_project!(b_odd, bx, by, bz, bt, r_blk, 1)
    _amg_mul_dirac!(A_b_odd, D, b_odd; fine_apply_counter=fine_apply_counter)
    block_project!(A_b_odd, bx, by, bz, bt, r_blk)

    block_copy!(schur_r, local_b, bx, by, bz, bt, r_blk)
    block_parity_project!(schur_r, bx, by, bz, bt, r_blk, 0)
    block_axpby!(-1, A_b_odd, 1, schur_r, bx, by, bz, bt, r_blk)
    block_parity_project!(schur_r, bx, by, bz, bt, r_blk, 0)
    _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(block_dofs))

    for _ = 1:nblock_mr
        _amg_apply_block_odd_even_schur!(
            schur_Ar, D, schur_r, bx, by, bz, bt, r_blk,
            odd_part, A_odd_part;
            fine_apply_counter=fine_apply_counter,
            flop_counter=flop_counter,
            block_dofs=block_dofs)

        num = block_inner_product(schur_Ar, schur_r, bx, by, bz, bt, r_blk)
        den = block_inner_product(schur_Ar, schur_Ar, bx, by, bz, bt, r_blk)
        _amg_count_flops!(flop_counter, 2 * _amg_complex_dot_flops(block_dofs))
        if abs(den) < 1e-30
            break
        end
        α = num / den

        block_axpby!(α, schur_r, 1, δ, bx, by, bz, bt, r_blk)
        block_axpby!(-α, schur_Ar, 1, schur_r, bx, by, bz, bt, r_blk)
        block_parity_project!(schur_r, bx, by, bz, bt, r_blk, 0)
        _amg_count_flops!(flop_counter, 2 * _amg_complex_axpby_flops(block_dofs))
    end

    # Reconstruct odd part: x_o = A_oo^{-1}(b_o - A_oe x_e).
    _amg_mul_dirac!(A_odd_part, D, δ; fine_apply_counter=fine_apply_counter)
    block_project!(A_odd_part, bx, by, bz, bt, r_blk)
    block_copy!(delta_odd, local_b, bx, by, bz, bt, r_blk)
    block_axpby!(-1, A_odd_part, 1, delta_odd, bx, by, bz, bt, r_blk)
    block_parity_project!(delta_odd, bx, by, bz, bt, r_blk, 1)
    block_axpby!(1, delta_odd, 1, δ, bx, by, bz, bt, r_blk)
    block_project!(δ, bx, by, bz, bt, r_blk)
    _amg_count_flops!(flop_counter, 2 * _amg_complex_axpby_flops(block_dofs))
    return δ
end


"""
    sap_smooth!(x, D, b, setup; niter, block_solver, nblock_mr, nblock_gmres)

Schwarz Alternating Procedure (SAP) smoother. It sweeps over coarse blocks,
approximates each block-local system with MR by default, matching the
DD-alphaAMG paper parameters. Set `block_solver=:gmres` to use a stronger local
GMRES variant.
"""
function sap_smooth!(x::WF3x1, D, b::WF3x1, setup::DDalphaAMG_Setup;
                     niter::Int=1,
                     block_solver::Symbol=:mr,
                     nblock_mr::Int=4,
                     nblock_gmres::Int=nblock_mr,
                     odd_even::Bool=true,
                     sap_schedule::Symbol=:boundary8,
                     sparse_residual_update::Bool=true,
                     fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                     flop_counter::Union{Nothing,Ref{Int}}=nothing)
    block_solver in (:mr, :gmres) ||
        throw(ArgumentError("block_solver must be :mr or :gmres, got $block_solver"))
    _amg_sap_schedule_code(sap_schedule)
    if odd_even && block_solver != :mr
        throw(ArgumentError("odd-even SAP is currently implemented for block_solver=:mr"))
    end
    if x isa WF3x1GPU && b isa WF3x1GPU && _amg_can_fuse_wilson(D, x, x)
        if odd_even
            return _sap_smooth_gpu_odd_even_mr_batched!(x, D, b, setup;
                                                        niter=niter,
                                                        nblock_mr=nblock_mr,
                                                        sap_schedule=sap_schedule,
                                                        sparse_residual_update=sparse_residual_update,
                                                        fine_apply_counter=fine_apply_counter,
                                                        flop_counter=flop_counter)
        end
        if block_solver == :mr
            return _sap_smooth_gpu_mr_batched!(x, D, b, setup;
                                               niter=niter,
                                               nblock_mr=nblock_mr,
                                               sap_schedule=sap_schedule,
                                               sparse_residual_update=sparse_residual_update,
                                               fine_apply_counter=fine_apply_counter,
                                               flop_counter=flop_counter)
        else
            return _sap_smooth_gpu_gmres_batched!(x, D, b, setup;
                                                  niter=niter,
                                                  nblock_gmres=nblock_gmres,
                                                  sap_schedule=sap_schedule,
                                                  fine_apply_counter=fine_apply_counter,
                                                  flop_counter=flop_counter)
        end
    end

    r_blk = setup.r
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT

    # Preallocate reusable work buffers.
    δ      = similar(x)
    res    = similar(x)
    Ddelta = similar(x)
    local_r = block_solver == :mr ? similar(x) : nothing
    Alr = block_solver == :mr ? similar(x) : nothing
    oe_b_odd = odd_even ? similar(x) : nothing
    oe_A_b_odd = odd_even ? similar(x) : nothing
    oe_schur_r = odd_even ? similar(x) : nothing
    oe_schur_Ar = odd_even ? similar(x) : nothing
    oe_odd_part = odd_even ? similar(x) : nothing
    oe_A_odd_part = odd_even ? similar(x) : nothing
    oe_delta_odd = odd_even ? similar(x) : nothing
    gmres_V = block_solver == :gmres ? [similar(x) for _ in 1:(nblock_gmres + 1)] : WF3x1[]
    gmres_w = block_solver == :gmres ? similar(x) : nothing
    gmres_H = block_solver == :gmres ? zeros(ComplexF64, nblock_gmres + 1, nblock_gmres) : zeros(ComplexF64, 0, 0)
    gmres_e1 = block_solver == :gmres ? zeros(ComplexF64, nblock_gmres + 1) : ComplexF64[]
    block_dofs = r_blk^4 * setup.NC * setup.NG

    for _ = 1:niter
        # Global residual: res = b - D*x
        _amg_mul_dirac!(res, D, x; fine_apply_counter=fine_apply_counter)           # res = D*x
        axpby!(1, b, -1, res)     # res = b - D*x
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))

        # Boundary-aware red-black SAP follows the 8-list ordering used by DD-alphaAMG.
        for step in _amg_sap_schedule_steps(sap_schedule)
            for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
                _amg_sap_block_step(bx, by, bz, bt, NBX, NBY, NBZ, NBT, sap_schedule) == step || continue

                if block_solver == :gmres
                    _amg_local_gmres_block_solve!(
                        δ, D, res, bx, by, bz, bt, r_blk, nblock_gmres,
                        gmres_V, gmres_w::WF3x1, gmres_H, gmres_e1;
                        fine_apply_counter=fine_apply_counter,
                        flop_counter=flop_counter,
                        block_dofs=block_dofs)
                elseif odd_even
                    _amg_local_odd_even_mr_block_solve!(
                        δ, D, res, bx, by, bz, bt, r_blk, nblock_mr,
                        local_r::WF3x1, oe_b_odd::WF3x1, oe_A_b_odd::WF3x1,
                        oe_schur_r::WF3x1, oe_schur_Ar::WF3x1,
                        oe_odd_part::WF3x1, oe_A_odd_part::WF3x1, oe_delta_odd::WF3x1;
                        fine_apply_counter=fine_apply_counter,
                        flop_counter=flop_counter,
                        block_dofs=block_dofs)
                else
                    block_zero!(δ, bx, by, bz, bt, r_blk)
                    block_project!(local_r::WF3x1, bx, by, bz, bt, r_blk)
                    block_copy!(local_r::WF3x1, res, bx, by, bz, bt, r_blk)

                    for _ = 1:nblock_mr
                        _amg_mul_dirac!(Alr::WF3x1, D, local_r::WF3x1;
                                        fine_apply_counter=fine_apply_counter)
                        block_project!(Alr::WF3x1, bx, by, bz, bt, r_blk)

                        num = block_inner_product(Alr::WF3x1, local_r::WF3x1, bx, by, bz, bt, r_blk)
                        den = block_inner_product(Alr::WF3x1, Alr::WF3x1, bx, by, bz, bt, r_blk)
                        _amg_count_flops!(flop_counter, 2 * _amg_complex_dot_flops(block_dofs))
                        if abs(den) < 1e-30
                            break
                        end
                        α = num / den

                        block_axpby!(α, local_r::WF3x1, 1, δ, bx, by, bz, bt, r_blk)
                        block_axpby!(-α, Alr::WF3x1, 1, local_r::WF3x1, bx, by, bz, bt, r_blk)
                        _amg_count_flops!(flop_counter, 2 * _amg_complex_axpby_flops(block_dofs))
                    end
                end

                block_axpby!(1, δ, 1, x, bx, by, bz, bt, r_blk)
                _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(block_dofs))
                block_project!(δ, bx, by, bz, bt, r_blk)
                _amg_mul_dirac!(Ddelta, D, δ; fine_apply_counter=fine_apply_counter)
                axpby!(-1, Ddelta, 1, res)
                _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))
            end
        end
    end
    return x
end

function _amg_initial_inverse_smooth!(test_vectors::Vector{T}, D,
                                      setup::DDalphaAMG_Setup;
                                      smoother::Symbol=:sap,
                                      nstages::Int=3,
                                      block_solver::Symbol=:mr,
                                      nblock_mr::Int=4,
                                      nblock_gmres::Int=nblock_mr,
                                      sap_schedule::Symbol=:boundary8,
                                      fine_apply_counter::Union{Nothing,Ref{Int}}=nothing) where {T<:WF3x1}
    nstages > 0 || throw(ArgumentError("nstages must be positive, got $nstages"))
    _amg_sap_schedule_code(sap_schedule)
    work = similar(test_vectors[1])
    for v in eachindex(test_vectors)
        for stage in 1:nstages
            zero_wf!(work)
            if smoother == :sap
                sap_smooth!(work, D, test_vectors[v], setup;
                            niter=stage,
                            block_solver=block_solver,
                            nblock_mr=nblock_mr,
                            nblock_gmres=nblock_gmres,
                            sap_schedule=sap_schedule,
                            fine_apply_counter=fine_apply_counter)
            elseif smoother == :mr
                mr_smooth!(work, D, test_vectors[v];
                           niter=stage,
                           fine_apply_counter=fine_apply_counter)
            else
                throw(ArgumentError("initial inverse smoother must be :sap or :mr, got $smoother"))
            end
            LatticeDiracOperators.substitute_fermion!(test_vectors[v], work)
        end
        normalize_fermion!(test_vectors[v])
    end
    return test_vectors
end

function _amg_initial_direct_smooth!(test_vectors::Vector{T}, D,
                                     setup::DDalphaAMG_Setup;
                                     smoother::Symbol=:sap,
                                     niter::Int=3,
                                     block_solver::Symbol=:mr,
                                     nblock_mr::Int=4,
                                     nblock_gmres::Int=nblock_mr,
                                     sap_schedule::Symbol=:twocolor,
                                     fine_apply_counter::Union{Nothing,Ref{Int}}=nothing) where {T<:WF3x1}
    niter > 0 || throw(ArgumentError("niter must be positive, got $niter"))
    zero_rhs = similar(test_vectors[1])
    zero_wf!(zero_rhs)
    for vector in test_vectors
        if smoother == :sap
            sap_smooth!(vector, D, zero_rhs, setup;
                        niter=niter,
                        block_solver=block_solver,
                        nblock_mr=nblock_mr,
                        nblock_gmres=nblock_gmres,
                        odd_even=false,
                        sap_schedule=sap_schedule,
                        fine_apply_counter=fine_apply_counter)
        elseif smoother == :mr
            mr_smooth!(vector, D, zero_rhs;
                       niter=niter,
                       fine_apply_counter=fine_apply_counter)
        else
            throw(ArgumentError("initial direct smoother must be :sap or :mr, got $smoother"))
        end
    end
    return test_vectors
end

function _amg_direct_near_null_update!(vector::WF3x1, D,
                                       setup::DDalphaAMG_Setup,
                                       r_fine::WF3x1,
                                       r_coarse::CoarseVector,
                                       correction::WF3x1;
                                       update_smoother::Symbol=:sap,
                                       post_sap_iters::Int=1,
                                       post_smooth::Int=4,
                                       block_solver::Symbol=:mr,
                                       nblock_mr::Int=4,
                                       nblock_gmres::Int=nblock_mr,
                                       coarse_tol::Float64=1e-2,
                                       coarse_maxiter::Int=50,
                                       coarse_restart::Int=20,
                                       sap_schedule::Symbol=:twocolor,
                                       arnoldi_mode::Symbol=:scalar,
                                       fine_apply_counter::Union{Nothing,Ref{Int}}=nothing)
    _amg_mul_dirac!(r_fine, D, vector;
                    fine_apply_counter=fine_apply_counter)
    axpby!(1, vector, -1, r_fine)
    restrict_f2c!(r_coarse, setup, r_fine)
    arnoldi_mode == :scalar || throw(ArgumentError(
        "direct near-null setup requires arnoldi_mode=:scalar"))
    e_coarse = _amg_coarse_gmres_restart_boundary(
        setup, D, r_coarse;
        maxiter=coarse_maxiter,
        restart=coarse_restart,
        tol=coarse_tol)
    prolongate_c2f!(correction, setup, e_coarse)
    if update_smoother == :sap
        sap_smooth!(correction, D, r_fine, setup;
                    niter=post_sap_iters,
                    block_solver=block_solver,
                    nblock_mr=nblock_mr,
                    nblock_gmres=nblock_gmres,
                    odd_even=false,
                    sap_schedule=sap_schedule,
                    fine_apply_counter=fine_apply_counter)
    elseif update_smoother == :mr
        mr_smooth!(correction, D, r_fine;
                   niter=post_smooth,
                   fine_apply_counter=fine_apply_counter)
    else
        throw(ArgumentError(
            "direct near-null update smoother must be :sap or :mr, got $update_smoother"))
    end
    axpby!(1, correction, 1, vector)
    normalize_fermion!(vector)
    return vector
end

# ==========================================
#  Coarse Grid Solver: GMRES
# ==========================================

function _ensure_gpu_coarse_arnoldi_cache!(setup::DDalphaAMG_Setup,
                                           ncoarse::Int, restart::Int)
    cache = _ensure_gpu_cache!(setup)
    needs_allocation = get(cache, :coarse_arnoldi_n, 0) != ncoarse ||
                       get(cache, :coarse_arnoldi_restart, 0) != restart
    if needs_allocation
        cache[:coarse_arnoldi_basis] = CUDA.zeros(ComplexF64, ncoarse, restart + 1)
        cache[:coarse_arnoldi_coeff_dev] = CUDA.zeros(ComplexF64, 2restart)
        cache[:coarse_arnoldi_coeff_host] = zeros(ComplexF64, restart)
        cache[:coarse_arnoldi_n] = ncoarse
        cache[:coarse_arnoldi_restart] = restart
    end
    return cache[:coarse_arnoldi_basis],
           cache[:coarse_arnoldi_coeff_dev],
           cache[:coarse_arnoldi_coeff_host]
end

"""
    coarse_gmres(setup, D, b_c; maxiter, restart, tol, coarse_apply_counter) -> CoarseVector

Solve the coarse-grid system `Dc * x_c = b_c` with GMRES. `Dc` is applied either
through a precomputed coarse operator or matrix-free as `R * D * P`.
"""
function coarse_gmres(setup::DDalphaAMG_Setup, D, b_c::CoarseVector;
                      maxiter::Int=20, restart::Int=20, tol::Float64=1e-2,
                      coarse_odd_even::Bool=true,
                      arnoldi_mode::Symbol=:auto,
                      coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                      flop_counter::Union{Nothing,Ref{Int}}=nothing,
                      profile::Union{Nothing,AMGProfile}=nothing)
    arnoldi_mode in (:auto, :scalar, :batched) ||
        throw(ArgumentError("arnoldi_mode must be :auto, :scalar, or :batched"))
    has_Dc = setup.Dc_matrix !== nothing
    has_stencil = setup.coarse_stencil !== nothing
    op = has_stencil ? setup.coarse_stencil::CoarseStencilOperator : nothing
    use_odd_even = coarse_odd_even && has_stencil && _amg_can_coarse_odd_even(op)

    # Allocate a fine-grid buffer only when no precomputed operator is available.
    local fine_buf1, fine_buf2
    if !has_Dc && !has_stencil
        fine_buf1 = similar(setup.test_vectors[1])
        fine_buf2 = similar(setup.test_vectors[1])
    end

    x_c = similar(b_c)
    zero!(x_c)
    rhs_c = b_c
    local oe_self_inv, oe_tmp1, oe_tmp2, oe_tmp3, oe_rhs
    if use_odd_even
        oe_self_inv = _amg_profile_time!(profile, :coarse_odd_even_invert_diagonal) do
            _amg_coarse_self_inverse!(setup, op)
        end
        oe_tmp1 = similar(b_c)
        oe_tmp2 = similar(b_c)
        oe_tmp3 = similar(b_c)
        oe_rhs = similar(b_c)
        _amg_profile_time!(profile, :coarse_odd_even_rhs) do
            _amg_coarse_odd_even_rhs!(oe_rhs, op, oe_self_inv, b_c, oe_tmp1, oe_tmp2, oe_tmp3)
        end
        rhs_c = oe_rhs
    end

    b_norm = _amg_profile_time!(profile, :coarse_rhs_norm) do
        coarse_norm(rhs_c)
    end
    ncoarse = length(rhs_c.data)
    _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))

    if b_norm < 1e-30
        if use_odd_even
            result = similar(x_c)
            _amg_coarse_odd_even_reconstruct!(result, op, oe_self_inv, x_c, b_c,
                                              oe_tmp1, oe_tmp2, oe_tmp3)
            return result
        end
        return x_c
    end

    m = ensure_coarse_gmres_cache!(setup, rhs_c, restart)
    V = setup._gmres_V
    w_c = setup._gmres_w::CoarseVector
    r_c = setup._gmres_r::CoarseVector
    H = setup._gmres_H
    gamma = setup._gmres_e1
    givens_c = zeros(ComplexF64, m)
    givens_s = zeros(ComplexF64, m)
    y = zeros(ComplexF64, m)
    use_batched_arnoldi = rhs_c.data isa CuArray && arnoldi_mode != :scalar
    if arnoldi_mode == :batched && !(rhs_c.data isa CuArray)
        throw(ArgumentError("arnoldi_mode=:batched requires GPU coarse vectors"))
    end
    local arnoldi_basis, arnoldi_coeff_dev, arnoldi_coeff_host
    if use_batched_arnoldi
        arnoldi_basis, arnoldi_coeff_dev, arnoldi_coeff_host =
            _ensure_gpu_coarse_arnoldi_cache!(setup, ncoarse, m)
    end

    # Helper for coarse-operator application.
    @inline function apply_Dc!(w, v)
        _amg_profile_time!(profile, :coarse_apply) do
            if use_odd_even
                _amg_apply_coarse_odd_even_schur!(w, op, oe_self_inv, v, oe_tmp1, oe_tmp2, oe_tmp3)
            elseif has_stencil
                apply_coarse_stencil!(w, op, v)
            elseif has_Dc
                mul!(w.data, setup.Dc_matrix, v.data)
            else
                coarse_apply!(w, setup, D, v, fine_buf1, fine_buf2)
            end
        end
        coarse_apply_counter !== nothing && (coarse_apply_counter[] += 1)
    end

    @inline function finalize_solution!(x)
        use_odd_even || return x
        result = similar(x)
        _amg_profile_time!(profile, :coarse_odd_even_reconstruct) do
            _amg_coarse_odd_even_reconstruct!(result, op, oe_self_inv, x, b_c,
                                              oe_tmp1, oe_tmp2, oe_tmp3)
        end
        return result
    end

    for iter = 1:maxiter

        # residual: r = b - Dc*x
        if iter == 1
            r_c.data .= rhs_c.data
        else
            r_c.data .= rhs_c.data
            apply_Dc!(w_c, x_c)
            _amg_profile_time!(profile, :coarse_residual_axpby) do
                coarse_axpby!(-1.0, w_c, 1.0, r_c)
            end
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(ncoarse))
        end

        β = _amg_profile_time!(profile, :coarse_residual_norm) do
            coarse_norm(r_c)
        end
        _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))
        if β / b_norm < tol
            return finalize_solution!(x_c)
        end

        # Arnoldi process
        fill!(H, zero(ComplexF64))
        fill!(gamma, zero(ComplexF64))
        fill!(givens_c, zero(ComplexF64))
        fill!(givens_s, zero(ComplexF64))
        V[1].data .= r_c.data ./ β
        if use_batched_arnoldi
            view(arnoldi_basis, :, 1) .= V[1].data
        end
        gamma[1] = β
        _amg_count_flops!(flop_counter, 2 * ncoarse)

        j_final = 0
        converged = false
        for j = 1:m

            apply_Dc!(w_c, V[j])

            if use_batched_arnoldi
                H[j+1,j] = _amg_profile_time!(profile, :coarse_batched_mgs) do
                    _amg_gpu_batched_orthogonalize!(
                        w_c.data, arnoldi_basis, j,
                        arnoldi_coeff_dev, arnoldi_coeff_host)
                end
                @inbounds for i in 1:j
                    H[i,j] = arnoldi_coeff_host[i]
                end
                _amg_count_flops!(flop_counter,
                    2j * (_amg_complex_dot_flops(ncoarse) +
                          _amg_complex_axpby_flops(ncoarse)))
            else
                # Modified Gram-Schmidt
                for i = 1:j
                    H[i,j] = _amg_profile_time!(profile, :coarse_mgs_dot) do
                        coarse_dot(V[i], w_c)
                    end
                    _amg_profile_time!(profile, :coarse_mgs_axpby) do
                        coarse_axpby!(-H[i,j], V[i], 1.0, w_c)
                    end
                    _amg_count_flops!(flop_counter,
                        _amg_complex_dot_flops(ncoarse) + _amg_complex_axpby_flops(ncoarse))
                end
                H[j+1,j] = _amg_profile_time!(profile, :coarse_mgs_norm) do
                    coarse_norm(w_c)
                end
            end
            _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))

            j_final = j
            if abs(H[j+1,j]) <= tol / 10
                converged = true
                break
            end
            _amg_profile_time!(profile, :coarse_mgs_normalize) do
                V[j+1].data .= w_c.data ./ H[j+1,j]
            end
            if use_batched_arnoldi
                view(arnoldi_basis, :, j + 1) .= V[j+1].data
            end
            _amg_count_flops!(flop_counter, 2 * ncoarse)

            _amg_profile_time!(profile, :coarse_givens_update) do
                for i = 1:(j - 1)
                    tmp = -givens_s[i] * H[i,j] + givens_c[i] * H[i+1,j]
                    H[i,j] = conj(givens_c[i]) * H[i,j] + conj(givens_s[i]) * H[i+1,j]
                    H[i+1,j] = tmp
                end

                denom = sqrt(abs2(H[j,j]) + abs2(H[j+1,j]))
                givens_s[j] = H[j+1,j] / denom
                givens_c[j] = H[j,j] / denom
                gamma[j+1] = -givens_s[j] * gamma[j]
                gamma[j] = conj(givens_c[j]) * gamma[j]
                H[j,j] = denom
                H[j+1,j] = zero(ComplexF64)
            end
            _amg_count_flops!(flop_counter, 6 * j + 6)

            if abs(gamma[j+1]) / b_norm < tol
                converged = true
                j_final = j
                break
            end
        end

        # Back-substitution on the incrementally updated triangular GMRES system.
        fill!(y, zero(ComplexF64))
        _amg_profile_time!(profile, :coarse_backsolve) do
            for i = j_final:-1:1
                y[i] = gamma[i]
                for k = (i + 1):j_final
                    y[i] -= H[i,k] * y[k]
                end
                y[i] /= H[i,i]
            end
        end
        _amg_count_flops!(flop_counter, 16 * j_final^3)

        for k = 1:j_final
            _amg_profile_time!(profile, :coarse_solution_update) do
                coarse_axpby!(y[k], V[k], 1.0, x_c)
            end
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(ncoarse))
        end

        converged && return finalize_solution!(x_c)

        # Convergence check
        r_c.data .= rhs_c.data
        apply_Dc!(w_c, x_c)
        _amg_profile_time!(profile, :coarse_check_axpby) do
            coarse_axpby!(-1.0, w_c, 1.0, r_c)
        end
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(ncoarse))
        res = _amg_profile_time!(profile, :coarse_check_norm) do
            coarse_norm(r_c)
        end / b_norm
        _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))
        # println("res:$(res),iter:$(iter)")
        if res < tol
            return finalize_solution!(x_c)
        end
    end

    return finalize_solution!(x_c)
end

function _amg_coarse_gmres_restart_boundary(
        setup::DDalphaAMG_Setup, D, b_c::CoarseVector;
        maxiter::Int=50, restart::Int=20, tol::Float64=1e-2,
        coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing,
        flop_counter::Union{Nothing,Ref{Int}}=nothing,
        profile::Union{Nothing,AMGProfile}=nothing)
    x_c = similar(b_c)
    zero!(x_c)
    b_norm = _amg_profile_time!(profile, :coarse_rhs_norm) do
        coarse_norm(b_c)
    end
    ncoarse = length(b_c.data)
    _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))
    b_norm < 1e-30 && return x_c

    has_Dc = setup.Dc_matrix !== nothing
    has_stencil = setup.coarse_stencil !== nothing
    local fine_buf1, fine_buf2
    if !has_Dc && !has_stencil
        fine_buf1 = similar(setup.test_vectors[1])
        fine_buf2 = similar(setup.test_vectors[1])
    end

    m = ensure_coarse_gmres_cache!(setup, b_c, restart)
    V = setup._gmres_V
    w_c = setup._gmres_w::CoarseVector
    r_c = setup._gmres_r::CoarseVector
    H = setup._gmres_H
    e1 = setup._gmres_e1

    @inline function apply_Dc!(w, v)
        _amg_profile_time!(profile, :coarse_apply) do
            if has_stencil
                apply_coarse_stencil!(w, setup.coarse_stencil, v)
            elseif has_Dc
                mul!(w.data, setup.Dc_matrix, v.data)
            else
                coarse_apply!(w, setup, D, v, fine_buf1, fine_buf2)
            end
        end
        coarse_apply_counter !== nothing && (coarse_apply_counter[] += 1)
        return w
    end

    for _ in 1:maxiter
        r_c.data .= b_c.data
        apply_Dc!(w_c, x_c)
        _amg_profile_time!(profile, :coarse_residual_axpby) do
            coarse_axpby!(-1.0, w_c, 1.0, r_c)
        end
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(ncoarse))

        β = _amg_profile_time!(profile, :coarse_residual_norm) do
            coarse_norm(r_c)
        end
        _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))
        β / b_norm < tol && return x_c

        fill!(H, zero(ComplexF64))
        V[1].data .= r_c.data ./ β
        _amg_count_flops!(flop_counter, 2 * ncoarse)

        j_final = m
        for j in 1:m
            apply_Dc!(w_c, V[j])
            for i in 1:j
                H[i,j] = _amg_profile_time!(profile, :coarse_mgs_dot) do
                    coarse_dot(V[i], w_c)
                end
                _amg_profile_time!(profile, :coarse_mgs_axpby) do
                    coarse_axpby!(-H[i,j], V[i], 1.0, w_c)
                end
                _amg_count_flops!(flop_counter,
                    _amg_complex_dot_flops(ncoarse) +
                    _amg_complex_axpby_flops(ncoarse))
            end
            H[j+1,j] = _amg_profile_time!(profile, :coarse_mgs_norm) do
                coarse_norm(w_c)
            end
            _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))
            if abs(H[j+1,j]) < 1e-14
                j_final = j
                break
            end
            _amg_profile_time!(profile, :coarse_mgs_normalize) do
                V[j+1].data .= w_c.data ./ H[j+1,j]
            end
            _amg_count_flops!(flop_counter, 2 * ncoarse)
        end

        fill!(e1, zero(ComplexF64))
        e1[1] = β
        y = _amg_profile_time!(profile, :coarse_lstsq) do
            H[1:j_final+1, 1:j_final] \ view(e1, 1:j_final+1)
        end
        _amg_count_flops!(flop_counter, 16 * j_final^3)
        for k in 1:j_final
            _amg_profile_time!(profile, :coarse_solution_update) do
                coarse_axpby!(y[k], V[k], 1.0, x_c)
            end
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(ncoarse))
        end

        r_c.data .= b_c.data
        apply_Dc!(w_c, x_c)
        _amg_profile_time!(profile, :coarse_check_axpby) do
            coarse_axpby!(-1.0, w_c, 1.0, r_c)
        end
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(ncoarse))
        residual = _amg_profile_time!(profile, :coarse_check_norm) do
            coarse_norm(r_c)
        end / b_norm
        _amg_count_flops!(flop_counter, _amg_complex_norm_flops(ncoarse))
        residual < tol && return x_c
    end
    return x_c
end


# ==========================================
#  V-cycle
# ==========================================

"""
    vcycle(D, b, setup; pre_smooth, post_smooth, coarse_tol, smoother)

Apply one DD-alphaAMG V-cycle and return an approximate solution to `D*x = b`.

Steps:
1. Pre-smooth on the fine grid.
2. Restrict the residual: `r_c = R * (b - D*x)`.
3. Solve the coarse system `Dc * e_c = r_c` with GMRES.
4. Prolongate and add the correction: `x += P * e_c`.
5. Post-smooth on the fine grid.
"""
function vcycle(D, b::WF3x1, setup::DDalphaAMG_Setup;
                pre_smooth::Int=0, post_smooth::Int=2,
                sap_block_mr_steps::Int=4,
                coarse_maxiter::Int=50, coarse_restart::Int=30,
                coarse_tol::Float64=5e-2,
                smoother::Symbol=:sap,
                coarse_odd_even::Bool=true,
                coarse_gmres_mode::Symbol=:incremental,
                sap_odd_even::Bool=true,
                sap_schedule::Symbol=:boundary8,
                sparse_residual_update::Bool=true,
                coarse_arnoldi_mode::Symbol=:auto,
                coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                flop_counter::Union{Nothing,Ref{Int}}=nothing,
                profile::Union{Nothing,AMGProfile}=nothing)
    coarse_gmres_mode in (:incremental, :restart_boundary) ||
        throw(ArgumentError(
            "coarse_gmres_mode must be :incremental or :restart_boundary, got $coarse_gmres_mode"))
    x = similar(b)
    zero_wf!(x)

    # Pre-allocate work buffers to avoid temporary WF allocations in hot path.
    r_fine = similar(b)
    correction = similar(b)
    r_coarse = _coarse_vector(setup)

    # 1. Pre-smoothing
    if pre_smooth > 0
        _amg_profile_time!(profile, :vcycle_pre_smooth) do
            if smoother == :sap
                sap_smooth!(x, D, b, setup; niter=pre_smooth,
                            block_solver=:mr, nblock_mr=sap_block_mr_steps,
                            odd_even=sap_odd_even,
                            sap_schedule=sap_schedule,
                            sparse_residual_update=sparse_residual_update,
                            fine_apply_counter=fine_apply_counter,
                            flop_counter=flop_counter)
            else
                mr_smooth!(x, D, b; niter=pre_smooth,
                           fine_apply_counter=fine_apply_counter,
                           flop_counter=flop_counter)
            end
        end
    end

    # 2. Compute residual and restrict it to the coarse grid.
    _amg_profile_time!(profile, :vcycle_residual_restrict) do
        _amg_mul_dirac!(r_fine, D, x; fine_apply_counter=fine_apply_counter)       # r_fine = D*x
        axpby!(1, b, -1, r_fine) # r_fine = b - D*x
        restrict_f2c!(r_coarse, setup, r_fine)
    end
    _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(b)))
    _amg_count_flops!(flop_counter, _amg_restrict_flops(setup))

    # 3. Coarse-grid solve.
    e_coarse = _amg_profile_time!(profile, :coarse_gmres_total) do
        if coarse_gmres_mode == :restart_boundary
            coarse_odd_even && throw(ArgumentError(
                "coarse_gmres_mode=:restart_boundary does not support coarse_odd_even=true"))
            _amg_coarse_gmres_restart_boundary(
                setup, D, r_coarse;
                maxiter=coarse_maxiter, restart=coarse_restart, tol=coarse_tol,
                coarse_apply_counter=coarse_apply_counter,
                flop_counter=flop_counter,
                profile=profile)
        else
            coarse_gmres(setup, D, r_coarse;
                         maxiter=coarse_maxiter, restart=coarse_restart, tol=coarse_tol,
                         coarse_odd_even=coarse_odd_even,
                         arnoldi_mode=coarse_arnoldi_mode,
                         coarse_apply_counter=coarse_apply_counter,
                         flop_counter=flop_counter,
                         profile=profile)
        end
    end

    # 4. Prolongate the correction to the fine grid.
    _amg_profile_time!(profile, :vcycle_prolongate_update) do
        prolongate_c2f!(correction, setup, e_coarse)
        axpby!(1, correction, 1, x)
    end
    _amg_count_flops!(flop_counter, _amg_prolongate_flops(setup))
    _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(b)))

    # 5. Post-smoothing
    if post_smooth > 0
        _amg_profile_time!(profile, :vcycle_post_smooth) do
            if smoother == :sap
                sap_smooth!(x, D, b, setup; niter=post_smooth,
                            block_solver=:mr, nblock_mr=sap_block_mr_steps,
                            odd_even=sap_odd_even,
                            sap_schedule=sap_schedule,
                            sparse_residual_update=sparse_residual_update,
                            fine_apply_counter=fine_apply_counter,
                            flop_counter=flop_counter)
            else
                mr_smooth!(x, D, b; niter=post_smooth,
                           fine_apply_counter=fine_apply_counter,
                           flop_counter=flop_counter)
            end
        end
    end

    return x
end

function _sap_smooth_gpu_mr_batched!(x::WF3x1GPU, D, b::WF3x1GPU, setup::DDalphaAMG_Setup;
                                     niter::Int=1, nblock_mr::Int=4,
                                     sap_schedule::Symbol=:boundary8,
                                     sparse_residual_update::Bool=true,
                                     fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                                     flop_counter::Union{Nothing,Ref{Int}}=nothing)
    cache = _ensure_gpu_cache!(setup)
    bi = x.blockinfo
    total = Int32(length(x.f))
    nc_ng = Int32(size(x.f, 1) * size(x.f, 2))
    threads = 256
    grid = cld(Int(total), threads)
    NBX = Int32(setup.NBX); NBY = Int32(setup.NBY); NBZ = Int32(setup.NBZ); NBT = Int32(setup.NBT)
    schedule_code = _amg_sap_schedule_code(sap_schedule)

    delta = similar(x)
    local_r = similar(x)
    Alr = similar(x)
    res = similar(x)
    Ddelta = similar(x)
    num = CUDA.zeros(ComplexF64, setup.Nblocks)
    den = CUDA.zeros(ComplexF64, setup.Nblocks)
    block_dofs = setup.r^4 * setup.NC * setup.NG

    for _ in 1:niter
        _amg_mul_dirac!(res, D, x; fine_apply_counter=fine_apply_counter)
        axpby!(1, b, -1, res)
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))

        for step_i in _amg_sap_schedule_steps(sap_schedule)
            step = Int32(step_i)
            @cuda threads=threads blocks=grid _amg_init_color_fields_kernel!(
                reshape(delta.f, :), reshape(local_r.f, :), reshape(res.f, :),
                cache[:bmap], nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code)

            for _ in 1:nblock_mr
                @cuda threads=bi.blocksize blocks=bi.rsize _amg_local_wilson_block_kernel!(
                    Alr.f, local_r.f, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
                    cache[:bmap], Float64(_amg_dirac_kappa(D)), bi, step, NBX, NBY, NBZ, NBT, schedule_code)

                @cuda threads=threads blocks=setup.Nblocks shmem=(2 * threads * sizeof(ComplexF64)) _amg_block_mr_dot_kernel!(
                    num, den, reshape(Alr.f, :), reshape(local_r.f, :),
                    cache[:block_sites], cache[:block_counts], nc_ng, step, NBX, NBY, NBZ, NBT, schedule_code)
                _amg_count_flops!(flop_counter,
                    2 * setup.Nblocks * _amg_complex_dot_flops(block_dofs))

                @cuda threads=threads blocks=grid _amg_block_mr_update_kernel!(
                    reshape(delta.f, :), reshape(local_r.f, :), reshape(Alr.f, :),
                    num, den, cache[:bmap], nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code)
                _amg_count_flops!(flop_counter,
                    2 * setup.Nblocks * _amg_complex_axpby_flops(block_dofs))
            end

            @cuda threads=threads blocks=grid _amg_add_color_field_kernel!(
                reshape(x.f, :), reshape(delta.f, :), cache[:bmap],
                nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code)
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))

            handled = sparse_residual_update && sap_schedule == :boundary8 &&
                      _amg_boundary8_residual_update!(res, delta, D, setup, step_i)
            if !handled
                _amg_mul_dirac!(Ddelta, D, delta; fine_apply_counter=fine_apply_counter)
                @cuda threads=threads blocks=grid _amg_sub_field_kernel!(
                    reshape(res.f, :), reshape(Ddelta.f, :))
                _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))
            end
        end
    end
    return x
end

function _sap_smooth_gpu_odd_even_mr_batched!(x::WF3x1GPU, D, b::WF3x1GPU,
                                              setup::DDalphaAMG_Setup;
                                              niter::Int=1, nblock_mr::Int=4,
                                              sap_schedule::Symbol=:boundary8,
                                              sparse_residual_update::Bool=true,
                                              fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                                              flop_counter::Union{Nothing,Ref{Int}}=nothing)
    cache = _ensure_gpu_cache!(setup)
    bi = x.blockinfo
    total = Int32(length(x.f))
    nc_ng = Int32(size(x.f, 1) * size(x.f, 2))
    threads = 256
    grid = cld(Int(total), threads)
    NBX = Int32(setup.NBX); NBY = Int32(setup.NBY); NBZ = Int32(setup.NBZ); NBT = Int32(setup.NBT)
    schedule_code = _amg_sap_schedule_code(sap_schedule)

    delta = similar(x)
    schur_r = similar(x)
    b_odd = similar(x)
    A_b_odd = similar(x)
    schur_Ar = similar(x)
    odd_part = similar(x)
    A_odd_part = similar(x)
    res = similar(x)
    Ddelta = similar(x)
    num = CUDA.zeros(ComplexF64, setup.Nblocks)
    den = CUDA.zeros(ComplexF64, setup.Nblocks)
    block_dofs = setup.r^4 * setup.NC * setup.NG

    for _ in 1:niter
        _amg_mul_dirac!(res, D, x; fine_apply_counter=fine_apply_counter)
        axpby!(1, b, -1, res)
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))

        for step_i in _amg_sap_schedule_steps(sap_schedule)
            step = Int32(step_i)
            @cuda threads=threads blocks=grid _amg_oe_init_fields_kernel!(
                reshape(delta.f, :), reshape(schur_r.f, :), reshape(b_odd.f, :),
                reshape(res.f, :), cache[:bmap], nc_ng, total, step,
                NBX, NBY, NBZ, NBT, schedule_code, bi)

            @cuda threads=bi.blocksize blocks=bi.rsize _amg_local_wilson_block_kernel!(
                A_b_odd.f, b_odd.f, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
                cache[:bmap], Float64(_amg_dirac_kappa(D)), bi, step, NBX, NBY, NBZ, NBT, schedule_code)

            @cuda threads=threads blocks=grid _amg_oe_sub_even_kernel!(
                reshape(schur_r.f, :), reshape(A_b_odd.f, :), cache[:bmap],
                nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code, bi)
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(block_dofs * setup.Nblocks ÷ 2))

            for _ in 1:nblock_mr
                @cuda threads=bi.blocksize blocks=bi.rsize _amg_local_wilson_block_kernel!(
                    schur_Ar.f, schur_r.f, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
                    cache[:bmap], Float64(_amg_dirac_kappa(D)), bi, step, NBX, NBY, NBZ, NBT, schedule_code)

                @cuda threads=threads blocks=grid _amg_oe_extract_odd_kernel!(
                    reshape(odd_part.f, :), reshape(schur_Ar.f, :), cache[:bmap],
                    nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code, bi)

                @cuda threads=bi.blocksize blocks=bi.rsize _amg_local_wilson_block_kernel!(
                    A_odd_part.f, odd_part.f, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
                    cache[:bmap], Float64(_amg_dirac_kappa(D)), bi, step, NBX, NBY, NBZ, NBT, schedule_code)

                @cuda threads=threads blocks=grid _amg_oe_finish_schur_kernel!(
                    reshape(schur_Ar.f, :), reshape(A_odd_part.f, :), cache[:bmap],
                    nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code, bi)

                @cuda threads=threads blocks=setup.Nblocks shmem=(2 * threads * sizeof(ComplexF64)) _amg_block_mr_dot_kernel!(
                    num, den, reshape(schur_Ar.f, :), reshape(schur_r.f, :),
                    cache[:block_sites], cache[:block_counts], nc_ng, step, NBX, NBY, NBZ, NBT, schedule_code)
                _amg_count_flops!(flop_counter,
                    2 * setup.Nblocks * _amg_complex_dot_flops(block_dofs ÷ 2))

                @cuda threads=threads blocks=grid _amg_block_mr_update_kernel!(
                    reshape(delta.f, :), reshape(schur_r.f, :), reshape(schur_Ar.f, :),
                    num, den, cache[:bmap], nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code)
                _amg_count_flops!(flop_counter,
                    2 * setup.Nblocks * _amg_complex_axpby_flops(block_dofs ÷ 2))
            end

            @cuda threads=bi.blocksize blocks=bi.rsize _amg_local_wilson_block_kernel!(
                A_odd_part.f, delta.f, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
                cache[:bmap], Float64(_amg_dirac_kappa(D)), bi, step, NBX, NBY, NBZ, NBT, schedule_code)

            @cuda threads=threads blocks=grid _amg_oe_reconstruct_odd_kernel!(
                reshape(delta.f, :), reshape(b_odd.f, :), reshape(A_odd_part.f, :),
                cache[:bmap], nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code, bi)
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(block_dofs * setup.Nblocks ÷ 2))

            @cuda threads=threads blocks=grid _amg_add_color_field_kernel!(
                reshape(x.f, :), reshape(delta.f, :), cache[:bmap],
                nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code)
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))

            handled = sparse_residual_update && sap_schedule == :boundary8 &&
                      _amg_boundary8_residual_update!(res, delta, D, setup, step_i)
            if !handled
                _amg_mul_dirac!(Ddelta, D, delta; fine_apply_counter=fine_apply_counter)
                @cuda threads=threads blocks=grid _amg_sub_field_kernel!(
                    reshape(res.f, :), reshape(Ddelta.f, :))
                _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))
            end
        end
    end
    return x
end


# ==========================================
#  FGMRES outer solver
# ==========================================

"""
    fgmres(D, b, setup; x0, maxiter, restart, tol, verbose, smoother, ...)

Flexible GMRES outer solver for `D*x = b`, using the DD-alphaAMG V-cycle as a
right preconditioner. Flexible GMRES allows the preconditioner to vary between
iterations, which is useful for nonlinear or inexact V-cycles.
"""
function fgmres(D, b::WF3x1, setup::DDalphaAMG_Setup;
                x0::Union{Nothing,WF3x1}=nothing,
                maxiter::Int=100, restart::Int=25, tol::Float64=1e-10,
                verbose::Bool=true, smoother::Symbol=:sap,
                pre_smooth::Int=0, post_smooth::Int=2,
                sap_block_mr_steps::Int=4,
                coarse_tol::Float64=5e-2,
                coarse_maxiter::Int=50,
                coarse_restart::Int=30,
                coarse_odd_even::Bool=true,
                coarse_gmres_mode::Symbol=:incremental,
                sap_odd_even::Bool=true,
                sap_schedule::Symbol=:boundary8,
                profile::Bool=false,
                profile_synchronize::Bool=true)

    x = x0 === nothing ? similar(b) : _copy_wf(x0)
    x0 === nothing && zero_wf!(x)
    b_norm = sqrt(real(inner_product(b, b)))
    if b_norm < 1e-30
        return x
    end

    r = _copy_wf(b)  # r = b - D*x buffer
    zero_wf!(r)

    total_coarse_gmres = 0  # coarse GMRES call counter
    _coarse_apply_counter = Ref(0)  # coarse operator application counter
    _fine_apply_counter = Ref(0)
    _variable_coarse_apply_counter = Ref(0)
    _variable_fine_apply_counter = Ref(0)
    _variable_vector_flop_counter = Ref(0)
    _outer_fine_apply_counter = Ref(0)
    _outer_vector_flop_counter = Ref(0)
    _profile = profile ? AMGProfile(synchronize=profile_synchronize) : nothing
    nfine = _amg_fine_dofs(b)

    for cycle = 1:maxiter
        # r = b - D * x
        outer_fine_before = _fine_apply_counter[]
        β = _amg_profile_time!(_profile, :outer_residual) do
            _amg_mul_dirac!(r, D, x; fine_apply_counter=_fine_apply_counter)           # r = D*x
            axpby!(1, b, -1, r)     # r = b - D*x
            sqrt(real(inner_product(r, r)))
        end
        _outer_fine_apply_counter[] += _fine_apply_counter[] - outer_fine_before
        _outer_vector_flop_counter[] += _amg_complex_axpby_flops(nfine) + _amg_complex_norm_flops(nfine)

        if verbose
            println("FGMRES cycle $cycle: relative residual = $(β/b_norm)")
        end

        if β / b_norm < tol
            if verbose
                println("FGMRES converged! ($(cycle) cycles)")
                _amg_print_solver_counts(setup, total_coarse_gmres,
                                         _coarse_apply_counter[],
                                         _fine_apply_counter[],
                                         _variable_coarse_apply_counter[],
                                         _variable_fine_apply_counter[],
                                         _variable_vector_flop_counter[],
                                         _outer_fine_apply_counter[],
                                         _outer_vector_flop_counter[])
                _profile !== nothing && _amg_print_profile(stdout, _profile)
            end
            return x
        end

        # Arnoldi with flexible preconditioning + Givens rotation for early stop
        m = restart
        V = Vector{typeof(b)}(undef, m + 1)
        Z = Vector{typeof(b)}(undef, m)
        H = zeros(ComplexF64, m + 1, m)
        cs = zeros(ComplexF64, m)
        sn = zeros(ComplexF64, m)
        g  = zeros(ComplexF64, m + 1)
        g[1] = β

        V[1] = r / β
        _outer_vector_flop_counter[] += 2 * nfine

        j_final = m
        converged = false
        for j = 1:m
            # V-cycle preconditioner: apply M^{-1} * v_j.
            fine_before_vcycle = _fine_apply_counter[]
            coarse_before_vcycle = _coarse_apply_counter[]
            Z[j] = _amg_profile_time!(_profile, :vcycle_total) do
                vcycle(D, V[j], setup;
                       pre_smooth=pre_smooth, post_smooth=post_smooth,
                       sap_block_mr_steps=sap_block_mr_steps,
                       coarse_maxiter=coarse_maxiter,
                       coarse_restart=coarse_restart,
                       coarse_tol=coarse_tol, smoother=smoother,
                       coarse_odd_even=coarse_odd_even,
                       coarse_gmres_mode=coarse_gmres_mode,
                       sap_odd_even=sap_odd_even,
                       sap_schedule=sap_schedule,
                       coarse_apply_counter=_coarse_apply_counter,
                       fine_apply_counter=_fine_apply_counter,
                       flop_counter=_variable_vector_flop_counter,
                       profile=_profile)
            end
            total_coarse_gmres += 1
            _variable_fine_apply_counter[] += _fine_apply_counter[] - fine_before_vcycle
            _variable_coarse_apply_counter[] += _coarse_apply_counter[] - coarse_before_vcycle

            w = similar(b)
            outer_fine_before = _fine_apply_counter[]
            _amg_profile_time!(_profile, :outer_apply_preconditioned_vector) do
                _amg_mul_dirac!(w, D, Z[j]; fine_apply_counter=_fine_apply_counter)
            end
            _outer_fine_apply_counter[] += _fine_apply_counter[] - outer_fine_before

            # Modified Gram-Schmidt
            _amg_profile_time!(_profile, :outer_mgs) do
                for i = 1:j
                    H[i,j] = inner_product(V[i], w)
                    axpby!(-H[i,j], V[i], 1, w)
                end
                H[j+1,j] = sqrt(real(inner_product(w, w)))
            end
            _outer_vector_flop_counter[] +=
                j * (_amg_complex_dot_flops(nfine) + _amg_complex_axpby_flops(nfine)) +
                _amg_complex_norm_flops(nfine)

            if abs(H[j+1,j]) < 1e-14
                j_final = j
                for i = 1:(j-1)
                    tmp = cs[i] * H[i,j] + sn[i] * H[i+1,j]
                    H[i+1,j] = -conj(sn[i]) * H[i,j] + conj(cs[i]) * H[i+1,j]
                    H[i,j] = tmp
                end
                converged = true
                break
            end
            V[j+1] = w / H[j+1,j]
            _outer_vector_flop_counter[] += 2 * nfine

            for i = 1:(j-1)
                tmp = cs[i] * H[i,j] + sn[i] * H[i+1,j]
                H[i+1,j] = -conj(sn[i]) * H[i,j] + conj(cs[i]) * H[i+1,j]
                H[i,j] = tmp
            end

            denom = sqrt(abs(H[j,j])^2 + abs(H[j+1,j])^2)
            cs[j] = conj(H[j,j]) / denom
            sn[j] = H[j+1,j] / denom
            H[j,j] = denom
            H[j+1,j] = zero(ComplexF64)

            g[j+1] = -conj(sn[j]) * g[j]
            g[j] = cs[j] * g[j]

            res_norm = abs(g[j+1]) / b_norm
            if verbose
                println(" maxiter = $cycle, restart = $j, relative residual = $res_norm")
            end

            if res_norm < tol
                j_final = j
                converged = true
                break
            end
        end

        y = zeros(ComplexF64, j_final)
        for i = j_final:-1:1
            y[i] = g[i]
            for k = (i+1):j_final
                y[i] -= H[i,k] * y[k]
            end
            y[i] /= H[i,i]
        end

        # Update solution: x += Z * y.
        _amg_profile_time!(_profile, :outer_solution_update) do
            for k = 1:j_final
                axpby!(y[k], Z[k], 1, x)
            end
        end
        _outer_vector_flop_counter[] += j_final * _amg_complex_axpby_flops(nfine)

        if converged
            if verbose
                println("FGMRES converged at cycle $cycle, step $(j_final)!")
                _amg_print_solver_counts(setup, total_coarse_gmres,
                                         _coarse_apply_counter[],
                                         _fine_apply_counter[],
                                         _variable_coarse_apply_counter[],
                                         _variable_fine_apply_counter[],
                                         _variable_vector_flop_counter[],
                                         _outer_fine_apply_counter[],
                                         _outer_vector_flop_counter[])
                _profile !== nothing && _amg_print_profile(stdout, _profile)
            end
            return x
        end
    end

    if verbose
        r_final = similar(b)
        outer_fine_before = _fine_apply_counter[]
        _amg_mul_dirac!(r_final, D, x; fine_apply_counter=_fine_apply_counter)
        axpby!(1, b, -1, r_final)
        _outer_fine_apply_counter[] += _fine_apply_counter[] - outer_fine_before
        _outer_vector_flop_counter[] += _amg_complex_axpby_flops(nfine) + _amg_complex_norm_flops(nfine)
        println("FGMRES: max iterations reached. " *
                "Final relative residual = $(sqrt(real(inner_product(r_final, r_final)))/b_norm)")
        _amg_print_solver_counts(setup, total_coarse_gmres,
                                 _coarse_apply_counter[],
                                 _fine_apply_counter[],
                                 _variable_coarse_apply_counter[],
                                 _variable_fine_apply_counter[],
                                 _variable_vector_flop_counter[],
                                 _outer_fine_apply_counter[],
                                 _outer_vector_flop_counter[])
        _profile !== nothing && _amg_print_profile(stdout, _profile)
    end
    return x
end

function _amg_solve_batched_gmres_y!(yhost, Hhost, betahost,
                                     m::Int, nblocks::Int,
                                     active_blocks::Vector{Bool})
    mplus = m + 1
    fill!(yhost, zero(ComplexF64))
    @inbounds for bid in 1:nblocks
        active_blocks[bid] || continue
        β = betahost[bid]
        β > 1.0e-30 || continue
        Hloc = Matrix{ComplexF64}(undef, mplus, m)
        for col in 1:m, row in 1:mplus
            Hloc[row, col] = Hhost[row + (col - 1) * mplus + (bid - 1) * mplus * m]
        end
        rhs = zeros(ComplexF64, mplus)
        rhs[1] = β
        try
            y = Hloc \ rhs
            for k in 1:m
                yhost[k + (bid - 1) * m] = y[k]
            end
        catch
        end
    end
    return yhost
end

function _sap_smooth_gpu_gmres_batched!(x::WF3x1GPU, D, b::WF3x1GPU,
                                        setup::DDalphaAMG_Setup;
                                        niter::Int=1, nblock_gmres::Int=4,
                                        sap_schedule::Symbol=:boundary8,
                                        fine_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                                        flop_counter::Union{Nothing,Ref{Int}}=nothing)
    cache = _ensure_gpu_cache!(setup)
    bi = x.blockinfo
    total = Int32(length(x.f))
    nc_ng = Int32(size(x.f, 1) * size(x.f, 2))
    threads = 256
    grid = cld(Int(total), threads)
    block_grid = cld(setup.Nblocks, threads)
    NBX = Int32(setup.NBX); NBY = Int32(setup.NBY); NBZ = Int32(setup.NBZ); NBT = Int32(setup.NBT)
    schedule_code = _amg_sap_schedule_code(sap_schedule)
    m = Int32(nblock_gmres)
    mplus = Int32(nblock_gmres + 1)

    δ = similar(x)
    res = similar(x)
    Ddelta = similar(x)
    w = similar(x)
    V = [similar(x) for _ in 1:(nblock_gmres + 1)]
    H = CUDA.zeros(ComplexF64, (nblock_gmres + 1) * nblock_gmres * setup.Nblocks)
    beta = CUDA.zeros(Float64, setup.Nblocks)
    dotbuf = CUDA.zeros(ComplexF64, setup.Nblocks)
    coeff = CUDA.zeros(ComplexF64, setup.Nblocks)
    ydev = CUDA.zeros(ComplexF64, nblock_gmres * setup.Nblocks)

    active_by_step = [
        [_amg_sap_block_step_from_bid(bid, setup.NBX, setup.NBY, setup.NBZ, setup.NBT,
                                      Int(schedule_code)) == step
         for bid in 1:setup.Nblocks]
        for step in _amg_sap_schedule_steps(sap_schedule)
    ]

    for _ in 1:niter
        _amg_mul_dirac!(res, D, x; fine_apply_counter=fine_apply_counter)
        axpby!(1, b, -1, res)
        _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))

        for (step_index, step_i) in enumerate(_amg_sap_schedule_steps(sap_schedule))
            step = Int32(step_i)
            fill!(H, zero(ComplexF64))
            fill!(ydev, zero(ComplexF64))

            @cuda threads=threads blocks=grid _amg_init_color_fields_kernel!(
                reshape(δ.f, :), reshape(V[1].f, :), reshape(res.f, :),
                cache[:bmap], nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code)

            @cuda threads=threads blocks=setup.Nblocks shmem=(threads * sizeof(ComplexF64)) _amg_color_block_dot_kernel!(
                dotbuf, reshape(V[1].f, :), reshape(V[1].f, :),
                cache[:block_sites], cache[:block_counts], nc_ng, step, NBX, NBY, NBZ, NBT, schedule_code)
            @cuda threads=threads blocks=block_grid _amg_color_invnorm_from_vec_kernel!(
                coeff, beta, dotbuf, step, NBX, NBY, NBZ, NBT, schedule_code)
            @cuda threads=threads blocks=grid _amg_color_block_axpby_coeff_kernel!(
                reshape(V[1].f, :), reshape(V[1].f, :), coeff, cache[:bmap],
                nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code, ComplexF64(1.0), ComplexF64(0.0))

            for j in 1:nblock_gmres
                @cuda threads=bi.blocksize blocks=bi.rsize _amg_local_wilson_block_kernel!(
                    w.f, V[j].f, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
                    cache[:bmap], Float64(_amg_dirac_kappa(D)), bi, step, NBX, NBY, NBZ, NBT, schedule_code)

                for i in 1:j
                    @cuda threads=threads blocks=setup.Nblocks shmem=(threads * sizeof(ComplexF64)) _amg_color_block_dot_h_kernel!(
                        H, Int32(i), Int32(j), reshape(V[i].f, :), reshape(w.f, :),
                        cache[:block_sites], cache[:block_counts], nc_ng, step, NBX, NBY, NBZ, NBT, schedule_code,
                        mplus, m)
                    @cuda threads=threads blocks=grid _amg_color_block_axpby_h_kernel!(
                        reshape(w.f, :), reshape(V[i].f, :), H, Int32(i), Int32(j),
                        cache[:bmap], nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code,
                        mplus, m, ComplexF64(-1.0), ComplexF64(1.0))
                    _amg_count_flops!(flop_counter,
                        setup.Nblocks * (_amg_complex_dot_flops(setup.r^4 * setup.NC * setup.NG) +
                                         _amg_complex_axpby_flops(setup.r^4 * setup.NC * setup.NG)))
                end

                @cuda threads=threads blocks=setup.Nblocks shmem=(threads * sizeof(ComplexF64)) _amg_color_block_dot_h_kernel!(
                    H, Int32(j + 1), Int32(j), reshape(w.f, :), reshape(w.f, :),
                    cache[:block_sites], cache[:block_counts], nc_ng, step, NBX, NBY, NBZ, NBT, schedule_code,
                    mplus, m)
                @cuda threads=threads blocks=block_grid _amg_color_invnorm_from_h_kernel!(
                    coeff, H, Int32(j + 1), Int32(j), step, NBX, NBY, NBZ, NBT, schedule_code, mplus, m)
                _amg_count_flops!(flop_counter,
                    setup.Nblocks * _amg_complex_norm_flops(setup.r^4 * setup.NC * setup.NG))

                if j < nblock_gmres
                    @cuda threads=threads blocks=grid _amg_color_block_axpby_coeff_kernel!(
                        reshape(V[j + 1].f, :), reshape(w.f, :), coeff, cache[:bmap],
                        nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code, ComplexF64(1.0), ComplexF64(0.0))
                end
            end

            Hhost = Array(H)
            betahost = Array(beta)
            yhost = zeros(ComplexF64, nblock_gmres * setup.Nblocks)
            _amg_solve_batched_gmres_y!(yhost, Hhost, betahost,
                                        nblock_gmres, setup.Nblocks,
                                        active_by_step[step_index])
            copyto!(ydev, CuArray(yhost))
            fill!(δ.f, zero(ComplexF64))
            for k in 1:nblock_gmres
                @cuda threads=threads blocks=grid _amg_color_gmres_update_kernel!(
                    reshape(δ.f, :), reshape(V[k].f, :), ydev, Int32(k),
                    cache[:bmap], nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code, m)
                _amg_count_flops!(flop_counter,
                    setup.Nblocks * _amg_complex_axpby_flops(setup.r^4 * setup.NC * setup.NG))
            end

            @cuda threads=threads blocks=grid _amg_add_color_field_kernel!(
                reshape(x.f, :), reshape(δ.f, :), cache[:bmap],
                nc_ng, total, step, NBX, NBY, NBZ, NBT, schedule_code)
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))

            _amg_mul_dirac!(Ddelta, D, δ; fine_apply_counter=fine_apply_counter)
            @cuda threads=threads blocks=grid _amg_sub_field_kernel!(
                reshape(res.f, :), reshape(Ddelta.f, :))
            _amg_count_flops!(flop_counter, _amg_complex_axpby_flops(_amg_fine_dofs(x)))
        end
    end
    return x
end


# ==========================================
#  Full setup
# ==========================================

"""
    setup_DDalphaAMG(D, template, r, Nv; nsmooth, verbose, smoother, ...) -> DDalphaAMG_Setup

Build a complete DD-alphaAMG setup.

Steps:
1. Generate `Nv` random test vectors, unless `coarse_vectors` are provided.
2. Smooth them so they approximate low modes (`D*psi ~= 0`).
3. Block-orthonormalize the vectors.
4. Precompute the coarse operator.

Arguments:
- `D`: Dirac operator.
- `template`: WF3x1 field used for lattice/layout information.
- `r`: block size in each direction.
- `Nv`: number of test vectors.
- `setup_mode`: `:inverse_sap` keeps the validated paper-style setup;
  `:direct_near_null` restores the direct near-null setup used by the
  2026-07-14 comparison.
- `nsmooth`: setup smoothing iterations.
- `smoother`: `:mr`, `:sap`, or `:vcycle`.
- `nvcycle`: adaptive V-cycle setup iterations when `smoother=:vcycle`.
- `vcycle_pre_smooth`, `vcycle_post_smooth`: V-cycle smoothing counts.
- `setup_sap_iters`: red-black SAP sweeps used when setup smoothing uses SAP.
- `setup_block_solver`: block-local solver inside setup SAP sweeps. `:mr`
  matches the DD-alphaAMG paper default; `:gmres` keeps the stronger optional path.
- `setup_block_mr_steps`: block-local MR steps inside each setup SAP sweep.
- `setup_block_gmres_steps`: block-local GMRES steps inside each setup SAP sweep
  when `setup_block_solver=:gmres`.
- `vcycle_post_sap_iters`: red-black SAP sweeps used by the adaptive setup
  V-cycle post-smoother when `vcycle_update_smoother=:sap`.
- `vcycle_post_block_solver`: block-local solver inside each adaptive setup
  V-cycle SAP post-smoothing sweep.
- `vcycle_post_block_mr_steps`: block-local MR steps inside each adaptive setup
  V-cycle SAP post-smoothing sweep.
- `vcycle_post_block_gmres_steps`: block-local GMRES steps inside each adaptive setup
  V-cycle SAP post-smoothing sweep when `vcycle_post_block_solver=:gmres`.
- `setup_coarse_tol`: relative residual tolerance for coarse solves inside the
  adaptive setup V-cycles. The paper's Table 6.1 default is `5e-2`.
- `setup_coarse_maxiter`: maximum restarted coarse-GMRES cycles inside adaptive
  setup.
- `setup_coarse_restart`: restart length for coarse GMRES inside adaptive setup.
  The paper uses restart length 30 for coarse systems.
- `sap_schedule`: SAP block-list ordering. `:boundary8` uses the boundary-aware
  eight-stage DD-alphaAMG method-2 ordering; `:twocolor` keeps the older two-color
  red-black ordering.
- `use_chirality`: follow the paper's chiral coarsening: split each prototype
  into upper/lower chiral components, yielding `2*Nv` coarse vectors. In this
  basis `P' * gamma5 = sigma3 * P'`, so the gamma5 factors cancel in the
  correction and the implemented restriction remains `R = P'`.
- `precompute_coarse`: precompute the coarse operator stencil. Solves need this,
  but matrix-free training losses can disable it.
- `smooth_external_vectors`: when `coarse_vectors` are provided, apply the
  mode-specific setup smoothing before chiral splitting and block
  orthonormalization.

Example:
```julia
setup = setup_DDalphaAMG(D, x, 2, 24; nsmooth=16)
solution = solve_DDalphaAMG(D, b, setup; tol=1e-10)
```
"""
function setup_DDalphaAMG(D, template::WF3x1, r::Int, Nv::Int;
                          setup_mode::Symbol=:inverse_sap,
                          nsmooth::Int=16, verbose::Bool=true,
                          coarse_vectors::Union{Nothing,Vector{<:WF3x1}}=nothing,
                          smoother::Symbol=:mr,
                          nvcycle::Int=4,
                          vcycle_pre_smooth::Int=4, vcycle_post_smooth::Int=4,
                          setup_smoother::Symbol=:mr,
                          setup_sap_iters::Int=1,
                          setup_block_solver::Symbol=:mr,
                          setup_block_mr_steps::Int=4,
                          setup_block_gmres_steps::Int=setup_block_mr_steps,
                          vcycle_update_smoother::Symbol=:sap,
                          vcycle_post_sap_iters::Union{Nothing,Int}=nothing,
                          vcycle_post_block_solver::Symbol=setup_block_solver,
                          vcycle_post_block_mr_steps::Int=4,
                          vcycle_post_block_gmres_steps::Int=vcycle_post_block_mr_steps,
                          setup_coarse_tol::Union{Nothing,Float64}=nothing,
                          setup_coarse_maxiter::Union{Nothing,Int}=nothing,
                          setup_coarse_restart::Union{Nothing,Int}=nothing,
                          sap_schedule::Union{Nothing,Symbol}=nothing,
                          use_chirality::Bool=false,
                          precompute_coarse::Bool=true,
                          block_normalize::Bool=true,
                          smooth_external_vectors::Bool=true)
    _amg_validate_setup_mode(setup_mode)
    direct_mode = setup_mode == :direct_near_null
    setup_defaults = _amg_setup_defaults(setup_mode)
    vcycle_post_sap_iters = something(
        vcycle_post_sap_iters, setup_defaults.vcycle_post_sap_iters)
    setup_coarse_tol = something(setup_coarse_tol, setup_defaults.coarse_tol)
    setup_coarse_maxiter = something(
        setup_coarse_maxiter, setup_defaults.coarse_maxiter)
    setup_coarse_restart = something(
        setup_coarse_restart, setup_defaults.coarse_restart)
    sap_schedule = something(sap_schedule, setup_defaults.sap_schedule)

    NX, NY, NZ, NT = template.NX, template.NY, template.NZ, template.NT
    NC, NG = template.NC, template.NG

    @assert NX % r == 0 "NX=$NX must be divisible by r=$r"
    @assert NY % r == 0 "NY=$NY must be divisible by r=$r"
    @assert NZ % r == 0 "NZ=$NZ must be divisible by r=$r"
    @assert NT % r == 0 "NT=$NT must be divisible by r=$r"

    Nv_smooth = Nv

    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r
    Nblocks = NBX * NBY * NBZ * NBT
    _amg_sap_schedule_code(sap_schedule)

    if verbose
        println("DD-alphaAMG Setup:")
        println("  setup mode: $setup_mode")
        println("  fine lattice: $(NX)x$(NY)x$(NZ)x$(NT)")
        println("  block size: $(r)^4")
        println("  coarse lattice: $(NBX)x$(NBY)x$(NBZ)x$(NBT) = $(Nblocks) blocks")
        println("  test vectors: $Nv", use_chirality ? " prototypes -> $(2 * Nv) chiral coarse vectors" : "")
        println("  coarse dof: $(Nblocks * (use_chirality ? 2 * Nv : Nv))")
        println("  restriction: R = P'")
    end

    # 1. Generate or load test vectors.
    if coarse_vectors !== nothing
        @assert length(coarse_vectors) == Nv "length(coarse_vectors) must match Nv"
        # Keep caller-provided vectors immutable from the setup's point of view.
        # setup_DDalphaAMG performs in-place block orthonormalization, so aliasing
        # the input vector array here corrupts later uses such as deflated_initial_guess(D, b, v).
        test_vectors = _copy_wf_vector(coarse_vectors)
        if verbose
            println("  using externally provided test vectors")
        end
        if smooth_external_vectors && nsmooth > 0
            setup_smoother in (:mr, :sap) ||
                throw(ArgumentError("setup_smoother must be :mr or :sap, got $setup_smoother"))
            if setup_smoother == :sap
                setup_sap_iters > 0 ||
                    throw(ArgumentError("setup_sap_iters must be positive, got $setup_sap_iters"))
                setup_block_solver in (:mr, :gmres) ||
                    throw(ArgumentError("setup_block_solver must be :mr or :gmres, got $setup_block_solver"))
                setup_block_mr_steps > 0 ||
                    throw(ArgumentError("setup_block_mr_steps must be positive, got $setup_block_mr_steps"))
                setup_block_gmres_steps > 0 ||
                    throw(ArgumentError("setup_block_gmres_steps must be positive, got $setup_block_gmres_steps"))
                block_steps = setup_block_solver == :mr ? setup_block_mr_steps : setup_block_gmres_steps
                block_name = setup_block_solver == :mr ? "block-MR" : "block-GMRES"
                if verbose
                    if direct_mode
                        println("  homogeneous-smoothing external test vectors with SAP ($(setup_sap_iters) sweeps, $(block_steps) $(block_name) steps)...")
                    else
                        println("  inverse-smoothing external test vectors with SAP stages 1:$(setup_sap_iters) ($(block_steps) $(block_name) steps per local block solve)...")
                    end
                end
                sap_setup = DDalphaAMG_Setup{typeof(template)}(
                    test_vectors, r, Nv_smooth, NX, NY, NZ, NT, NC, NG,
                    NBX, NBY, NBZ, NBT, Nblocks, nothing,
                    nothing,
                    CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing
                )
                if direct_mode
                    _amg_initial_direct_smooth!(test_vectors, D, sap_setup;
                                                smoother=:sap,
                                                niter=setup_sap_iters,
                                                block_solver=setup_block_solver,
                                                nblock_mr=setup_block_mr_steps,
                                                nblock_gmres=setup_block_gmres_steps,
                                                sap_schedule=sap_schedule)
                else
                    _amg_initial_inverse_smooth!(test_vectors, D, sap_setup;
                                                 smoother=:sap,
                                                 nstages=setup_sap_iters,
                                                 block_solver=setup_block_solver,
                                                 nblock_mr=setup_block_mr_steps,
                                                 nblock_gmres=setup_block_gmres_steps,
                                                 sap_schedule=sap_schedule)
                end
            else
                if verbose
                    if direct_mode
                        println("  homogeneous-smoothing external test vectors with MR ($nsmooth iterations)...")
                    else
                        println("  inverse-smoothing external test vectors with MR stages 1:$nsmooth...")
                    end
                end
                external_setup = DDalphaAMG_Setup{typeof(template)}(
                    test_vectors, r, Nv_smooth, NX, NY, NZ, NT, NC, NG,
                    NBX, NBY, NBZ, NBT, Nblocks, nothing,
                    nothing,
                    CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing
                )
                if direct_mode
                    _amg_initial_direct_smooth!(test_vectors, D, external_setup;
                                                smoother=:mr,
                                                niter=nsmooth)
                else
                    _amg_initial_inverse_smooth!(test_vectors, D, external_setup;
                                                 smoother=:mr,
                                                 nstages=nsmooth)
                end
            end
        end
    else
        test_vectors = [similar(template) for _ in 1:Nv_smooth]
        for v in 1:Nv_smooth
            random_fermion!(test_vectors[v])
        end

        # 2. Smooth test vectors so they approximate low modes.
        zero_rhs = similar(template)
        zero_wf!(zero_rhs)

        if smoother == :vcycle
            _setup_fine_apply_counter = Ref(0)
            # Build near-null seeds and refine them with the selected adaptive
            # setup algorithm.
            sap_setup = DDalphaAMG_Setup{typeof(template)}(
                test_vectors, r, Nv_smooth, NX, NY, NZ, NT, NC, NG,
                NBX, NBY, NBZ, NBT, Nblocks, nothing,
                nothing,
                CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing
            )
            setup_smoother in (:mr, :sap) ||
                throw(ArgumentError("setup_smoother must be :mr or :sap, got $setup_smoother"))
            vcycle_update_smoother in (:mr, :sap) ||
                throw(ArgumentError("vcycle_update_smoother must be :mr or :sap, got $vcycle_update_smoother"))
            setup_block_solver in (:mr, :gmres) ||
                throw(ArgumentError("setup_block_solver must be :mr or :gmres, got $setup_block_solver"))
            vcycle_post_block_solver in (:mr, :gmres) ||
                throw(ArgumentError("vcycle_post_block_solver must be :mr or :gmres, got $vcycle_post_block_solver"))
            setup_sap_iters > 0 ||
                throw(ArgumentError("setup_sap_iters must be positive, got $setup_sap_iters"))
            setup_block_mr_steps > 0 ||
                throw(ArgumentError("setup_block_mr_steps must be positive, got $setup_block_mr_steps"))
            setup_block_gmres_steps > 0 ||
                throw(ArgumentError("setup_block_gmres_steps must be positive, got $setup_block_gmres_steps"))
            vcycle_post_sap_iters > 0 ||
                throw(ArgumentError("vcycle_post_sap_iters must be positive, got $vcycle_post_sap_iters"))
            vcycle_post_block_mr_steps > 0 ||
                throw(ArgumentError("vcycle_post_block_mr_steps must be positive, got $vcycle_post_block_mr_steps"))
            vcycle_post_block_gmres_steps > 0 ||
                throw(ArgumentError("vcycle_post_block_gmres_steps must be positive, got $vcycle_post_block_gmres_steps"))
            setup_coarse_tol > 0 ||
                throw(ArgumentError("setup_coarse_tol must be positive, got $setup_coarse_tol"))
            setup_coarse_maxiter > 0 ||
                throw(ArgumentError("setup_coarse_maxiter must be positive, got $setup_coarse_maxiter"))
            setup_coarse_restart > 0 ||
                throw(ArgumentError("setup_coarse_restart must be positive, got $setup_coarse_restart"))
            if verbose
                setup_label = direct_mode ? "direct near-null setup" : "inverse-SAP setup"
                if setup_smoother == :sap
                    setup_block_steps = setup_block_solver == :mr ? setup_block_mr_steps : setup_block_gmres_steps
                    setup_block_name = setup_block_solver == :mr ? "block-MR" : "block-GMRES"
                    if direct_mode
                        println("  [direct near-null setup] Phase 1: homogeneous SAP ($(setup_sap_iters) sweeps, $(setup_block_steps) $(setup_block_name) steps)...")
                    else
                        println("  [inverse-SAP setup] Phase 1: inverse SAP stages 1:$(setup_sap_iters) ($(setup_block_steps) $(setup_block_name) steps per local block solve)...")
                    end
                else
                    if direct_mode
                        println("  [direct near-null setup] Phase 1: homogeneous MR ($nsmooth iterations)...")
                    else
                        println("  [inverse-SAP setup] Phase 1: inverse MR stages 1:$nsmooth...")
                    end
                end
                if vcycle_update_smoother == :sap
                    post_block_steps = vcycle_post_block_solver == :mr ? vcycle_post_block_mr_steps : vcycle_post_block_gmres_steps
                    post_block_name = vcycle_post_block_solver == :mr ? "block-MR" : "block-GMRES"
                    println("  [$setup_label] adaptive post-smoother: SAP ($(vcycle_post_sap_iters) sweeps, $(post_block_steps) $(post_block_name) steps)")
                else
                    println("  [$setup_label] adaptive post-smoother: MR ($vcycle_post_smooth steps)")
                end
                println("  [$setup_label] adaptive coarse solve: tol=$(setup_coarse_tol), restart=$(setup_coarse_restart), max cycles=$(setup_coarse_maxiter)")
            end
            initial_steps = setup_smoother == :sap ? setup_sap_iters : nsmooth
            if direct_mode
                _amg_initial_direct_smooth!(test_vectors, D, sap_setup;
                                            smoother=setup_smoother,
                                            niter=initial_steps,
                                            block_solver=setup_block_solver,
                                            nblock_mr=setup_block_mr_steps,
                                            nblock_gmres=setup_block_gmres_steps,
                                            sap_schedule=sap_schedule,
                                            fine_apply_counter=_setup_fine_apply_counter)
            else
                _amg_initial_inverse_smooth!(test_vectors, D, sap_setup;
                                             smoother=setup_smoother,
                                             nstages=initial_steps,
                                             block_solver=setup_block_solver,
                                             nblock_mr=setup_block_mr_steps,
                                             nblock_gmres=setup_block_gmres_steps,
                                             sap_schedule=sap_schedule,
                                             fine_apply_counter=_setup_fine_apply_counter)
            end

            r_fine = direct_mode ? similar(template) : nothing
            correction = direct_mode ? similar(template) : nothing

            for iter in 1:nvcycle
                cur_vectors = _copy_wf_vector(test_vectors)
                if use_chirality
                    cur_vectors = _chiral_split_vectors(cur_vectors)
                    _block_orthonormalize_chiral_split!(cur_vectors, r; normalize=block_normalize)
                else
                    block_orthonormalize!(cur_vectors, r; normalize=block_normalize)
                end
                cur_Nv = length(cur_vectors)
                cur_setup = DDalphaAMG_Setup{typeof(template)}(
                    cur_vectors, r, cur_Nv, NX, NY, NZ, NT, NC, NG,
                    NBX, NBY, NBZ, NBT, Nblocks, nothing,
                    nothing,
                    CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing,
                    setup_mode
                )
                precompute_coarse_operator!(cur_setup, D; mode=:direct_stencil)
                r_coarse = direct_mode ? _coarse_vector(cur_Nv, Nblocks, template) : nothing

                for v in 1:Nv_smooth
                    if direct_mode
                        _amg_direct_near_null_update!(
                            test_vectors[v], D, cur_setup,
                            r_fine::WF3x1, r_coarse::CoarseVector,
                            correction::WF3x1;
                            update_smoother=vcycle_update_smoother,
                            post_sap_iters=vcycle_post_sap_iters,
                            post_smooth=vcycle_post_smooth,
                            block_solver=vcycle_post_block_solver,
                            nblock_mr=vcycle_post_block_mr_steps,
                            nblock_gmres=vcycle_post_block_gmres_steps,
                            coarse_tol=setup_coarse_tol,
                            coarse_maxiter=setup_coarse_maxiter,
                            coarse_restart=setup_coarse_restart,
                            sap_schedule=sap_schedule,
                            arnoldi_mode=:scalar,
                            fine_apply_counter=_setup_fine_apply_counter)
                    else
                        updated = vcycle(D, test_vectors[v], cur_setup;
                                         pre_smooth=0,
                                         post_smooth=(vcycle_update_smoother == :sap ?
                                                      vcycle_post_sap_iters : vcycle_post_smooth),
                                         sap_block_mr_steps=vcycle_post_block_mr_steps,
                                         coarse_maxiter=setup_coarse_maxiter,
                                         coarse_restart=setup_coarse_restart,
                                         coarse_tol=setup_coarse_tol,
                                         smoother=vcycle_update_smoother,
                                         coarse_odd_even=true,
                                         sap_odd_even=true,
                                         sap_schedule=sap_schedule,
                                         fine_apply_counter=_setup_fine_apply_counter)
                        LatticeDiracOperators.substitute_fermion!(test_vectors[v], updated)
                        normalize_fermion!(test_vectors[v])
                    end
                end

                if verbose
                    setup_label = direct_mode ? "direct near-null setup" : "inverse-SAP setup"
                    println("  [$setup_label] iteration $iter/$nvcycle done")
                end
            end
            if verbose
                setup_label = direct_mode ? "direct near-null setup" : "inverse-SAP setup"
                println("  [$setup_label] fine Dirac applications: $(_setup_fine_apply_counter[])")
            end
        elseif smoother == :sap
            setup_block_solver in (:mr, :gmres) ||
                throw(ArgumentError("setup_block_solver must be :mr or :gmres, got $setup_block_solver"))
            setup_sap_iters > 0 ||
                throw(ArgumentError("setup_sap_iters must be positive, got $setup_sap_iters"))
            setup_block_mr_steps > 0 ||
                throw(ArgumentError("setup_block_mr_steps must be positive, got $setup_block_mr_steps"))
            setup_block_gmres_steps > 0 ||
                throw(ArgumentError("setup_block_gmres_steps must be positive, got $setup_block_gmres_steps"))
            if verbose
                setup_block_steps = setup_block_solver == :mr ? setup_block_mr_steps : setup_block_gmres_steps
                setup_block_name = setup_block_solver == :mr ? "block-MR" : "block-GMRES"
                println("  smoothing test vectors with SAP ($(setup_sap_iters) sweeps, $(setup_block_steps) $(setup_block_name) steps)...")
            end
            # SAP needs a temporary setup for block-local smoothing.
            block_orthonormalize!(test_vectors, r; normalize=block_normalize)
            pre_setup = DDalphaAMG_Setup{typeof(template)}(
                test_vectors, r, Nv_smooth, NX, NY, NZ, NT, NC, NG,
                NBX, NBY, NBZ, NBT, Nblocks, nothing,
                nothing,
                CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing
            )
            for v in 1:Nv_smooth
                sap_smooth!(test_vectors[v], D, zero_rhs, pre_setup;
                            niter=setup_sap_iters,
                            block_solver=setup_block_solver,
                            nblock_mr=setup_block_mr_steps,
                            nblock_gmres=setup_block_gmres_steps,
                            sap_schedule=sap_schedule)
            end
        else  # :mr (default)
            if verbose
                println("  smoothing test vectors with MR ($nsmooth iterations)...")
            end
            for v in 1:Nv_smooth
                mr_smooth!(test_vectors[v], D, zero_rhs; niter=nsmooth)
            end
        end
    end

    if use_chirality
        test_vectors = _chiral_split_vectors(test_vectors)
        if verbose
            println("  chiral split: $Nv prototypes -> $(length(test_vectors)) coarse vectors")
        end
    end
    Nv_setup = length(test_vectors)

    # 3. Block orthogonalization / orthonormalization
    if verbose
        println(block_normalize ? "  block orthonormalizing..." : "  block orthogonalizing without normalization...")
    end
    if use_chirality
        _block_orthonormalize_chiral_split!(test_vectors, r; normalize=block_normalize)
    else
        block_orthonormalize!(test_vectors, r; normalize=block_normalize)
    end

    setup = DDalphaAMG_Setup{typeof(template)}(
        test_vectors, r, Nv_setup, NX, NY, NZ, NT, NC, NG,
        NBX, NBY, NBZ, NBT, Nblocks, nothing,
        nothing,
        CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing,
        setup_mode
    )

    # Precompute the coarse operator for AMG solves. Matrix-free training losses
    # can skip this expensive assembly.
    if precompute_coarse
        precompute_coarse_operator!(setup, D; verbose=verbose, mode=:direct_stencil)
    end

    if verbose
        println("  setup done")
    end

    return setup
end


# ==========================================
#  Full solve
# ==========================================

"""
    solve_DDalphaAMG(D, b, setup; x0, tol, maxiter, restart, verbose, smoother, ...)

Solve `D*x = b` using DD-alphaAMG: FGMRES as the outer solver and a V-cycle as
the preconditioner.

Arguments:
- `D`: Dirac operator.
- `b`: right-hand side vector.
- `setup`: `DDalphaAMG_Setup` from `setup_DDalphaAMG`.
- `tol`: relative residual tolerance.
- `smoother`: `:mr` or `:sap` for fine-grid smoothing.
- `pre_smooth`, `post_smooth`: smoothing counts inside the V-cycle.

Solve options are independent of `setup.setup_mode`. Explicit keyword values
always take precedence over the common solver defaults.
"""
function solve_DDalphaAMG(D, b::WF3x1, setup::DDalphaAMG_Setup;
                          x0::Union{Nothing,WF3x1}=nothing,
                          tol::Float64=1e-10, maxiter::Int=100,
                          restart::Union{Nothing,Int}=nothing,
                          verbose::Bool=true,
                          smoother::Union{Nothing,Symbol}=nothing,
                          pre_smooth::Union{Nothing,Int}=nothing,
                          post_smooth::Union{Nothing,Int}=nothing,
                          sap_block_mr_steps::Int=4,
                          coarse_tol::Union{Nothing,Float64}=nothing,
                          coarse_maxiter::Union{Nothing,Int}=nothing,
                          coarse_restart::Union{Nothing,Int}=nothing,
                          coarse_odd_even::Union{Nothing,Bool}=nothing,
                          coarse_gmres_mode::Union{Nothing,Symbol}=nothing,
                          sap_odd_even::Union{Nothing,Bool}=nothing,
                          sap_schedule::Union{Nothing,Symbol}=nothing,
                          profile::Bool=false,
                          profile_synchronize::Bool=true)
    options = _amg_resolve_solve_options(;
        restart=restart,
        smoother=smoother,
        pre_smooth=pre_smooth,
        post_smooth=post_smooth,
        coarse_tol=coarse_tol,
        coarse_maxiter=coarse_maxiter,
        coarse_restart=coarse_restart,
        coarse_odd_even=coarse_odd_even,
        coarse_gmres_mode=coarse_gmres_mode,
        sap_odd_even=sap_odd_even,
        sap_schedule=sap_schedule)
    if verbose
        println("DD-alphaAMG Solve settings:")
        println("  setup mode: $(setup.setup_mode)")
        println("  outer: tol=$tol, max cycles=$maxiter, restart=$(options.restart)")
        println("  fine smoother: $(options.smoother), pre=$(options.pre_smooth), post=$(options.post_smooth), block MR steps=$sap_block_mr_steps")
        println("  SAP: schedule=$(options.sap_schedule), odd-even=$(options.sap_odd_even)")
        println("  coarse: tol=$(options.coarse_tol), max cycles=$(options.coarse_maxiter), restart=$(options.coarse_restart), odd-even=$(options.coarse_odd_even), GMRES mode=$(options.coarse_gmres_mode)")
    end
    return fgmres(D, b, setup;
                  x0=x0, tol=tol, maxiter=maxiter, restart=options.restart,
                  verbose=verbose, smoother=options.smoother,
                  pre_smooth=options.pre_smooth, post_smooth=options.post_smooth,
                  sap_block_mr_steps=sap_block_mr_steps,
                  coarse_tol=options.coarse_tol,
                  coarse_maxiter=options.coarse_maxiter,
                  coarse_restart=options.coarse_restart,
                  coarse_odd_even=options.coarse_odd_even,
                  coarse_gmres_mode=options.coarse_gmres_mode,
                  sap_odd_even=options.sap_odd_even,
                  sap_schedule=options.sap_schedule,
                  profile=profile,
                  profile_synchronize=profile_synchronize)
end

# ==========================================
#  Deflation initial guess
# ==========================================
"""
    deflated_initial_guess(D, b, test_vectors; verbose=false)

Build a deflated initial guess from test vectors. It forms
`G[i,j] = <v_i, D*v_j>` and `rhs[i] = <v_i, b>`, solves `G*c = rhs`, and returns
`x0 = sum_j c_j * v_j`.
"""
function _assert_same_wf_layout(label_a, a::WF3x1, label_b, b::WF3x1)
    typeof(a) == typeof(b) ||
        throw(ArgumentError("$label_a and $label_b have different field types: $(typeof(a)) vs $(typeof(b))"))
    size(a.f) == size(b.f) ||
        throw(ArgumentError("$label_a and $label_b have different storage sizes: $(size(a.f)) vs $(size(b.f))"))
    a.NX == b.NX && a.NY == b.NY && a.NZ == b.NZ && a.NT == b.NT ||
        throw(ArgumentError("$label_a and $label_b have different lattice sizes: " *
                            "$(a.NX)x$(a.NY)x$(a.NZ)x$(a.NT) vs $(b.NX)x$(b.NY)x$(b.NZ)x$(b.NT)"))
    a.NC == b.NC && a.NG == b.NG ||
        throw(ArgumentError("$label_a and $label_b have different component sizes: " *
                            "NC/NG=$(a.NC)/$(a.NG) vs $(b.NC)/$(b.NG)"))
    if a.f isa CuArray
        a.blockinfo.blocksize == b.blockinfo.blocksize &&
        a.blockinfo.rsize == b.blockinfo.rsize ||
            throw(ArgumentError("$label_a and $label_b have different GPU block layouts: " *
                                "$(a.blockinfo.blocksize)x$(a.blockinfo.rsize) vs " *
                                "$(b.blockinfo.blocksize)x$(b.blockinfo.rsize)"))
    end
    return nothing
end

function deflated_initial_guess(D, b::WF3x1, test_vectors::Vector{T};
                                 verbose::Bool=false) where T<:WF3x1
    Nv = length(test_vectors)
    Nv == 0 && return similar(b)
    _assert_same_wf_layout("test_vectors[1]", test_vectors[1], "b", b)
    for i in 2:Nv
        _assert_same_wf_layout("test_vectors[$i]", test_vectors[i], "test_vectors[1]", test_vectors[1])
    end

    G   = zeros(ComplexF64, Nv, Nv)
    rhs = zeros(ComplexF64, Nv)

    Dv = similar(b)
    for j in 1:Nv
        _amg_mul_dirac!(Dv, D, test_vectors[j])
        for i in 1:Nv
            G[i,j] = inner_product(test_vectors[i], Dv)
        end
        rhs[j] = inner_product(test_vectors[j], b)
    end

    # Solve G*c = rhs.
    c = G \ rhs

    # x0 = Σ c_j v_j
    x0 = similar(b)
    zero_wf!(x0)
    for j in 1:Nv
        axpby!(c[j], test_vectors[j], 1, x0)
    end

    return x0
end


end # module DDalphaAMG
