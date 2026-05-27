module DDalphaAMG

using ..elementry_base
using ..gpu
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
export gamma5!, chiral_project!, gamma5_sign
export compare_ddalphaamg_cpu_gpu


const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

# ==========================================
#  邊玲ｼ蟄舌・繧ｯ繝医Ν (Coarse Grid Vector)
# ==========================================

"""
    CoarseVector

邊玲ｼ蟄蝉ｸ翫・繝吶け繝医Ν縲・
data[(bidx-1)*Nv + v] 縺ｧ繝悶Ο繝・け bidx 縺ｮ繝・せ繝医・繧ｯ繝医Ν逡ｪ蜿ｷ v 縺ｮ謌仙・縺ｫ繧｢繧ｯ繧ｻ繧ｹ縲・
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

function coarse_axpby!(ﾎｱ, x::CoarseVector, ﾎｲ, y::CoarseVector)
    if y.data isa CuArray
        y.data .= ﾎｱ .* x.data .+ ﾎｲ .* y.data
        return y
    end
    @inbounds @simd for i in eachindex(y.data)
        y.data[i] = ﾎｱ * x.data[i] + ﾎｲ * y.data[i]
    end
    return y
end


# ==========================================
#  繧ｻ繝・ヨ繧｢繝・・讒矩菴・(Setup Structure)
# ==========================================

"""
    DDalphaAMG_Setup

DD-ﾎｱAMG 縺ｮ繧ｻ繝・ヨ繧｢繝・・諠・ｱ繧剃ｿ晄戟縺吶ｋ讒矩菴薙・
- test_vectors: 繝悶Ο繝・け逶ｴ莠､豁｣隕丞喧縺輔ｌ縺溘ユ繧ｹ繝医・繧ｯ繝医Ν
- r: 蜷・婿蜷代・繝悶Ο繝・け繧ｵ繧､繧ｺ
- Nv: 繝・せ繝医・繧ｯ繝医Ν謨ｰ
- NBX,NBY,NBZ,NBT: 邊玲ｼ蟄舌・蜷・婿蜷代ヶ繝ｭ繝・け謨ｰ
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
end


"""
    CoarseStencilOperator

Coarse operator stored in block-stencil form on coarse grid DOFs:
- onsite block: self[:, :, b]
- nearest-neighbor links: plus[:, :, b, ﾎｼ], minus[:, :, b, ﾎｼ]  (ﾎｼ=1..4)

Each block matrix has size Nvﾃ湧v, and total coarse size is Nv*Nblocks.
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

        # +/竏・nearest neighbors: dst = nb_plus/minus(bs, mu)
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


# ==========================================
#  繝悶Ο繝・け繝ｦ繝ｼ繝・ぅ繝ｪ繝・ぅ
# ==========================================

@inline function block_linear_index(bx, by, bz, bt, NBX, NBY, NBZ)
    return (bt-1)*NBZ*NBY*NBX + (bz-1)*NBY*NBX + (by-1)*NBX + bx
end

function _ensure_gpu_cache!(setup)
    setup.test_vectors[1] isa WF3x1GPU || return nothing
    cache = hasproperty(setup, :_gpu_cache) ? setup._gpu_cache : nothing
    if cache isa Dict && get(cache, :r, nothing) == setup.r
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

function _amg_restrict_one_kernel!(out, tv, x, block_sites, block_counts, nc_ng::Int32, Nv::Int32, v::Int32)
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

function _amg_block_axpby_kernel!(y, x, bmap, nc_ng::Int32, total::Int32, bid_target::Int32, ﾎｱ, ﾎｲ)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        if bmap[site] == bid_target
            @inbounds y[idx] = ﾎｱ * x[idx] + ﾎｲ * y[idx]
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

function _amg_orthogonalize_kernel!(v, u, block_sites, block_counts, coeffs, nc_ng::Int32)
    bid = Int32(blockIdx().x)
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
        @inbounds acc += conj(u[fidx]) * v[fidx]
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
    coeff = sh[Int32(1)]
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

function _amg_init_color_fields_kernel!(delta, local_r, res, bmap,
                                        nc_ng::Int32, total::Int32, color::Int32,
                                        NBX::Int32, NBY::Int32, NBZ::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        use = Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color
        delta[idx] = zero(ComplexF64)
        local_r[idx] = use ? res[idx] : zero(ComplexF64)
    end
    return nothing
end

function _amg_add_color_field_kernel!(x, delta, bmap,
                                      nc_ng::Int32, total::Int32, color::Int32,
                                      NBX::Int32, NBY::Int32, NBZ::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        if Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color
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

function _amg_block_mr_dot_kernel!(num, den, Alr, local_r,
                                   block_sites, block_counts,
                                   nc_ng::Int32, color::Int32,
                                   NBX::Int32, NBY::Int32, NBZ::Int32)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh_num = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))
    sh_den = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz), Int(bsz) * sizeof(ComplexF64))
    acc_num = zero(ComplexF64)
    acc_den = zero(ComplexF64)
    if Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color
        total_sites = block_counts[bid]
        p = tid
        while p <= total_sites * nc_ng
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
                                      nc_ng::Int32, total::Int32, color::Int32,
                                      NBX::Int32, NBY::Int32, NBZ::Int32)
    idx = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if idx <= total
        site = (idx - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        if Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color
            d = den[bid]
            if abs(d) > 1e-30
                alpha = num[bid] / d
                delta[idx] += alpha * local_r[idx]
                local_r[idx] -= alpha * Alr[idx]
            end
        end
    end
    return nothing
end

function _amg_local_wilson_block_kernel!(yf, xf, U1, U2, U3, U4, bmap,
                                         κ::Float64, blockinfo,
                                         color::Int32, NBX::Int32, NBY::Int32, NBZ::Int32)
    b = Int64(threadIdx().x)
    r = Int64(blockIdx().x)
    site = Int32((r - 1) * blockinfo.blocksize + b)
    bid = bmap[site]
    active = Int32(_amg_block_color_from_bid(bid, NBX, NBY, NBZ)) == color

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

繝悶Ο繝・け(bx,by,bz,bt)螟悶・蜈ｨ繧ｵ繧､繝医ｒ繧ｼ繝ｭ縺ｫ縺吶ｋ (in-place)縲・
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
    block_copy!(dest, src, bx, by, bz, bt, r)

src 縺ｮ繝悶Ο繝・け(bx,by,bz,bt)蜀・・蛟､繧・dest 縺ｮ蜷後§菴咲ｽｮ縺ｫ繧ｳ繝斐・縺吶ｋ (in-place)縲・
dest 縺ｮ繝悶Ο繝・け螟悶・螟画峩縺励↑縺・・
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

f 縺ｮ繝悶Ο繝・け(bx,by,bz,bt)蜀・・蛟､繧偵ぞ繝ｭ縺ｫ縺吶ｋ (in-place)縲・
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
    block_axpby!(ﾎｱ, x, ﾎｲ, y, bx, by, bz, bt, r)

y[block] = ﾎｱ * x[block] + ﾎｲ * y[block] 繧偵ヶ繝ｭ繝・け蜀・〒縺ｮ縺ｿ螳溯｡・(in-place)縲・
"""
function block_axpby!(ﾎｱ, x::WF3x1, ﾎｲ, y::WF3x1, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
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
                                ﾎｱ * x[ic, ix, iy, iz, it, ig] + ﾎｲ * y[ic, ix, iy, iz, it, ig]
                        end
                    end
                end
            end
        end
    end
    return y
end

"""
    block_inner_product(x, y, bx, by, bz, bt, r) 竊・ComplexF64

繝悶Ο繝・け(bx,by,bz,bt)蜀・・縺ｿ縺ｮ蜀・ｩ・筺ｨx|y筺ｩ_B縲・
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

function block_axpby!(ﾎｱ, x::WF3x1GPU, ﾎｲ, y::WF3x1GPU, bx::Int, by::Int, bz::Int, bt::Int, r::Int)
    setup_like, cache = _gpu_block_context_from_field(y, r)
    nc_ng = Int32(size(y.f, 1) * size(y.f, 2))
    total = Int32(length(y.f))
    bid = _gpu_bid(setup_like, bx, by, bz, bt)
    @cuda threads=256 blocks=cld(Int(total), 256) _amg_block_axpby_kernel!(
        reshape(y.f, :), reshape(x.f, :), cache[:bmap], nc_ng, total, bid, ﾎｱ, ﾎｲ)
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
#  ﾎｳ竄・繝ｦ繝ｼ繝・ぅ繝ｪ繝・ぅ (Chiral Splitting)
# ==========================================

"""
ﾎｳ竄・縺ｮ隨ｦ蜿ｷ縲ゅ％縺ｮ繧ｳ繝ｼ繝峨〒縺ｮ convention:
  ﾎｳ竄・= diag(-1, -1, +1, +1)   (ig=1,2 竊・-1,  ig=3,4 竊・+1)
"""
@inline gamma5_sign(ig::Int) = ig <= 2 ? -1 : 1

"""
    gamma5!(x::WF3x1)

x 縺ｫ ﾎｳ竄・繧・in-place 縺ｧ驕ｩ逕ｨ: x 竊・ﾎｳ竄・x
"""
function gamma5!(x::WF3x1)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC, NG = x.NC, x.NG
    @inbounds for it=1:NT, iz=1:NZ, iy=1:NY, ix=1:NX
        for ic=1:NC
            # ig=1,2 竊・ﾃ・-1)
            x[ic,ix,iy,iz,it,1] = -x[ic,ix,iy,iz,it,1]
            x[ic,ix,iy,iz,it,2] = -x[ic,ix,iy,iz,it,2]
            # ig=3,4 縺ｯ ﾎｳ竄・+1 縺ｪ縺ｮ縺ｧ螟画峩荳崎ｦ・
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

繧ｫ繧､繝ｩ繝ｫ蟆・ｽｱ繧・in-place 縺ｧ驕ｩ逕ｨ:
  :plus  竊・P竄・= (1+ﾎｳ竄・/2: ig=3,4 繧剃ｿ晄戟, ig=1,2 繧偵ぞ繝ｭ蛹・
  :minus 竊・P竄・= (1-ﾎｳ竄・/2: ig=1,2 繧剃ｿ晄戟, ig=3,4 繧偵ぞ繝ｭ蛹・

Note: ﾎｳ竄・diag(-1,-1,+1,+1) 縺ｪ縺ｮ縺ｧ
  P竄・= diag(0,0,1,1),  P竄・= diag(1,1,0,0)
"""
function chiral_project!(x::WF3x1, chirality::Symbol)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NC = x.NC
    @inbounds for it=1:NT, iz=1:NZ, iy=1:NY, ix=1:NX
        for ic=1:NC
            if chirality == :plus
                # P竄・ zero out ig=1,2 (ﾎｳ竄・eigenvalue -1)
                x[ic,ix,iy,iz,it,1] = zero(ComplexF64)
                x[ic,ix,iy,iz,it,2] = zero(ComplexF64)
            else
                # P竄・ zero out ig=3,4 (ﾎｳ竄・eigenvalue +1)
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


# ==========================================
#  繝悶Ο繝・け逶ｴ莠､豁｣隕丞喧 (Block Orthonormalization)
# ==========================================

"""
    block_orthonormalize!(vectors, r)

蜷・ヶ繝ｭ繝・け蜀・〒繝・せ繝医・繧ｯ繝医Ν繧・Modified Gram-Schmidt 縺ｧ逶ｴ莠､豁｣隕丞喧縺吶ｋ縲・
蜃ｦ逅・ｾ・ 蜷・ヶ繝ｭ繝・けB縺ｫ縺､縺・※ 筺ｨﾏ違ｵ｢|ﾏ遺ｱｼ筺ｩ_B = ﾎｴ盞｢箜ｼ
"""
function block_orthonormalize!(vectors::Vector{T}, r::Int) where {T<:WF3x1}
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
                @cuda threads=256 blocks=setup_like.Nblocks shmem=(256 * sizeof(ComplexF64)) _amg_orthogonalize_kernel!(
                    reshape(vectors[v].f, :), reshape(vectors[u].f, :),
                    cache[:block_sites], cache[:block_counts], coeffs, nc_ng)
            end
            @cuda threads=256 blocks=setup_like.Nblocks shmem=(256 * sizeof(Float64)) _amg_normalize_kernel!(
                reshape(vectors[v].f, :), cache[:block_sites], cache[:block_counts], norms, nc_ng)
        end
        return vectors
    end
    Nv = length(vectors)
    v1 = vectors[1]
    NX, NY, NZ, NT = v1.NX, v1.NY, v1.NZ, v1.NT
    NC, NG = v1.NC, v1.NG
    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r
    block_dof = r^4 * NC * NG

    # 繝ｭ繝ｼ繧ｫ繝ｫ繝舌ャ繝輔ぃ繧剃ｺ句燕遒ｺ菫・(繝悶Ο繝・け豈弱↓蜀榊茜逕ｨ)
    local_vecs = [Vector{ComplexF64}(undef, block_dof) for _ in 1:Nv]

    @inbounds for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
        # 繝悶Ο繝・け蜀・ョ繝ｼ繧ｿ繧呈歓蜃ｺ
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
                proj = dot(local_vecs[u], local_vecs[v])
                local_vecs[v] .-= proj .* local_vecs[u]
            end
            nrm = norm(local_vecs[v])
            if nrm > 1e-14
                local_vecs[v] ./= nrm
            end
        end

        # 譖ｸ縺肴綾縺・
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


# ==========================================
#  蛻ｶ髯先ｼ皮ｮ怜ｭ・(Restriction: fine 竊・coarse)
# ==========================================

"""
    restrict_f2c(setup, x) 竊・CoarseVector

邏ｰ譬ｼ蟄舌・繧ｯ繝医Ν繧堤ｲ玲ｼ蟄舌↓蟆・ｽｱ:
  v_coarse[B, i] = 筺ｨﾏ違ｵ｢|x筺ｩ_B  (繝悶Ο繝・けB蜀・・蜀・ｩ・
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
                        val += conj(tv[ic, ix, iy, iz, it, ig]) * x[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            cv.data[(bidx-1)*Nv + v] = val
        end
    end

    return cv
end

"""
    restrict_f2c!(cv, setup, x) 竊・CoarseVector

restrict_f2c 縺ｮ in-place 迚医らｵ先棡繧・cv 縺ｫ譖ｸ縺崎ｾｼ繧縲・
"""
function restrict_f2c!(cv::CoarseVector, setup::DDalphaAMG_Setup, x::WF3x1)
    if x isa WF3x1GPU
        cache = _ensure_gpu_cache!(setup)
        nc_ng = Int32(size(x.f, 1) * size(x.f, 2))
        fill!(cv.data, zero(ComplexF64))
        for v in 1:setup.Nv
            @cuda threads=256 blocks=setup.Nblocks shmem=(256 * sizeof(ComplexF64)) _amg_restrict_one_kernel!(
                cv.data, reshape(setup.test_vectors[v].f, :), reshape(x.f, :),
                cache[:block_sites], cache[:block_counts], nc_ng, Int32(setup.Nv), Int32(v))
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
                        val += conj(tv[ic, ix, iy, iz, it, ig]) * x[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            cv.data[(bidx-1)*Nv + v] = val
        end
    end

    return cv
end


# ==========================================
#  蟒ｶ髟ｷ貍皮ｮ怜ｭ・(Prolongation: coarse 竊・fine)
# ==========================================

"""
    prolongate_c2f(setup, cv) 竊・WF3x1

邊玲ｼ蟄舌・繧ｯ繝医Ν繧堤ｴｰ譬ｼ蟄舌↓陬憺俣:
  x(site) = ﾎ｣盞｢ c[B(site), i] * ﾏ違ｵ｢(site)
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
    prolongate_c2f!(f, setup, cv) 竊・WF3x1

prolongate_c2f 縺ｮ in-place 迚医らｵ先棡繧・f 縺ｫ譖ｸ縺崎ｾｼ繧縲・
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
#  邊玲ｼ蟄先ｼ皮ｮ怜ｭ・(Coarse Grid Operator, matrix-free)
# ==========================================

"""
    coarse_apply(setup, D, cv) 竊・CoarseVector

邊玲ｼ蟄織irac貍皮ｮ怜ｭ舌・驕ｩ逕ｨ: Dc = R ﾂｷ D ﾂｷ P
  (Prolongate 竊・Apply D on fine grid 竊・Restrict)
"""
function coarse_apply(setup::DDalphaAMG_Setup, D, cv::CoarseVector)
    fine_v = prolongate_c2f(setup, cv)
    Dfine_v = similar(fine_v)
    _amg_mul_dirac!(Dfine_v, D, fine_v)
    return restrict_f2c(setup, Dfine_v)
end

"""
    coarse_apply!(result, setup, D, cv, fine_buf1, fine_buf2) 竊・CoarseVector

coarse_apply 縺ｮ in-place 迚医ゆｺ句燕遒ｺ菫昴＠縺・fine_buf1, fine_buf2 繧剃ｽ懈･ｭ繝舌ャ繝輔ぃ縺ｨ縺励※菴ｿ逕ｨ縺励・
邨先棡繧・result 縺ｫ譖ｸ縺崎ｾｼ繧縲ゅΓ繝｢繝ｪ蜑ｲ繧雁ｽ薙※繧ｼ繝ｭ縲・
  fine_buf1: prolongation 邨先棡逕ｨ
  fine_buf2: D 驕ｩ逕ｨ邨先棡逕ｨ
"""
function coarse_apply!(result::CoarseVector, setup::DDalphaAMG_Setup, D, cv::CoarseVector,
                       fine_buf1::WF3x1, fine_buf2::WF3x1)
    prolongate_c2f!(fine_buf1, setup, cv)
    _amg_mul_dirac!(fine_buf2, D, fine_buf1)
    restrict_f2c!(result, setup, fine_buf2)
    return result
end


# ==========================================
#  邊玲ｼ蟄先ｼ皮ｮ怜ｭ舌・莠句燕險育ｮ・
# ==========================================

"""
    precompute_coarse_operator!(setup, D; verbose=false, mode=:stencil)

Build coarse operator in one of two forms:
- `mode=:stencil` (default): block-stencil operator with Nvﾃ湧v couplings per block/neighbor.
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
            println("  coarse operator direct GPU stencil assembly... blocks=$(Nblocks), Nv=$(Nv)")
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
            println("  邊玲ｼ蟄先ｼ皮ｮ怜ｭ・stencil)繧剃ｺ句燕險育ｮ嶺ｸｭ... blocks=$(Nblocks), Nv=$(Nv), approx $(round(bytes_est/1024/1024, digits=1)) MB")
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
            println("  邊玲ｼ蟄先ｼ皮ｮ怜ｭ・stencil)縺ｮ莠句燕險育ｮ怜ｮ御ｺ・")
        end
        return setup
    elseif mode != :dense
        throw(ArgumentError("unknown coarse operator mode: $mode (use :stencil or :dense)"))
    end
    if setup.test_vectors[1] isa WF3x1GPU
        throw(ArgumentError("mode=:dense is CPU-only; use mode=:stencil for GPU DDalphaAMG"))
    end

    if verbose
        println("  邊玲ｼ蟄先ｼ皮ｮ怜ｭ舌ｒ莠句燕險育ｮ嶺ｸｭ... ($(n)ﾃ・(n) 陦悟・, $(round(n*n*16/1024/1024, digits=1)) MB)")
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
        println("  邊玲ｼ蟄先ｼ皮ｮ怜ｭ舌・莠句燕險育ｮ怜ｮ御ｺ・")
    end
    return setup
end


# ==========================================
#  繝ｩ繝ｳ繝繝繝輔ぉ繝ｫ繝溘が繝ｳ逕滓・ (Setup逕ｨ)
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
#  繧ｹ繝繝ｼ繧ｶ繝ｼ (Smoothers)
# ==========================================

"""
    mr_smooth!(x, D, b; niter)

Minimal Residual (MR) 繧ｹ繝繝ｼ繧ｶ繝ｼ縲・
蜷・う繝・Ξ繝ｼ繧ｷ繝ｧ繝ｳ:
  r = b - D*x,  ﾎｱ = 筺ｨD*r, r筺ｩ / 筺ｨD*r, D*r筺ｩ,  x += ﾎｱ*r
"""
function mr_smooth!(x::WF3x1, D, b::WF3x1; niter::Int=4,
                    fine_apply_counter::Union{Nothing,Ref{Int}}=nothing)

    Dr = similar(x)  # D*r work buffer
    r = similar(x)   # b - D*x work buffer
    zero_wf!(Dr)
    zero_wf!(r)
    for _ = 1:niter
        # r = b - D * x
        _amg_mul_dirac!(r, D, x; fine_apply_counter=fine_apply_counter)
        axpby!(1, b, -1, r)  # r = b - D*x  竊・axpby!(1, b, -1, r) => r = 1*b + (-1)*r
        # Dr = D * r
        _amg_mul_dirac!(Dr, D, r; fine_apply_counter=fine_apply_counter)
        num = inner_product(Dr, r)
        den = inner_product(Dr, Dr)
        if abs(den) < 1e-30
            break
        end
        ﾎｱ = num / den
        axpby!(ﾎｱ, r, 1, x)
    end
    return x
end


"""
    sap_smooth!(x, D, b, setup; niter, nblock_mr)

Schwarz Alternating Procedure (SAP) 繧ｹ繝繝ｼ繧ｶ繝ｼ縲・
繝悶Ο繝・け豈弱↓繝ｭ繝ｼ繧ｫ繝ｫ Minimal Residual 繧ｹ繝・ャ繝励ｒ螳溯｡・(multiplicative Schwarz)縲・
蜷・ヶ繝ｭ繝・け蜀・〒 Dirichlet B.C. 繧堤畑縺・．_BB * ﾎｴ 竕・r_B 繧定ｿ台ｼｼ逧・↓隗｣縺上・

繝舌ャ繝輔ぃ繧剃ｺ句燕遒ｺ菫昴＠縲√ヶ繝ｭ繝・け繝ｫ繝ｼ繝怜・縺ｧ縺ｮ繝｡繝｢繝ｪ蜑ｲ繧雁ｽ薙※繧貞屓驕ｿ縲・
蜀・ｩ阪・繝悶Ο繝・け繝ｭ繝ｼ繧ｫ繝ｫ縺ｧ險育ｮ励＠縲√ヵ繝ｫ譬ｼ蟄先ｼ皮ｮ励ｒ譛蟆丞喧縲・
"""
function sap_smooth!(x::WF3x1, D, b::WF3x1, setup::DDalphaAMG_Setup;
                     niter::Int=1, nblock_mr::Int=4,
                     fine_apply_counter::Union{Nothing,Ref{Int}}=nothing)
    if x isa WF3x1GPU && b isa WF3x1GPU && _amg_can_fuse_wilson(D, x, x)
        return _sap_smooth_gpu_paper!(x, D, b, setup;
                                      niter=niter, nblock_mr=nblock_mr,
                                      fine_apply_counter=fine_apply_counter)
    end

    r_blk = setup.r
    NBX, NBY, NBZ, NBT = setup.NBX, setup.NBY, setup.NBZ, setup.NBT

    # 繝舌ャ繝輔ぃ繧剃ｺ句燕遒ｺ菫晢ｼ医Ν繝ｼ繝怜・縺ｧ蜀榊茜逕ｨ・・
    ﾎｴ      = similar(x)
    local_r = similar(x)
    Alr    = similar(x)
    res    = similar(x)

    for _ = 1:niter
        # 繧ｰ繝ｭ繝ｼ繝舌Ν谿句ｷｮ: res = b - D*x
        _amg_mul_dirac!(res, D, x; fine_apply_counter=fine_apply_counter)           # res = D*x
        axpby!(1, b, -1, res)     # res = b - D*x  竊・axpby!(1, b, -1, res) => res = 1*b + (-1)*res

        # Red-black SAP. Same-color blocks are independent for nearest-neighbor Wilson D.
        for color in 0:1
            for bt = 1:NBT, bz = 1:NBZ, by = 1:NBY, bx = 1:NBX
                ((bx + by + bz + bt) & 1) == color || continue

                block_zero!(ﾎｴ, bx, by, bz, bt, r_blk)
                block_project!(local_r, bx, by, bz, bt, r_blk)
                block_copy!(local_r, res, bx, by, bz, bt, r_blk)

                for _ = 1:nblock_mr
                    _amg_mul_dirac!(Alr, D, local_r; fine_apply_counter=fine_apply_counter)
                    block_project!(Alr, bx, by, bz, bt, r_blk)

                    num = block_inner_product(Alr, local_r, bx, by, bz, bt, r_blk)
                    den = block_inner_product(Alr, Alr, bx, by, bz, bt, r_blk)
                    if abs(den) < 1e-30
                        break
                    end
                    ﾎｱ = num / den

                    block_axpby!(ﾎｱ, local_r, 1, ﾎｴ, bx, by, bz, bt, r_blk)
                    block_axpby!(-ﾎｱ, Alr, 1, local_r, bx, by, bz, bt, r_blk)
                end

                block_axpby!(1, ﾎｴ, 1, x, bx, by, bz, bt, r_blk)
                block_project!(ﾎｴ, bx, by, bz, bt, r_blk)
                _amg_mul_dirac!(Alr, D, ﾎｴ; fine_apply_counter=fine_apply_counter)
                axpby!(-1, Alr, 1, res)
            end
        end
    end
    return x
end

function _sap_smooth_gpu_paper!(x::WF3x1GPU, D, b::WF3x1GPU, setup::DDalphaAMG_Setup;
                                niter::Int=1, nblock_mr::Int=4,
                                fine_apply_counter::Union{Nothing,Ref{Int}}=nothing)
    cache = _ensure_gpu_cache!(setup)
    bi = x.blockinfo
    total = Int32(length(x.f))
    nc_ng = Int32(size(x.f, 1) * size(x.f, 2))
    threads = 256
    grid = cld(Int(total), threads)
    NBX = Int32(setup.NBX); NBY = Int32(setup.NBY); NBZ = Int32(setup.NBZ)

    delta = similar(x)
    local_r = similar(x)
    Alr = similar(x)
    res = similar(x)
    Ddelta = similar(x)
    num = CUDA.zeros(ComplexF64, setup.Nblocks)
    den = CUDA.zeros(ComplexF64, setup.Nblocks)

    for _ in 1:niter
        _amg_mul_dirac!(res, D, x; fine_apply_counter=fine_apply_counter)
        axpby!(1, b, -1, res)

        for color in Int32(0):Int32(1)
            @cuda threads=threads blocks=grid _amg_init_color_fields_kernel!(
                reshape(delta.f, :), reshape(local_r.f, :), reshape(res.f, :),
                cache[:bmap], nc_ng, total, color, NBX, NBY, NBZ)

            for _ in 1:nblock_mr
                @cuda threads=bi.blocksize blocks=bi.rsize _amg_local_wilson_block_kernel!(
                    Alr.f, local_r.f, D.U[1].U, D.U[2].U, D.U[3].U, D.U[4].U,
                    cache[:bmap], Float64(_amg_dirac_kappa(D)), bi, color, NBX, NBY, NBZ)

                @cuda threads=threads blocks=setup.Nblocks shmem=(2 * threads * sizeof(ComplexF64)) _amg_block_mr_dot_kernel!(
                    num, den, reshape(Alr.f, :), reshape(local_r.f, :),
                    cache[:block_sites], cache[:block_counts], nc_ng, color, NBX, NBY, NBZ)

                @cuda threads=threads blocks=grid _amg_block_mr_update_kernel!(
                    reshape(delta.f, :), reshape(local_r.f, :), reshape(Alr.f, :),
                    num, den, cache[:bmap], nc_ng, total, color, NBX, NBY, NBZ)
            end

            @cuda threads=threads blocks=grid _amg_add_color_field_kernel!(
                reshape(x.f, :), reshape(delta.f, :), cache[:bmap],
                nc_ng, total, color, NBX, NBY, NBZ)

            _amg_mul_dirac!(Ddelta, D, delta; fine_apply_counter=fine_apply_counter)
            @cuda threads=threads blocks=grid _amg_sub_field_kernel!(
                reshape(res.f, :), reshape(Ddelta.f, :))
        end
    end
    return x
end


# ==========================================
#  邊玲ｼ蟄・GMRES (Coarse Grid Solver)
# ==========================================

"""
    coarse_gmres(setup, D, b_c; maxiter, restart, tol, coarse_apply_counter) 竊・CoarseVector

邊玲ｼ蟄蝉ｸ翫・ GMRES 繧ｽ繝ｫ繝舌・縲・c * x_c = b_c 繧定ｧ｣縺上・
Dc 縺ｯ matrix-free 縺ｧ驕ｩ逕ｨ: Dc(v) = R ﾂｷ D ﾂｷ P(v)
    coarse_apply_counter 縺梧ｸ｡縺輔ｌ縺溷ｴ蜷医∫ｲ玲ｼ蟄先ｼ皮ｮ怜ｭ・Dc 縺ｮ驕ｩ逕ｨ蝗樊焚繧貞刈邂励☆繧九・
"""
function coarse_gmres(setup::DDalphaAMG_Setup, D, b_c::CoarseVector;
                      maxiter::Int=50, restart::Int=20, tol::Float64=1e-1,
                      coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing)
    x_c = similar(b_c)
    zero!(x_c)
    b_norm = coarse_norm(b_c)
    if b_norm < 1e-30
        return x_c
    end
    has_Dc = setup.Dc_matrix !== nothing
    has_stencil = setup.coarse_stencil !== nothing

    # 莠句燕險育ｮ玲ｸ医∩陦悟・縺後↑縺・ｴ蜷医・縺ｿ邏ｰ譬ｼ蟄舌ヰ繝・ヵ繧｡繧堤｢ｺ菫・
    local fine_buf1, fine_buf2
    if !has_Dc
        fine_buf1 = similar(setup.test_vectors[1])
        fine_buf2 = similar(setup.test_vectors[1])
    end
    m = ensure_coarse_gmres_cache!(setup, b_c, restart)
    V = setup._gmres_V
    w_c = setup._gmres_w::CoarseVector
    r_c = setup._gmres_r::CoarseVector
    H = setup._gmres_H
    e1 = setup._gmres_e1

    # 邊玲ｼ蟄先ｼ皮ｮ怜ｭ宣←逕ｨ縺ｮ繝倥Ν繝代・
    @inline function apply_Dc!(w, v)
        if has_stencil
            apply_coarse_stencil!(w, setup.coarse_stencil, v)
        elseif has_Dc
            mul!(w.data, setup.Dc_matrix, v.data)
        else
            coarse_apply!(w, setup, D, v, fine_buf1, fine_buf2)
        end
        coarse_apply_counter !== nothing && (coarse_apply_counter[] += 1)
    end

    for _ = 1:maxiter
        # 谿句ｷｮ: r = b - Dc*x
        r_c.data .= b_c.data
        apply_Dc!(w_c, x_c)
        coarse_axpby!(-1.0, w_c, 1.0, r_c)

        ﾎｲ = coarse_norm(r_c)
        if ﾎｲ / b_norm < tol
            return x_c
        end

        # Arnoldi 驕守ｨ・
        fill!(H, zero(ComplexF64))
        V[1].data .= r_c.data ./ ﾎｲ

        j_final = m
        for j = 1:m
            apply_Dc!(w_c, V[j])

            # Modified Gram-Schmidt
            for i = 1:j
                H[i,j] = coarse_dot(V[i], w_c)
                coarse_axpby!(-H[i,j], V[i], 1.0, w_c)
            end
            H[j+1,j] = coarse_norm(w_c)

            if abs(H[j+1,j]) < 1e-14
                j_final = j
                break
            end
            V[j+1].data .= w_c.data ./ H[j+1,j]
        end

        # 譛蟆丈ｺ御ｹ怜撫鬘・ min ||ﾎｲ*e竄・- H*y||
        fill!(e1, zero(ComplexF64))
        e1[1] = ﾎｲ
        y = H[1:j_final+1, 1:j_final] \ view(e1, 1:j_final+1)

        for k = 1:j_final
            coarse_axpby!(y[k], V[k], 1.0, x_c)
        end

        # 蜿取據遒ｺ隱・
        r_c.data .= b_c.data
        apply_Dc!(w_c, x_c)
        coarse_axpby!(-1.0, w_c, 1.0, r_c)
        if coarse_norm(r_c) / b_norm < tol
            return x_c
        end
    end

    return x_c
end


# ==========================================
#  V-cycle
# ==========================================

"""
    vcycle(D, b, setup; pre_smooth, post_smooth, coarse_tol, smoother)

1蝗槭・繝槭Ν繝√げ繝ｪ繝・ラ V-cycle 繧貞ｮ溯｡後＠縲．*x 竕・b 縺ｮ霑台ｼｼ隗｣繧定ｿ斐☆縲・

1. Pre-smoothing (MR or SAP)
2. 谿句ｷｮ繧堤ｲ玲ｼ蟄舌↓蟆・ｽｱ: r_c = R(b - D*x)
3. 邊玲ｼ蟄舌〒 GMRES: Dc * e_c = r_c
4. 陬懈ｭ｣繧堤ｴｰ譬ｼ蟄舌↓謌ｻ縺・ x += P(e_c)
5. Post-smoothing
"""
function vcycle(D, b::WF3x1, setup::DDalphaAMG_Setup;
                pre_smooth::Int=4, post_smooth::Int=4,
                coarse_maxiter::Int=50, coarse_tol::Float64=1e-1,
                smoother::Symbol=:mr,
                coarse_apply_counter::Union{Nothing,Ref{Int}}=nothing,
                fine_apply_counter::Union{Nothing,Ref{Int}}=nothing)
    x = similar(b)
    zero_wf!(x)

    # Pre-allocate work buffers to avoid temporary WF allocations in hot path.
    r_fine = similar(b)
    correction = similar(b)
    r_coarse = _coarse_vector(setup)

    # 1. Pre-smoothing
    if smoother == :sap
        sap_smooth!(x, D, b, setup; niter=1, nblock_mr=pre_smooth,
                    fine_apply_counter=fine_apply_counter)
    else
        mr_smooth!(x, D, b; niter=pre_smooth,
                   fine_apply_counter=fine_apply_counter)
    end

    # 2. 谿句ｷｮ險育ｮ・竊・邊玲ｼ蟄舌↓蟆・ｽｱ
    _amg_mul_dirac!(r_fine, D, x; fine_apply_counter=fine_apply_counter)       # r_fine = D*x
    axpby!(1, b, -1, r_fine) # r_fine = b - D*x
    restrict_f2c!(r_coarse, setup, r_fine)

    # 3. 邊玲ｼ蟄先ｱりｧ｣
    e_coarse = coarse_gmres(setup, D, r_coarse;
                            maxiter=coarse_maxiter, tol=coarse_tol,
                            coarse_apply_counter=coarse_apply_counter)

    # 4. 邏ｰ譬ｼ蟄舌∈縺ｮ陬懈ｭ｣
    prolongate_c2f!(correction, setup, e_coarse)
    axpby!(1, correction, 1, x)

    # 5. Post-smoothing
    if smoother == :sap
        sap_smooth!(x, D, b, setup; niter=1, nblock_mr=post_smooth,
                    fine_apply_counter=fine_apply_counter)
    else
        mr_smooth!(x, D, b; niter=post_smooth,
                   fine_apply_counter=fine_apply_counter)
    end

    return x
end


# ==========================================
#  FGMRES (Flexible GMRES) 螟夜Κ繧ｽ繝ｫ繝舌・
# ==========================================

"""
    fgmres(D, b, setup; x0, maxiter, restart, tol, verbose, smoother, ...)

Flexible GMRES (V-cycle 繝励Μ繧ｳ繝ｳ繝・ぅ繧ｷ繝ｧ繝翫・莉倥″)縲・
D*x = b 繧定ｧ｣縺上・

FGMRES 縺ｯ蜷・う繝・Ξ繝ｼ繧ｷ繝ｧ繝ｳ縺ｧ逡ｰ縺ｪ繧九・繝ｪ繧ｳ繝ｳ繝・ぅ繧ｷ繝ｧ繝翫・繧定ｨｱ螳ｹ縺吶ｋ縺溘ａ縲・
髱樒ｷ壼ｽ｢繝槭Ν繝√げ繝ｪ繝・ラ V-cycle 繧貞､牙虚繝励Μ繧ｳ繝ｳ繝・ぅ繧ｷ繝ｧ繝翫・縺ｨ縺励※菴ｿ逕ｨ蜿ｯ閭ｽ縲・
"""
function fgmres(D, b::WF3x1, setup::DDalphaAMG_Setup;
                x0::Union{Nothing,WF3x1}=nothing,
                maxiter::Int=100, restart::Int=20, tol::Float64=1e-10,
                verbose::Bool=true, smoother::Symbol=:mr,
                pre_smooth::Int=4, post_smooth::Int=4,
                coarse_tol::Float64=1e-1)

    x = x0 === nothing ? similar(b) : _copy_wf(x0)
    x0 === nothing && zero_wf!(x)
    b_norm = sqrt(real(inner_product(b, b)))
    if b_norm < 1e-30
        return x
    end

    r = _copy_wf(b)  # r = b - D*x buffer
    zero_wf!(r)

    total_coarse_gmres = 0  # coarse GMRES 蜻ｼ蜃ｺ縺怜屓謨ｰ繧ｫ繧ｦ繝ｳ繧ｿ
    _coarse_apply_counter = Ref(0)  # 邊玲ｼ蟄先ｼ皮ｮ怜ｭ・Dc 驕ｩ逕ｨ蝗樊焚繧ｫ繧ｦ繝ｳ繧ｿ
    _fine_apply_counter = Ref(0)

    for cycle = 1:maxiter
        # r = b - D * x
        _amg_mul_dirac!(r, D, x; fine_apply_counter=_fine_apply_counter)           # r = D*x
        axpby!(1, b, -1, r)     # r = b - D*x  竊・axpby!(1, b, -1, r) => r = 1*b + (-1)*r
        ﾎｲ = sqrt(real(inner_product(r, r)))

        if verbose
            println("FGMRES cycle $cycle: relative residual = $(ﾎｲ/b_norm)")
        end

        if ﾎｲ / b_norm < tol
            if verbose
                println("FGMRES converged! ($(cycle) cycles)")
                println("  Total coarse GMRES calls: $total_coarse_gmres")
                println("  Total coarse operator applications: $(_coarse_apply_counter[])")
                println("  Total fine Dirac applications: $(_fine_apply_counter[])")
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
        g[1] = ﾎｲ

        V[1] = r / ﾎｲ

        j_final = m
        converged = false
        for j = 1:m
            # V-cycle 繝励Μ繧ｳ繝ｳ繝・ぅ繧ｷ繝ｧ繝翫・: M竅ｻﾂｹ * v_j
            Z[j] = vcycle(D, V[j], setup;
                          pre_smooth=pre_smooth, post_smooth=post_smooth,
                          coarse_tol=coarse_tol, smoother=smoother,
                          coarse_apply_counter=_coarse_apply_counter,
                          fine_apply_counter=_fine_apply_counter)
            total_coarse_gmres += 1

            w = similar(b)
            _amg_mul_dirac!(w, D, Z[j]; fine_apply_counter=_fine_apply_counter)

            # Modified Gram-Schmidt
            for i = 1:j
                H[i,j] = inner_product(V[i], w)
                axpby!(-H[i,j], V[i], 1, w)
            end
            H[j+1,j] = sqrt(real(inner_product(w, w)))

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

        # 隗｣縺ｮ譖ｴ譁ｰ: x += Z * y
        for k = 1:j_final
            axpby!(y[k], Z[k], 1, x)
        end

        if converged
            if verbose
                println("FGMRES converged at cycle $cycle, step $(j_final)!")
                println("  Total coarse GMRES calls: $total_coarse_gmres")
                println("  Total coarse operator applications: $(_coarse_apply_counter[])")
                println("  Total fine Dirac applications: $(_fine_apply_counter[])")
            end
            return x
        end
    end

    if verbose
        r_final = similar(b)
        _amg_mul_dirac!(r_final, D, x; fine_apply_counter=_fine_apply_counter)
        axpby!(1, b, -1, r_final)
        println("FGMRES: max iterations reached. " *
                "Final relative residual = $(sqrt(real(inner_product(r_final, r_final)))/b_norm)")
        println("  Total coarse GMRES calls: $total_coarse_gmres")
        println("  Total coarse operator applications: $(_coarse_apply_counter[])")
        println("  Total fine Dirac applications: $(_fine_apply_counter[])")
    end
    return x
end


# ==========================================
#  繧ｻ繝・ヨ繧｢繝・・ (Full Setup)
# ==========================================

"""
    setup_DDalphaAMG(D, template, r, Nv; nsmooth, verbose, smoother, ...) 竊・DDalphaAMG_Setup

DD-ﾎｱAMG 縺ｮ螳悟・繧ｻ繝・ヨ繧｢繝・・繧定｡後≧縲・

謇矩・
1. 繝ｩ繝ｳ繝繝繝・せ繝医・繧ｯ繝医Ν繧・Nv 蛟狗函謌・
2. 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ縺ｧ菴弱Δ繝ｼ繝画・蛻・ｒ蠑ｷ蛹・(D*ﾏ・竕・0 縺ｮ霑台ｼｼ豎りｧ｣)
3. 繝悶Ο繝・け逶ｴ莠､豁｣隕丞喧

蠑墓焚:
- D: Dirac 貍皮ｮ怜ｭ・(Wilson_Dirac_operator 遲・
- template: 譬ｼ蟄舌し繧､繧ｺ蜿ら・逕ｨ縺ｮ WF3x1
- r: 蜷・婿蜷代・繝悶Ο繝・け繧ｵ繧､繧ｺ (NX,NY,NZ,NT 縺・r 縺ｧ蜑ｲ繧雁・繧後ｋ縺薙→)
- Nv: 繝・せ繝医・繧ｯ繝医Ν謨ｰ (謗ｨ螂ｨ: 20縲・0)
- nsmooth: 繧ｻ繝・ヨ繧｢繝・・繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ縺ｮ蜿榊ｾｩ謨ｰ (繝・ヵ繧ｩ繝ｫ繝・ 8)
- smoother: :mr (繝・ヵ繧ｩ繝ｫ繝・, :sap, :vcycle
  - :mr  窶・MR 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ縺ｮ縺ｿ
  - :sap 窶・SAP 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ縺ｮ縺ｿ
  - :vcycle 窶・MR 縺ｧ莠亥ｙ繧ｻ繝・ヨ繧｢繝・・蠕後〃-cycle 縺ｧ邊玲ｼ蟄先ｮ句ｷｮ謌仙・繧貞ｼｷ隱ｿ
- nvcycle: V-cycle 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ縺ｮ蜿榊ｾｩ謨ｰ (smoother=:vcycle 譎ゅ√ョ繝輔か繝ｫ繝・ 4)
- vcycle_pre_smooth, vcycle_post_smooth: V-cycle 蜀・・繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ蝗樊焚 (繝・ヵ繧ｩ繝ｫ繝・ 蜷・)

菴ｿ逕ｨ萓・
```julia
setup = setup_DDalphaAMG(D, x, 2, 24; nsmooth=16)
setup = setup_DDalphaAMG(D, x, 2, 24; smoother=:vcycle, nsmooth=8, nvcycle=4)
x = solve_DDalphaAMG(D, b, setup; tol=1e-10)
```
"""
function setup_DDalphaAMG(D, template::WF3x1, r::Int, Nv::Int;
                          nsmooth::Int=8, verbose::Bool=true,
                          coarse_vectors::Union{Nothing,Vector{<:WF3x1}}=nothing,
                          smoother::Symbol=:mr,
                          nvcycle::Int=4,
                          vcycle_pre_smooth::Int=4, vcycle_post_smooth::Int=4,
                          setup_smoother::Symbol=:mr,
                          vcycle_update_smoother::Symbol=:sap,
                          use_chirality::Bool=false)
    NX, NY, NZ, NT = template.NX, template.NY, template.NZ, template.NT
    NC, NG = template.NC, template.NG

    @assert NX % r == 0 "NX=$NX must be divisible by r=$r"
    @assert NY % r == 0 "NY=$NY must be divisible by r=$r"
    @assert NZ % r == 0 "NZ=$NZ must be divisible by r=$r"
    @assert NT % r == 0 "NT=$NT must be divisible by r=$r"

    if use_chirality
        @assert Nv % 2 == 0 "Nv must be even when use_chirality=true (Nv=$Nv)"
    end
    # Chiral setup also smooths Nv independent vectors.
    # We then assign independent halves to P+ and P- sectors,
    # instead of splitting each single vector into plus/minus copies.
    Nv_smooth = Nv

    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r
    Nblocks = NBX * NBY * NBZ * NBT

    if verbose
        println("DD-alphaAMG Setup:")
        println("  fine lattice: $(NX)x$(NY)x$(NZ)x$(NT)")
        println("  block size: $(r)^4")
        println("  coarse lattice: $(NBX)x$(NBY)x$(NBZ)x$(NBT) = $(Nblocks) blocks")
        println("  test vectors: $Nv", use_chirality ? " (chiral split: $(Nv_smooth) -> Nv)" : "")
        println("  coarse dof: $(Nblocks * Nv)")
    end

    # 1. 繝ｩ繝ｳ繝繝繝・せ繝医・繧ｯ繝医Ν逕滓・
    if coarse_vectors !== nothing
        @assert length(coarse_vectors) == Nv "length(coarse_vectors) must match Nv"
        # Keep caller-provided vectors immutable from the setup's point of view.
        # setup_DDalphaAMG performs in-place block orthonormalization, so aliasing
        # the input vector array here corrupts later uses such as deflated_initial_guess(D, b, v).
        test_vectors = _copy_wf_vector(coarse_vectors)
        if verbose
            println("  using externally provided test vectors")
        end
    else
        test_vectors = [similar(template) for _ in 1:Nv_smooth]
        for v in 1:Nv_smooth
            random_fermion!(test_vectors[v])
        end

        # 2. 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ: D*ﾏ・竕・0 繧定ｿ台ｼｼ豎りｧ｣ 竊・菴弱Δ繝ｼ繝画・蛻・歓蜃ｺ
        zero_rhs = similar(template)
        zero_wf!(zero_rhs)

        if smoother == :vcycle
            _setup_fine_apply_counter = Ref(0)
            # Algorithm 16: use the V-cycle correction
            # v_j <- v_j + C^(nu)(v_j - D*v_j) in the adaptive setup loop.
            # Keep the original strong setup smoothing controlled by `nsmooth`;
            # for this codebase it is essential for useful initial test vectors.
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
            if verbose
                println("  [Algorithm 16 setup] Phase 1: $(setup_smoother) smoothing ($nsmooth steps)...")
                println("  [Algorithm 16 setup] adaptive post-smoother: $(vcycle_update_smoother) ($vcycle_post_smooth steps)")
            end
            for v in 1:Nv_smooth
                if setup_smoother == :sap
                    sap_smooth!(test_vectors[v], D, zero_rhs, sap_setup;
                                niter=1, nblock_mr=nsmooth,
                                fine_apply_counter=_setup_fine_apply_counter)
                else
                    mr_smooth!(test_vectors[v], D, zero_rhs; niter=nsmooth,
                               fine_apply_counter=_setup_fine_apply_counter)
                end
            end

            r_fine = _copy_wf(zero_rhs)
            correction = _copy_wf(zero_rhs)
            r_coarse = _coarse_vector(Nv_smooth, Nblocks, template)

            for iter in 1:nvcycle
                cur_vectors = _copy_wf_vector(test_vectors)
                block_orthonormalize!(cur_vectors, r)
                cur_setup = DDalphaAMG_Setup{typeof(template)}(
                    cur_vectors, r, Nv_smooth, NX, NY, NZ, NT, NC, NG,
                    NBX, NBY, NBZ, NBT, Nblocks, nothing,
                    nothing,
                    CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing
                )
                precompute_coarse_operator!(cur_setup, D; mode=:direct_stencil)

                for v in 1:Nv_smooth
                    _amg_mul_dirac!(r_fine, D, test_vectors[v];
                                    fine_apply_counter=_setup_fine_apply_counter)
                    axpby!(1, test_vectors[v], -1, r_fine)
                    restrict_f2c!(r_coarse, cur_setup, r_fine)
                    e_coarse = coarse_gmres(cur_setup, D, r_coarse;
                                            maxiter=50, tol=1e-1)
                    prolongate_c2f!(correction, cur_setup, e_coarse)
                    if vcycle_update_smoother == :sap
                        sap_smooth!(correction, D, r_fine, cur_setup;
                                    niter=1, nblock_mr=vcycle_post_smooth,
                                    fine_apply_counter=_setup_fine_apply_counter)
                    else
                        mr_smooth!(correction, D, r_fine; niter=vcycle_post_smooth,
                                   fine_apply_counter=_setup_fine_apply_counter)
                    end
                    axpby!(1, correction, 1, test_vectors[v])
                    normalize_fermion!(test_vectors[v])
                end

                if verbose
                    println("  [Algorithm 16 setup] iteration $iter/$nvcycle done")
                end
            end
            verbose && println("  [Algorithm 16 setup] fine Dirac applications: $(_setup_fine_apply_counter[])")
        elseif smoother == :sap
            if verbose
                println("  smoothing test vectors with SAP ($nsmooth iterations)...")
            end
            # SAP 縺ｫ縺ｯ繧ｻ繝・ヨ繧｢繝・・縺悟ｿ・ｦ・竊・縺ｾ縺壻ｻｮ繧ｻ繝・ヨ繧｢繝・・繧呈ｧ狗ｯ・
            block_orthonormalize!(test_vectors, r)
            pre_setup = DDalphaAMG_Setup{typeof(template)}(
                test_vectors, r, Nv_smooth, NX, NY, NZ, NT, NC, NG,
                NBX, NBY, NBZ, NBT, Nblocks, nothing,
                nothing,
                CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing
            )
            for v in 1:Nv_smooth
                sap_smooth!(test_vectors[v], D, zero_rhs, pre_setup; niter=1, nblock_mr=nsmooth)
            end
        else  # :mr (繝・ヵ繧ｩ繝ｫ繝・
            if verbose
                println("  smoothing test vectors with MR ($nsmooth iterations)...")
            end
            for v in 1:Nv_smooth
                mr_smooth!(test_vectors[v], D, zero_rhs; niter=nsmooth)
            end
        end
    end



    # 2.5. 繧ｫ繧､繝ｩ繝ｫ蛻・牡 (chiral splitting)
    if use_chirality && coarse_vectors === nothing
        if verbose
            println("  applying chiral split...")
        end
        chiral_vectors = Vector{typeof(template)}(undef, Nv)
        half = Nv ÷ 2
        # First half -> P+ sector, second half -> P- sector.
        for v in 1:half
            chiral_vectors[v] = _copy_wf(test_vectors[v])
            chiral_project!(chiral_vectors[v], :plus)   # ig=3,4 菫晄戟
        end
        for v in 1:half
            chiral_vectors[half + v] = _copy_wf(test_vectors[half + v])
            chiral_project!(chiral_vectors[half + v], :minus)    # ig=1,2 菫晄戟
        end
        test_vectors = chiral_vectors
    end

    # 3. 繝悶Ο繝・け逶ｴ莠､豁｣隕丞喧
    if verbose
        println("  block orthonormalizing...")
    end
    if use_chirality
        # P竄・縺ｨ P竄・繧ｻ繧ｯ繧ｿ繝ｼ繧貞挨縲・↓逶ｴ莠､豁｣隕丞喧 竊・繧ｫ繧､繝ｩ繝ｫ讒矩繧剃ｿ晄戟
        half = Nv ÷ 2
        plus_vectors  = test_vectors[1:half]       # P竄・(ig=3,4)
        minus_vectors = test_vectors[half+1:end]   # P竄・(ig=1,2)
        block_orthonormalize!(plus_vectors, r)
        block_orthonormalize!(minus_vectors, r)
    else
        block_orthonormalize!(test_vectors, r)
    end

    setup = DDalphaAMG_Setup{typeof(template)}(
        test_vectors, r, Nv, NX, NY, NZ, NT, NC, NG,
        NBX, NBY, NBZ, NBT, Nblocks, nothing,
        nothing,
        CoarseVector[], nothing, nothing, zeros(ComplexF64, 0, 0), ComplexF64[], 0, nothing
    )

    # 邊玲ｼ蟄先ｼ皮ｮ怜ｭ舌ｒ莠句燕險育ｮ・
    precompute_coarse_operator!(setup, D; verbose=verbose, mode=:direct_stencil)

    if verbose
        println("  setup done")
    end

    return setup
end


# ==========================================
#  螳悟・繧ｽ繝ｫ繝舌・ (Full Solve)
# ==========================================

"""
    solve_DDalphaAMG(D, b, setup; x0, tol, maxiter, restart, verbose, smoother, ...)

DD-ﾎｱAMG 繧堤畑縺・※ D*x = b 繧定ｧ｣縺上・
FGMRES 螟夜Κ繧ｽ繝ｫ繝舌・ + V-cycle 繝励Μ繧ｳ繝ｳ繝・ぅ繧ｷ繝ｧ繝翫・繧剃ｽｿ逕ｨ縲・

蠑墓焚:
- D: Dirac 貍皮ｮ怜ｭ・
- b: 蜿ｳ霎ｺ繝吶け繝医Ν (WF3x1)
- setup: DDalphaAMG_Setup (setup_DDalphaAMG 縺ｧ逕滓・)
- tol: 蜿取據險ｱ螳ｹ隱､蟾ｮ (繝・ヵ繧ｩ繝ｫ繝・ 1e-10)
- smoother: :mr (Minimal Residual, 繝・ヵ繧ｩ繝ｫ繝・ 縺ｾ縺溘・ :sap (Schwarz Alternating Procedure)
- pre_smooth, post_smooth: 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ蜿榊ｾｩ謨ｰ (繝・ヵ繧ｩ繝ｫ繝・ 蜷・)

菴ｿ逕ｨ萓・
```julia
setup = setup_DDalphaAMG(D, x, 2, 24)
solution = solve_DDalphaAMG(D, b, setup; tol=1e-10, smoother=:mr)
residual = norm2(D * solution - b)
```
"""
function solve_DDalphaAMG(D, b::WF3x1, setup::DDalphaAMG_Setup;
                          x0::Union{Nothing,WF3x1}=nothing,
                          tol::Float64=1e-10, maxiter::Int=100, restart::Int=20,
                          verbose::Bool=true, smoother::Symbol=:mr,
                          pre_smooth::Int=4, post_smooth::Int=4)
    return fgmres(D, b, setup;
                  x0=x0, tol=tol, maxiter=maxiter, restart=restart,
                  verbose=verbose, smoother=smoother,
                  pre_smooth=pre_smooth, post_smooth=post_smooth)
end

function _debug_field_relerr(cpu_ref::WF3x1CPU, val::WF3x1, cpu_template::WF3x1CPU)
    val_cpu = val isa WF3x1CPU ? val : begin
        tmp = similar(cpu_template)
        substitute_fermion!(tmp, val)
        CUDA.synchronize()
        tmp
    end
    diff = deepcopy(cpu_ref)
    axpby!(-1, val_cpu, 1, diff)
    num = sqrt(real(inner_product(diff, diff)))
    den = sqrt(real(inner_product(cpu_ref, cpu_ref)))
    return num / max(den, eps(Float64))
end

function _debug_coarse_relerr(cpu_ref::CoarseVector, val::CoarseVector)
    vdata = val.data isa CuArray ? Array(val.data) : val.data
    return norm(cpu_ref.data .- vdata) / max(norm(cpu_ref.data), eps(Float64))
end

function _debug_report!(rows, name, err)
    push!(rows, (name=name, relerr=err))
    println(rpad(name, 28), " relerr = ", err)
    return err
end

function compare_ddalphaamg_cpu_gpu(Dcpu, Dgpu,
                                    bcpu::WF3x1CPU, bgpu::WF3x1GPU,
                                    setup_cpu::DDalphaAMG_Setup,
                                    setup_gpu::DDalphaAMG_Setup;
                                    pre_smooth::Int=4,
                                    post_smooth::Int=4,
                                    coarse_tol::Float64=1e-1,
                                    coarse_maxiter::Int=50)
    rows = NamedTuple[]
    println("CPU b size = ", size(bcpu.f))
    println("GPU b size = ", size(bgpu.f),
            " blocksize=", bgpu.blockinfo.blocksize,
            " rsize=", bgpu.blockinfo.rsize)
    println("CPU setup: Nv=", setup_cpu.Nv, " Nblocks=", setup_cpu.Nblocks)
    println("GPU setup: Nv=", setup_gpu.Nv, " Nblocks=", setup_gpu.Nblocks)

    _assert_same_wf_layout("setup_gpu.test_vectors[1]", setup_gpu.test_vectors[1], "bgpu", bgpu)

    ntv = min(length(setup_cpu.test_vectors), length(setup_gpu.test_vectors))
    for i in 1:ntv
        _debug_report!(rows, "test_vector[$i]",
            _debug_field_relerr(setup_cpu.test_vectors[i], setup_gpu.test_vectors[i], bcpu))
    end

    db_cpu = similar(bcpu)
    db_gpu = similar(bgpu)
    mul!(db_cpu, Dcpu, bcpu)
    mul!(db_gpu, Dgpu, bgpu)
    CUDA.synchronize()
    _debug_report!(rows, "D*b", _debug_field_relerr(db_cpu, db_gpu, bcpu))

    rc_cpu = restrict_f2c(setup_cpu, bcpu)
    rc_gpu = restrict_f2c(setup_gpu, bgpu)
    CUDA.synchronize()
    _debug_report!(rows, "restrict_f2c(b)", _debug_coarse_relerr(rc_cpu, rc_gpu))

    cg_cpu = coarse_gmres(setup_cpu, Dcpu, rc_cpu; maxiter=coarse_maxiter, tol=coarse_tol)
    cg_gpu = coarse_gmres(setup_gpu, Dgpu, rc_gpu; maxiter=coarse_maxiter, tol=coarse_tol)
    CUDA.synchronize()
    _debug_report!(rows, "coarse_gmres(Rb)", _debug_coarse_relerr(cg_cpu, cg_gpu))

    pc_cpu = prolongate_c2f(setup_cpu, cg_cpu)
    pc_gpu = prolongate_c2f(setup_gpu, cg_gpu)
    CUDA.synchronize()
    _debug_report!(rows, "prolongate(coarse)", _debug_field_relerr(pc_cpu, pc_gpu, bcpu))

    vc_cpu = vcycle(Dcpu, bcpu, setup_cpu;
        pre_smooth=pre_smooth, post_smooth=post_smooth,
        coarse_maxiter=coarse_maxiter, coarse_tol=coarse_tol)
    vc_gpu = vcycle(Dgpu, bgpu, setup_gpu;
        pre_smooth=pre_smooth, post_smooth=post_smooth,
        coarse_maxiter=coarse_maxiter, coarse_tol=coarse_tol)
    CUDA.synchronize()
    _debug_report!(rows, "vcycle(b)", _debug_field_relerr(vc_cpu, vc_gpu, bcpu))

    return rows
end


# ==========================================
#  Deflation: 蛻晄悄謗ｨ螳壹・繧ｯ繝医Ν逕滓・
# ==========================================
"""
    deflated_initial_guess(D, b, test_vectors; verbose=false)

繝・せ繝医・繧ｯ繝医Ν {v_i} 縺九ｉ蟆・ｽｱ陦悟・ G[i,j] = 筺ｨv_i, D v_j筺ｩ 繧呈ｧ狗ｯ峨＠縲・
蝗ｺ譛牙､蛻・ｧ｣ G = U ﾎ・U窶 繧定｡後▲縺ｦ縲‥eflation 蛻晄悄謗ｨ螳壹ｒ逕滓・縺吶ｋ縲・

謇矩・
1. G[i,j] = 筺ｨv_i, D v_j筺ｩ,  rhs[i] = 筺ｨv_i, b筺ｩ
2. 蝗ｺ譛牙､蛻・ｧ｣:  G = U ﾎ・U窶
3. 蝗ｺ譛牙､蝓ｺ蠎輔〒縺ｮ蟆・ｽｱ:  ﾎｱ_k = (U窶 rhs)_k / ﾎｻ_k
4. 蜈・・蝓ｺ蠎輔↓謌ｻ縺・  c = U ﾎｱ
5. x竄 = ﾎ｣_j c_j v_j
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

    # G c = rhs 繧定ｧ｣縺・
    c = G \ rhs

    # x竄 = ﾎ｣ c_j v_j
    x0 = similar(b)
    zero_wf!(x0)
    for j in 1:Nv
        axpby!(c[j], test_vectors[j], 1, x0)
    end

    return x0
end


end # module DDalphaAMG
