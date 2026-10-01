# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/gpu.jl.

function Generate_weight(x::Any; type="real")
    W = Vector{Spin_matrix}()

    if x isa Vector{WF3x1}
        for _ in x
            push!(W, Random_SpinM(type))
        end

    elseif x isa WF3x1
        push!(W, Random_SpinM(type))

    elseif x isa Hopping_operator
        U = x.Gauge
        if U isa GPUgauge
            W = [Random_SpinM_GPU(type) for _ in 1:length(x.P.various)]
        else
            W = [Random_SpinM(type) for _ in 1:length(x.P.various)]
        end

    else
        throw(ArgumentError("지원되지 않는 x 타입: $(typeof(x))"))
    end

    return W
end

"""
    Matmul_batch!(result, M_all, M_cpu, Qs, Ks) → result

Preallocated version of Matmul_batch. Reuses M_all (CuArray), M_cpu (CPU Array), result (Vector{Spin_matrix_GPU}).
"""
function Matmul_batch!(result, M_all, M_cpu::Array{ComplexF64,3},
                       Qs::AbstractVector, Ks::AbstractVector)
    n = length(Qs)
    NC, NG = Qs[1].NC, Qs[1].NG
    blocksize = Qs[1].blockinfo.blocksize
    rsize     = Qs[1].blockinfo.rsize

    M_all .= 0

    threads = 256
    blocks  = NG * NG

    for idx in 1:n
        CUDA.@cuda threads=threads blocks=blocks kernel_matmul_batched!(
            M_all, Qs[idx].f, Ks[idx].f, NC, NG, blocksize, rsize, idx)
    end

    copyto!(M_cpu, M_all)

    @inbounds for idx in 1:n
        result[idx].sm .= MMatrix{4,4,ComplexF64}(@view M_cpu[:,:,idx])
    end
    return result
end

# WF3x1GPU 전용 addmul! 커널
function kernel_addmul_spinmat_mul!(
    out::CuDeviceArray{ComplexF64,4},   # (c, α, b, r)
    M::SMatrix{4,4,ComplexF64,16},
    inp::CuDeviceArray{ComplexF64,4},
)
    b = Int(threadIdx().x)
    r = Int(blockIdx().x)

    @inbounds for c = 1:3
        x1 = inp[c, 1, b, r]
        x2 = inp[c, 2, b, r]
        x3 = inp[c, 3, b, r]
        x4 = inp[c, 4, b, r]

        out[c, 1, b, r] += M[1,1]*x1 + M[1,2]*x2 + M[1,3]*x3 + M[1,4]*x4
        out[c, 2, b, r] += M[2,1]*x1 + M[2,2]*x2 + M[2,3]*x3 + M[2,4]*x4
        out[c, 3, b, r] += M[3,1]*x1 + M[3,2]*x2 + M[3,3]*x3 + M[3,4]*x4
        out[c, 4, b, r] += M[4,1]*x1 + M[4,2]*x2 + M[4,3]*x3 + M[4,4]*x4
    end
    return nothing
end

# WF3x1GPU 전용 addmul!
function gpuaddmul!(out::WF3x1GPU, M::Spin_matrix_GPU, inp::WF3x1GPU)
    Ms = SMatrix{4,4,ComplexF64}(M.sm)  # isbits, 커널 인자로 OK
    CUDA.@cuda threads=out.blockinfo.blocksize blocks=out.blockinfo.rsize kernel_addmul_spinmat_mul!(out.f, Ms, inp.f)
    return out
end

"""
    _build_block_weight_gpu(W_col, r, x::WF3x1GPU)

BlockScalar_Layer용: GPU 페르미온 `.f`(shape NC×NG×blocksize×rsize)에
broadcast 곱셈 가능한 가중치 CuArray(shape 1×1×blocksize×rsize)를 생성.

`W_col[i]`는 블록 i의 가중치, `r`은 blockize 블록 크기.
Blockindices의 (b,r)→(ix,iy,iz,it) 매핑을 사용.
"""
function _build_block_weight_gpu(W_col::AbstractVector{<:Number}, r::Int, x::WF3x1GPU)
    bi = x.blockinfo
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NBX = NX ÷ r; NBY = NY ÷ r; NBZ = NZ ÷ r; NBT = NT ÷ r

    blocksize = bi.blocksize
    rsize     = bi.rsize

    w_cpu = zeros(ComplexF64, blocksize, rsize)

    @inbounds for r_idx = 1:rsize
        for b_idx = 1:blocksize
            ix = 1 + mod(div(b_idx-1, bi.blocks_s[1]), bi.blocks[1]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[1]), bi.blocknumbers[1]) * bi.blocks[1]
            iy = 1 + mod(div(b_idx-1, bi.blocks_s[2]), bi.blocks[2]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[2]), bi.blocknumbers[2]) * bi.blocks[2]
            iz = 1 + mod(div(b_idx-1, bi.blocks_s[3]), bi.blocks[3]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[3]), bi.blocknumbers[3]) * bi.blocks[3]
            it = 1 + mod(div(b_idx-1, bi.blocks_s[4]), bi.blocks[4]) +
                     mod(div(r_idx-1, bi.blocknumbers_s[4]), bi.blocknumbers[4]) * bi.blocks[4]

            bx = (ix - 1) ÷ r + 1
            by = (iy - 1) ÷ r + 1
            bz = (iz - 1) ÷ r + 1
            bt = (it - 1) ÷ r + 1
            block_id = bx + NBX * ((by - 1) + NBY * ((bz - 1) + NBZ * (bt - 1)))

            w_cpu[b_idx, r_idx] = W_col[block_id]
        end
    end

    return CuArray(reshape(w_cpu, 1, 1, blocksize, rsize))
end

"""
    _update_block_weight_gpu!(layer, W_col)

캐시된 맵으로 CPU 배열 갱신 → CuArray에 copyto! (할당 0)
"""
function _update_block_weight_gpu!(layer, W_col::AbstractVector{<:Number})
    bmap = layer._gpu_block_map
    w_cpu = layer._gpu_w_cpu
    @inbounds for i in eachindex(bmap)
        w_cpu[i] = W_col[bmap[i]]
    end
    cache = layer._gpu_w_buf
    if cache isa Dict
        if !haskey(cache, :expanded) || cache[:expanded] === nothing || size(cache[:expanded]) != (1, 1, size(w_cpu, 1), size(w_cpu, 2))
            cache[:expanded] = CUDA.zeros(ComplexF64, 1, 1, size(w_cpu, 1), size(w_cpu, 2))
        end
        copyto!(cache[:expanded], reshape(w_cpu, 1, 1, size(w_cpu)...))
        return cache[:expanded]
    else
        copyto!(layer._gpu_w_buf, reshape(w_cpu, 1, 1, size(w_cpu)...))
        return layer._gpu_w_buf
    end
end

function _blockscalar_grad_kernel!(
    grad_re::CuDeviceMatrix{Float64},
    grad_im::CuDeviceMatrix{Float64},
    z::CuDeviceVector{ComplexF64},
    d::CuDeviceVector{ComplexF64},
    bmap::CuDeviceVector{Int32},
    nc_ng::Int32,
    nsites::Int32,
    k::Int32,
)
    site = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if site <= nsites
        base = (site - Int32(1)) * nc_ng
        s = ComplexF64(0.0)
        for j in Int32(1):nc_ng
            @inbounds s += d[base + j] * z[base + j]
        end
        bid = bmap[site]
        cs = conj(s)
        CUDA.@atomic grad_re[bid, k] += real(cs)
        CUDA.@atomic grad_im[bid, k] += imag(cs)
    end
    return nothing
end

function _sync_blockmatrix_W_gpu!(layer; upload::Bool=true)
    cache = layer._gpu_w_buf
    if !(cache isa Dict) || !haskey(cache, :W) || size(cache[:W]) != size(layer.W)
        sites, counts = _blockscalar_sites_by_block(layer._gpu_block_map, size(layer.W, 1))
        cache = Dict{Symbol,Any}(
            :W => CuArray(layer.W),
            :bmap => CuArray(Int32.(layer._gpu_block_map)),
            :block_sites => CuArray(sites),
            :block_counts => CuArray(counts),
            :grad => nothing,
            :adam_m => _restored_adam_moment(layer._gpu_w_buf, :adam_m, layer.W),
            :adam_v => _restored_adam_moment(layer._gpu_w_buf, :adam_v, layer.W),
        )
        layer._gpu_w_buf = cache
    elseif upload
        copyto!(cache[:W], layer.W)
    end
    return cache[:W], cache[:bmap], cache
end

function _blockmatrix_apply_kernel!(
    out::CuDeviceVector{ComplexF64},
    xs::CuDeviceVector{ComplexF64},
    W::CuDeviceArray{ComplexF64,3},
    bmap::CuDeviceVector{Int32},
    nc_ng::Int32,
    field_len::Int32,
    total::Int32,
    nin::Int32,
    o::Int32,
    residual::Bool,
)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if tid <= total
        site = ((tid - Int32(1)) % field_len) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        s = ComplexF64(0.0)
        for i in Int32(1):nin
            s += W[bid, o, i] * xs[(i - Int32(1)) * field_len + tid]
        end
        if residual && o <= nin
            s += xs[(o - Int32(1)) * field_len + tid]
        end
        out[tid] = s
    end
    return nothing
end

function blockmatrix_apply_gpu!(out::AbstractVector, x::AbstractVector,
                                W::CuArray{ComplexF64,3}, bmap::CuArray{Int32,1},
                                residual::Bool)
    nout = size(W, 2)
    nin = size(W, 3)
    field_len = Int32(length(x[1].f))
    xs = CUDA.zeros(ComplexF64, Int(field_len) * nin)
    for i in 1:nin
        copyto!(view(xs, (i-1)*Int(field_len)+1:i*Int(field_len)), reshape(x[i].f, :))
    end
    total = field_len
    for o in 1:nout
        @cuda threads=256 blocks=cld(Int(total), 256) _blockmatrix_apply_kernel!(
            reshape(out[o].f, :), xs, W, bmap,
            Int32(size(x[1].f, 1) * size(x[1].f, 2)), field_len, total,
            Int32(nin), Int32(o), residual)
    end
    return out
end

function _blockmatrix_delta_kernel!(
    out::CuDeviceVector{ComplexF64},
    ds::CuDeviceVector{ComplexF64},
    W::CuDeviceArray{ComplexF64,3},
    bmap::CuDeviceVector{Int32},
    nc_ng::Int32,
    field_len::Int32,
    total::Int32,
    nout::Int32,
    i::Int32,
    residual::Bool,
)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if tid <= total
        site = ((tid - Int32(1)) % field_len) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        s = ComplexF64(0.0)
        for o in Int32(1):nout
            s += W[bid, o, i] * ds[(o - Int32(1)) * field_len + tid]
        end
        if residual && i <= nout
            s += ds[(i - Int32(1)) * field_len + tid]
        end
        out[tid] = s
    end
    return nothing
end

function blockmatrix_delta_gpu!(out::AbstractVector, d::AbstractVector,
                                W::CuArray{ComplexF64,3}, bmap::CuArray{Int32,1},
                                residual::Bool)
    nout = size(W, 2)
    nin = size(W, 3)
    field_len = Int32(length(d[1].f))
    ds = CUDA.zeros(ComplexF64, Int(field_len) * nout)
    for o in 1:nout
        copyto!(view(ds, (o-1)*Int(field_len)+1:o*Int(field_len)), reshape(d[o].f, :))
    end
    total = field_len
    for i in 1:nin
        @cuda threads=256 blocks=cld(Int(total), 256) _blockmatrix_delta_kernel!(
            reshape(out[i].f, :), ds, W, bmap,
            Int32(size(d[1].f, 1) * size(d[1].f, 2)), field_len, total,
            Int32(nout), Int32(i), residual)
    end
    return out
end

function _blockmatrix_grad_by_block_kernel!(
    grad::CuDeviceArray{ComplexF64,3},
    z::CuDeviceVector{ComplexF64},
    d::CuDeviceVector{ComplexF64},
    block_sites::CuDeviceMatrix{Int32},
    block_counts::CuDeviceVector{Int32},
    nc_ng::Int32,
    o_arg::Int32,
    i_arg::Int32,
)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sh = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz))

    s = ComplexF64(0.0)
    count = block_counts[bid]
    p = tid
    while p <= count
        site = block_sites[p, bid]
        base = (site - Int32(1)) * nc_ng
        for j in Int32(1):nc_ng
            @inbounds s += d[base + j] * z[base + j]
        end
        p += bsz
    end
    @inbounds sh[tid] = s
    sync_threads()

    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            @inbounds sh[tid] += sh[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end

    if tid == Int32(1)
        @inbounds grad[bid, o_arg, i_arg] = conj(sh[Int32(1)])
    end
    return nothing
end

function blockmatrix_gradient_gpu!(layer, z::AbstractVector, d::AbstractVector)
    cache = layer._gpu_w_buf
    if !(cache isa Dict) || !haskey(cache, :bmap)
        sites, counts = _blockscalar_sites_by_block(layer._gpu_block_map, size(layer.W, 1))
        cache = Dict{Symbol,Any}(
            :W => CuArray(layer.W),
            :bmap => CuArray(Int32.(layer._gpu_block_map)),
            :block_sites => CuArray(sites),
            :block_counts => CuArray(counts),
            :grad => nothing,
            :adam_m => _restored_adam_moment(layer._gpu_w_buf, :adam_m, layer.W),
            :adam_v => _restored_adam_moment(layer._gpu_w_buf, :adam_v, layer.W),
        )
        layer._gpu_w_buf = cache
    end
    nblocks, nout, nin = size(layer.W)
    if !haskey(cache, :block_sites) || cache[:block_sites] === nothing || size(cache[:block_sites], 2) != nblocks
        sites, counts = _blockscalar_sites_by_block(layer._gpu_block_map, nblocks)
        cache[:block_sites] = CuArray(sites)
        cache[:block_counts] = CuArray(counts)
    end
    if !haskey(cache, :grad) || cache[:grad] === nothing || size(cache[:grad]) != (nblocks, nout, nin)
        cache[:grad] = CUDA.zeros(ComplexF64, nblocks, nout, nin)
    end
    grad = cache[:grad]

    fref = z[1].f
    nc_ng = Int32(size(fref, 1) * size(fref, 2))
    for o in 1:nout, i in 1:nin
        @cuda threads=256 blocks=(nblocks, 1) shmem=(256 * sizeof(ComplexF64)) _blockmatrix_grad_by_block_kernel!(
            grad, reshape(z[i].f, :), reshape(d[o].f, :),
            cache[:block_sites], cache[:block_counts], nc_ng, Int32(o), Int32(i))
    end
    return grad
end

function _unpack_kernel!(ptrs_u64, src, inner::Int32)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    fid = Int32(blockIdx().y)
    if tid <= inner
        ptr = reinterpret(Core.LLVMPtr{ComplexF64, 1}, ptrs_u64[fid])
        unsafe_store!(ptr, src[(fid - Int32(1)) * inner + tid], tid)
    end
    return nothing
end

function blockmatrix_adam_update!(W_gpu, m_gpu, v_gpu, grad_gpu,
                                  alpha, beta1, beta2, eps, t)
    bias1 = 1.0 / (1.0 - beta1^t)
    bias2 = 1.0 / (1.0 - beta2^t)
    total = length(W_gpu)
    threads = min(256, total)
    nblocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_adam_complex_standard!(
        W_gpu, m_gpu, v_gpu, grad_gpu, alpha, beta1, beta2, eps, bias1, bias2)
end

# Fused kernel: positive direction (a > 0)
# y(b,r) = U†_a(bs,rs) * x(bs,rs) * factor
function _cudakernel_fused_hop_pos!(y, A, x, shift, blockinfo, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    bs, rs = _shifted_idx(b, r, shift, blockinfo)
    fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
    @inbounds for alpha = 1:4
        x1 = x[1, alpha, bs, rs] * fac
        x2 = x[2, alpha, bs, rs] * fac
        x3 = x[3, alpha, bs, rs] * fac
        y[1, alpha, b, r] = conj(A[1,1,bs,rs])*x1 + conj(A[2,1,bs,rs])*x2 + conj(A[3,1,bs,rs])*x3
        y[2, alpha, b, r] = conj(A[1,2,bs,rs])*x1 + conj(A[2,2,bs,rs])*x2 + conj(A[3,2,bs,rs])*x3
        y[3, alpha, b, r] = conj(A[1,3,bs,rs])*x1 + conj(A[2,3,bs,rs])*x2 + conj(A[3,3,bs,rs])*x3
    end
    return nothing
end

# Fused kernel: negative direction (a < 0, aa = -a)
# y(b,r) = U_aa(b,r) * x(bs,rs) * factor
function _cudakernel_fused_hop_neg!(y, A, x, shift, blockinfo, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    bs, rs = _shifted_idx(b, r, shift, blockinfo)
    fac = _bc_factor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
    @inbounds for alpha = 1:4
        x1 = x[1, alpha, bs, rs] * fac
        x2 = x[2, alpha, bs, rs] * fac
        x3 = x[3, alpha, bs, rs] * fac
        y[1, alpha, b, r] = A[1,1,b,r]*x1 + A[1,2,b,r]*x2 + A[1,3,b,r]*x3
        y[2, alpha, b, r] = A[2,1,b,r]*x1 + A[2,2,b,r]*x2 + A[2,3,b,r]*x3
        y[3, alpha, b, r] = A[3,1,b,r]*x1 + A[3,2,b,r]*x2 + A[3,3,b,r]*x3
    end
    return nothing
end

@inline function _dir_to_shift(a::Int)
    s1 = Int8(ifelse(abs(a) == 1, sign(a), 0))
    s2 = Int8(ifelse(abs(a) == 2, sign(a), 0))
    s3 = Int8(ifelse(abs(a) == 3, sign(a), 0))
    s4 = Int8(ifelse(abs(a) == 4, sign(a), 0))
    return (s1, s2, s3, s4)
end

const _shared_float_pool = Dict{Tuple{Symbol,Int}, CuArray{Float64,5}}()

function get_shared_float_buf!(key::Symbol, NC::Int, NG::Int, bs::Int, rs::Int, n::Int)
    k = (key, n)
    buf = get(_shared_float_pool, k, nothing)
    if buf === nothing || size(buf, 3) != bs || size(buf, 4) != rs
        buf = CUDA.zeros(Float64, NC, NG, bs, rs, n)
        _shared_float_pool[k] = buf
    end
    return buf
end

# ============================================================
#  Batched norm²/dot kernels — NormLayer fwd/bwd 전용
#  map!(abs2,…)+sum(…;dims=1:4) 을 완전히 대체. 0 allocation.
# ============================================================

function _batch_normsq_kernel!(
    out::CuDeviceVector{Float64},
    x::CuDeviceVector{ComplexF64},
    inner::Int32
)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sdata = CUDA.@cuDynamicSharedMem(Float64, Int(bsz))
    s = 0.0
    base = (bid - Int32(1)) * inner
    j = tid
    while j <= inner
        @inbounds v = x[base + j]
        s += real(v)^2 + imag(v)^2
        j += bsz
    end
    @inbounds sdata[tid] = s
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            @inbounds sdata[tid] += sdata[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        @inbounds out[bid] = sdata[Int32(1)]
    end
    nothing
end

function _batch_normsq!(out::CuVector{Float64}, x::CuArray{ComplexF64,5})
    n     = length(out)
    inner = Int32(length(x) ÷ n)
    @cuda threads=256 blocks=n shmem=(256*sizeof(Float64)) _batch_normsq_kernel!(
        out, reshape(x, :), inner)
end

function _normalize_batch_kernel!(
    out::CuDeviceVector{ComplexF64},
    x::CuDeviceVector{ComplexF64},
    norms::CuDeviceVector{Float64},
    alpha::Float64,
    inner::Int32,
    n::Int32
)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if tid <= inner * n
        batch_i = (tid - Int32(1)) ÷ inner + Int32(1)
        @inbounds scale = alpha / sqrt(norms[batch_i])
        @inbounds out[tid] = scale * x[tid]
    end
    nothing
end

function _normalize_batch!(out::CuArray{ComplexF64,5}, x::CuArray{ComplexF64,5},
                           norms::CuVector{Float64}, alpha::Float64)
    n     = Int32(length(norms))
    inner = Int32(length(x) ÷ Int(n))
    total = Int(inner) * Int(n)
    @cuda threads=256 blocks=cld(total,256) _normalize_batch_kernel!(
        reshape(out,:), reshape(x,:), norms, alpha, inner, n)
end

function _batch_normsq_dot_kernel!(
    normsq::CuDeviceVector{Float64},
    dotprod::CuDeviceVector{ComplexF64},
    z::CuDeviceVector{ComplexF64},
    d::CuDeviceVector{ComplexF64},
    inner::Int32
)
    bid = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sn = CUDA.@cuDynamicSharedMem(Float64,    Int(bsz))
    sd = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz), Int(bsz) * sizeof(Float64))
    sn_val = 0.0
    sd_val = ComplexF64(0.0)
    base = (bid - Int32(1)) * inner
    j = tid
    while j <= inner
        @inbounds zj = z[base + j]
        @inbounds dj = d[base + j]
        sn_val += real(zj)^2 + imag(zj)^2
        sd_val += zj * dj
        j += bsz
    end
    @inbounds sn[tid] = sn_val
    @inbounds sd[tid] = sd_val
    sync_threads()
    half = bsz >> Int32(1)
    while half > Int32(0)
        if tid <= half
            @inbounds sn[tid] += sn[tid + half]
            @inbounds sd[tid] += sd[tid + half]
        end
        sync_threads()
        half = half >> Int32(1)
    end
    if tid == Int32(1)
        @inbounds normsq[bid]  = sn[Int32(1)]
        @inbounds dotprod[bid] = sd[Int32(1)]
    end
    nothing
end

function _batch_normsq_dot!(normsq::CuVector{Float64}, dotprod::CuVector{ComplexF64},
                            z::CuArray{ComplexF64,5}, d::CuArray{ComplexF64,5})
    n     = length(normsq)
    inner = Int32(length(z) ÷ n)
    shmem = 256 * (sizeof(Float64) + sizeof(ComplexF64))
    @cuda threads=256 blocks=n shmem=shmem _batch_normsq_dot_kernel!(
        normsq, dotprod, reshape(z,:), reshape(d,:), inner)
end

function _backward_normalize_kernel!(
    out::CuDeviceVector{ComplexF64},
    z::CuDeviceVector{ComplexF64},
    d::CuDeviceVector{ComplexF64},
    normsq::CuDeviceVector{Float64},
    dotprod::CuDeviceVector{ComplexF64},
    inner::Int32,
    n::Int32
)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if tid <= inner * n
        batch_i = (tid - Int32(1)) ÷ inner + Int32(1)
        @inbounds nsq = normsq[batch_i]
        @inbounds dt  = dotprod[batch_i]
        inv_n = 1.0 / sqrt(nsq)
        coeff = -real(dt) * inv_n^3
        @inbounds out[tid] = inv_n * d[tid] + coeff * conj(z[tid])
    end
    nothing
end

function _backward_normalize!(out::CuArray{ComplexF64,5}, z::CuArray{ComplexF64,5},
                              d::CuArray{ComplexF64,5},
                              normsq::CuVector{Float64}, dotprod::CuVector{ComplexF64})
    n     = Int32(length(normsq))
    inner = Int32(length(z) ÷ Int(n))
    total = Int(inner) * Int(n)
    @cuda threads=256 blocks=cld(total,256) _backward_normalize_kernel!(
        reshape(out,:), reshape(z,:), reshape(d,:), normsq, dotprod, inner, n)
end

"""
    fused_hopping_gpu!(V, x, Gauge, paths_various)

GPU hopping with fused kernels, double-buffering via f ↔ fshifted.
Per step: 3 kernel launches + 3 CUDA.@sync → 1 kernel + 0 sync.
"""
function fused_hopping_gpu!(V::AbstractVector{<:WF3x1GPU}, x::WF3x1GPU,
                            Gauge::GPUgauge, paths_various)
    blocksize = x.blockinfo.blocksize
    rsize     = x.blockinfo.rsize
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    bi = x.blockinfo
    bc = _BC_DEFAULT

    @inbounds for k in eachindex(V)
        copyx = V[k]
        steps = paths_various[k].steps

        # Initialize copyx from x (GPU memcpy, no kernel)
        copyto!(copyx.f, x.f)

        nsteps = length(steps)
        if nsteps == 0 || (nsteps == 1 && steps[1].f == 0)
            continue   # identity path
        end

        # Double-buffering: f ↔ fshifted, no intermediate sync
        buf_a = copyx.f
        buf_b = copyx.fshifted
        flip = false

        for step in steps
            a = step.f
            src = flip ? buf_b : buf_a
            dst = flip ? buf_a : buf_b

            if a > 0
                shift = _dir_to_shift(-a)
                CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_fused_hop_pos!(
                    dst, Gauge[a].U, src, shift, bi, bc, NX, NY, NZ, NT)
            elseif a < 0
                aa = -a
                shift = _dir_to_shift(aa)
                CUDA.@cuda threads=blocksize blocks=rsize _cudakernel_fused_hop_neg!(
                    dst, Gauge[aa].U, src, shift, bi, bc, NX, NY, NZ, NT)
            else
                # a == 0: reset to original x
                copyto!(dst, x.f)
            end
            flip = !flip
        end

        # If result is in fshifted (odd steps), copy back to f
        if flip
            copyto!(copyx.f, copyx.fshifted)
        end
    end
end
