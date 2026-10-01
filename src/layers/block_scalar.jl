# BlockScalar_Layer: multiplies each field by complex scalars that are constant
# on blocks of r^4 sites.

"""
    BlockScalar_Layer(x, n, r, residual=false, activation=Identity())

Block-wise scalar layer for `n` fields shaped like `x`. The lattice is divided
into blocks of `r^4` sites, and field `k` is multiplied by the weight of the
block that contains each site:

    y[k](site) = W[block(site), k] * x[k](site)        (+ x[k](site) if residual)

`W` is an `nblocks x n` complex matrix, initialised to ones.

Backward pass: `delta[k](site) = W[block(site), k] * d[k](site)` (plus `d[k]`
with the residual connection).

Weight gradient: `dL/dW[b, k] = sum over sites in block b and over spin and
colour of d[k] * z[k]`.
"""
mutable struct BlockScalar_Layer <: Layer
    W::Matrix{ComplexF64}     # (nblocks, n): weight of each block for each field
    r::Int                    # block edge length
    residual::Bool
    activation::ActivationFunction
    # Buffers
    _result::Vector{WF3x1}    # output fields of the forward pass and DeltaRule
    _blocks::Vector{Vector{WF3x1}}  # unused
    _block_map::Vector{Int}   # CPU: block id of every site, x fastest
    _gpu_block_map::Vector{Int}  # GPU: block id of every (site in block, block) index
    _gpu_w_buf::Any           # GPU cache Dict: weights, block maps, gradient, Adam moments
    _gpu_w_cpu::Matrix{ComplexF64}  # (blocksize, rsize) staging array
    _initialized::Bool
    _grad_prod_buf::Any       # unused
    _grad_site_buf::Any       # unused
    _grad_site_cpu::Vector{ComplexF64}  # unused
    _grad_W::Matrix{ComplexF64}  # CPU gradient, (nblocks, n)

    function BlockScalar_Layer(x::WF3x1, n::Int, r::Int,
                               residual::Bool=false,
                               activation::ActivationFunction=Identity())
        nblocks = (x.NX ÷ r) * (x.NY ÷ r) * (x.NZ ÷ r) * (x.NT ÷ r)
        W = ones(ComplexF64, nblocks, n)
        new(W, r, residual, activation,
            WF3x1[], Vector{WF3x1}[], Int[], Int[], nothing,
            Matrix{ComplexF64}(undef, 0, 0), false,
            nothing, nothing, ComplexF64[], Matrix{ComplexF64}(undef, 0, 0))
    end
end

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

function (layer::BlockScalar_Layer)(x::AbstractVector{<:WF3x1})
    result = _ensure_block_buffers!(layer, x)
    _apply_block_scalars!(result, x, layer)
    return result
end

# Allocates the output buffers and the site-to-block map on first use, and
# again when the number of fields or their device changes.
function _ensure_block_buffers!(layer::BlockScalar_Layer, fields)
    n = length(fields)
    if !layer._initialized || length(layer._result) != n || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(fields[1]))
        layer._result = Vector{WF3x1}(undef, n)
        for k in 1:n
            layer._result[k] = deepcopy(fields[k])
        end
        if fields[1] isa WF3x1GPU
            _init_gpu_block_cache!(layer, fields[1])
        else
            layer._block_map = _cpu_block_map(fields[1], layer.r)
        end
        layer._initialized = true
    end
    return layer._result
end

# Block id (1-based) of every site, in the order ix, iy, iz, it with ix fastest.
function _cpu_block_map(x::WF3x1, r::Int)
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NBX = div(NX, r); NBY = div(NY, r); NBZ = div(NZ, r)
    bmap = Vector{Int}(undef, NX * NY * NZ * NT)
    idx = 0
    @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
        idx += 1
        bx = div(ix - 1, r) + 1; by = div(iy - 1, r) + 1
        bz = div(iz - 1, r) + 1; bt = div(it - 1, r) + 1
        bmap[idx] = bx + NBX*((by-1) + NBY*((bz-1) + NBZ*(bt-1)))
    end
    return bmap
end

# out[k](site) = W[block(site), k] * x[k](site), plus x[k] with the residual
# connection. Used by the forward pass and, with d in place of x, the backward pass.
function _apply_block_scalars!(out, x, layer::BlockScalar_Layer)
    n = length(x)
    if x[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockscalar_W_gpu!(layer; upload=false)
        for k in 1:n
            blockscalar_apply_gpu!(out[k].f, x[k].f, W_gpu, bmap_gpu, k, layer.residual)
        end
    else
        NC, NG = x[1].NC, x[1].NG
        NX, NY, NZ, NT = x[1].NX, x[1].NY, x[1].NZ, x[1].NT
        bmap = layer._block_map
        W = layer.W

        for k in 1:n
            ok = out[k]
            xk = x[k]
            site = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                w = W[bmap[site], k]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        ok[ic, ix, iy, iz, it, ig] = w * xk[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            if layer.residual
                axpby!(1, xk, 1, ok)
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

function DeltaRule!(layer::BlockScalar_Layer, ::Identity,
                    z::Vector{<:WF3x1}, d::Vector{T}, delta::Vector{<:WF3x1}) where T<:WF3x1
    _apply_block_scalars!(delta, d, layer)
    return nothing
end

# Writes into the layer's output buffer, like the forward pass.
function DeltaRule(layer::BlockScalar_Layer, ::Identity,
                   z::Vector{<:WF3x1}, d::Vector{T}) where T<:WF3x1
    result = _ensure_block_buffers!(layer, d)
    return _apply_block_scalars!(result, d, layer)
end

# ---------------------------------------------------------------------------
# Weight gradient
# ---------------------------------------------------------------------------

# Returns [grad] with grad an (nblocks, n) CuArray on the GPU, and the
# (nblocks, n) matrix itself on the CPU.
function gradient(layer::BlockScalar_Layer, ::Identity, z::Vector{<:WF3x1}, d::Vector{T}, γ::Vector{Spin_matrix}) where T<:WF3x1
    if z[1] isa WF3x1GPU
        return [blockscalar_gradient_gpu!(layer, z, d)]
    end

    nblocks, n = size(layer.W)
    if size(layer._grad_W) != (nblocks, n)
        layer._grad_W = zeros(ComplexF64, nblocks, n)
    else
        fill!(layer._grad_W, zero(ComplexF64))
    end
    grad_W = layer._grad_W

    NC, NG = z[1].NC, z[1].NG
    NX, NY, NZ, NT = z[1].NX, z[1].NY, z[1].NZ, z[1].NT
    bmap = layer._block_map

    for k in 1:n
        zk = z[k]; dk = d[k]
        site = 0
        @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
            site += 1
            bid = bmap[site]
            s = zero(ComplexF64)
            for ig in 1:NG
                @simd for ic in 1:NC
                    s += dk[ic, ix, iy, iz, it, ig] * zk[ic, ix, iy, iz, it, ig]
                end
            end
            grad_W[bid, k] += s
        end
    end

    return grad_W
end

# ---------------------------------------------------------------------------
# Adam update
# ---------------------------------------------------------------------------

# CPU: Adam on the real and imaginary part of every weight separately.
function update!(layer::BlockScalar_Layer, opt::Adam, grad::Matrix{ComplexF64})
    for idx in eachindex(layer.W)
        re_key = objectid(layer) + UInt(2idx - 1)
        st_re = get!(opt.scalar_state, re_key) do
            AdamState{Float64}(0.0, 0.0)
        end
        g_re = real(grad[idx])
        st_re.m = opt.β1 * st_re.m + (1 - opt.β1) * g_re
        st_re.v = opt.β2 * st_re.v + (1 - opt.β2) * g_re^2
        m_hat_re = st_re.m / (1 - opt.β1^opt.t)
        v_hat_re = st_re.v / (1 - opt.β2^opt.t)
        Δ_re = opt.α * m_hat_re / (sqrt(v_hat_re) + opt.ε)

        im_key = objectid(layer) + UInt(2idx)
        st_im = get!(opt.scalar_state, im_key) do
            AdamState{Float64}(0.0, 0.0)
        end
        g_im = imag(grad[idx])
        st_im.m = opt.β1 * st_im.m + (1 - opt.β1) * g_im
        st_im.v = opt.β2 * st_im.v + (1 - opt.β2) * g_im^2
        m_hat_im = st_im.m / (1 - opt.β1^opt.t)
        v_hat_im = st_im.v / (1 - opt.β2^opt.t)
        Δ_im = opt.α * m_hat_im / (sqrt(v_hat_im) + opt.ε)

        layer.W[idx] -= complex(Δ_re, -Δ_im)
    end
end

# GPU: grad is [CuArray]; the weights and moments live in the GPU cache.
function update!(layer::BlockScalar_Layer, opt::Adam, grad)
    if grad isa AbstractVector && !isempty(grad) && grad[1] isa CuArray
        cache = layer._gpu_w_buf
        if !(cache isa Dict) || !haskey(cache, :W)
            error("BlockScalar_Layer GPU weights are not initialized. Run the GPU forward pass before update!.")
        end
        W_gpu = cache[:W]
        grad_gpu = grad[1]
        if !haskey(cache, :adam_m) || cache[:adam_m] === nothing || size(cache[:adam_m]) != size(W_gpu)
            cache[:adam_m] = CUDA.zeros(ComplexF64, size(W_gpu)...)
            cache[:adam_v] = CUDA.zeros(ComplexF64, size(W_gpu)...)
        end
        blockscalar_adam_update!(W_gpu, cache[:adam_m], cache[:adam_v], grad_gpu,
                        opt.α, opt.β1, opt.β2, opt.ε, opt.t)
        return layer
    end
    return update!(layer, opt, grad::Matrix{ComplexF64})
end

function blockscalar_adam_update!(W_gpu, m_gpu, v_gpu, grad_gpu,
                                  alpha, beta1, beta2, eps, t)
    bias1 = 1.0 / (1.0 - beta1^t)
    bias2 = 1.0 / (1.0 - beta2^t)
    total = length(W_gpu)
    threads = min(256, total)
    nblocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=nblocks _cudakernel_adam_complex!(
        W_gpu, m_gpu, v_gpu, grad_gpu, alpha, beta1, beta2, eps, bias1, bias2)
end

# ---------------------------------------------------------------------------
# CPU/GPU conversion: both directions drop the device-specific caches
# ---------------------------------------------------------------------------

function _to_cpu(layer::BlockScalar_Layer)
    layer._gpu_w_buf = nothing
    layer._gpu_w_cpu = Matrix{ComplexF64}(undef, 0, 0)
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._initialized = false
    return layer
end

function _to_gpu(layer::BlockScalar_Layer, blocks=nothing)
    layer._gpu_w_buf = nothing
    layer._gpu_w_cpu = Matrix{ComplexF64}(undef, 0, 0)
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._block_map = Int[]
    layer._initialized = false
    return layer
end

# ---------------------------------------------------------------------------
# GPU cache
# ---------------------------------------------------------------------------

# load_checkpoint! can restore Adam moments before the GPU cache of a block
# layer exists. Carry them into the cache when it is built, so a resumed run
# continues with the saved optimizer state.
function _restored_adam_moment(cache, key::Symbol, W)
    cache isa Dict || return nothing
    moment = get(cache, key, nothing)
    (moment === nothing || size(moment) != size(W)) && return nothing
    return moment isa CuArray ? moment : CuArray{ComplexF64}(moment)
end

"""
    _init_gpu_block_cache!(layer, x)

Builds the GPU block map for the blocked layout of `x`, where
`layer._gpu_block_map[linear_idx]` is the block id of each site, and the GPU
cache of the weights. Later calls only need to update the weights.
"""
function _init_gpu_block_cache!(layer, x::WF3x1GPU)
    bi = x.blockinfo
    r = layer.r
    NX, NY, NZ, NT = x.NX, x.NY, x.NZ, x.NT
    NBX = NX÷r; NBY = NY÷r; NBZ = NZ÷r; NBT = NT÷r
    blocksize = bi.blocksize
    rsize = bi.rsize

    bmap = Vector{Int}(undef, blocksize * rsize)
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
            bx = (ix-1)÷r + 1; by = (iy-1)÷r + 1
            bz = (iz-1)÷r + 1; bt = (it-1)÷r + 1
            bmap[idx] = bx + NBX*((by-1) + NBY*((bz-1) + NBZ*(bt-1)))
        end
    end
    sites, counts = _blockscalar_sites_by_block(bmap, size(layer.W, 1))
    layer._gpu_block_map = bmap
    layer._gpu_w_cpu = zeros(ComplexF64, blocksize, rsize)
    layer._gpu_w_buf = Dict{Symbol,Any}(
        :W => CuArray(layer.W),
        :bmap => CuArray(Int32.(bmap)),
        :block_sites => CuArray(sites),
        :block_counts => CuArray(counts),
        :grad => nothing,
        :adam_m => _restored_adam_moment(layer._gpu_w_buf, :adam_m, layer.W),
        :adam_v => _restored_adam_moment(layer._gpu_w_buf, :adam_v, layer.W),
    )
end

# Returns the GPU weights and block map, building the cache if needed. With
# upload=true it also copies layer.W to the GPU.
function _sync_blockscalar_W_gpu!(layer; upload::Bool=true)
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

# Inverse of the block map: column b of `sites` lists the sites of block b,
# and counts[b] how many there are.
function _blockscalar_sites_by_block(bmap::Vector{Int}, nblocks::Int)
    counts = zeros(Int32, nblocks)
    @inbounds for bid in bmap
        counts[bid] += Int32(1)
    end
    max_count = maximum(counts)
    sites = zeros(Int32, Int(max_count), nblocks)
    pos = zeros(Int32, nblocks)
    @inbounds for site in eachindex(bmap)
        bid = bmap[site]
        pos[bid] += Int32(1)
        sites[Int(pos[bid]), bid] = Int32(site)
    end
    return sites, counts
end

# ---------------------------------------------------------------------------
# GPU kernels
# ---------------------------------------------------------------------------

# out = W[block, k] * x (+ x), one thread per field component.
function _blockscalar_apply_kernel!(
    out::CuDeviceVector{ComplexF64},
    x::CuDeviceVector{ComplexF64},
    W::CuDeviceMatrix{ComplexF64},
    bmap::CuDeviceVector{Int32},
    nc_ng::Int32,
    total::Int32,
    k::Int32,
    residual::Bool,
)
    tid = Int32(threadIdx().x) + (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x)
    if tid <= total
        site = (tid - Int32(1)) ÷ nc_ng + Int32(1)
        bid = bmap[site]
        v = W[bid, k] * x[tid]
        if residual
            v += x[tid]
        end
        out[tid] = v
    end
    return nothing
end

function blockscalar_apply_gpu!(out::CuArray{ComplexF64,4}, x::CuArray{ComplexF64,4},
                                W::CuArray{ComplexF64,2}, bmap::CuArray{Int32,1},
                                k::Int, residual::Bool)
    nc_ng = Int32(size(x, 1) * size(x, 2))
    total = Int32(length(x))
    @cuda threads=256 blocks=cld(Int(total), 256) _blockscalar_apply_kernel!(
        reshape(out, :), reshape(x, :), W, bmap, nc_ng, total, Int32(k), residual)
    return out
end

# grad[b, k] = sum over the sites of block b of d * z. One thread block per
# lattice block, reduced in shared memory.
function _blockscalar_grad_by_block_kernel!(
    grad::CuDeviceMatrix{ComplexF64},
    z::CuDeviceVector{ComplexF64},
    d::CuDeviceVector{ComplexF64},
    block_sites::CuDeviceMatrix{Int32},
    block_counts::CuDeviceVector{Int32},
    nc_ng::Int32,
    k_arg::Int32,
)
    bid = Int32(blockIdx().x)
    k = k_arg
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
        @inbounds grad[bid, k] = sh[Int32(1)]
    end
    return nothing
end

function blockscalar_gradient_gpu!(layer, z::AbstractVector, d::AbstractVector)
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
    nblocks, n = size(layer.W)
    if !haskey(cache, :block_sites) || cache[:block_sites] === nothing || size(cache[:block_sites], 2) != nblocks
        sites, counts = _blockscalar_sites_by_block(layer._gpu_block_map, nblocks)
        cache[:block_sites] = CuArray(sites)
        cache[:block_counts] = CuArray(counts)
    end
    block_sites = cache[:block_sites]
    block_counts = cache[:block_counts]
    if !haskey(cache, :grad) || cache[:grad] === nothing || size(cache[:grad]) != (nblocks, n)
        cache[:grad] = CUDA.zeros(ComplexF64, nblocks, n)
    end
    grad = cache[:grad]

    fref = z[1].f
    nc_ng = Int32(size(fref, 1) * size(fref, 2))
    for k in 1:n
        @cuda threads=256 blocks=(nblocks, 1) shmem=(256 * sizeof(ComplexF64)) _blockscalar_grad_by_block_kernel!(
            grad, reshape(z[k].f, :), reshape(d[k].f, :),
            block_sites, block_counts, nc_ng, Int32(k))
    end
    return grad
end
