# BlockScalar_Layer: multiplies each field by a complex scalar that is constant
# on blocks of r^4 sites.

"""
    BlockScalar_Layer

blockize로 블록화된 벡터 각각에 스칼라(복소수) 가중치를 곱하는 학습 레이어.

# Fields
- `W::Vector{ComplexF64}` — 블록 개수만큼의 스칼라 가중치
- `r::Int` — blockize에 사용되는 블록 크기
- `residual::Bool` — residual connection 활성화 여부
- `activation::ActivationFunction` — 활성화 함수
"""
mutable struct BlockScalar_Layer <: Layer
    W::Matrix{ComplexF64}   # (nblocks, n) — 블록별 × 벡터별 스칼라 가중치
    r::Int
    residual::Bool
    activation::ActivationFunction
    # --- 캐시 버퍼 (매 호출 deepcopy/blockize 방지) ---
    _result::Vector{WF3x1}      # forward/DeltaRule 출력 버퍼
    _blocks::Vector{Vector{WF3x1}}  # 벡터별 blockize 결과 캐시 (CPU)
    _block_map::Vector{Int}     # site→block_id 인덱스 맵 (CPU, 전체 사이트 수)
    _gpu_block_map::Vector{Int} # (b_idx, r_idx) → block_id 맵 (GPU용, blocksize*rsize)
    _gpu_w_buf::Any             # CuArray{ComplexF64,4} 재사용 버퍼 (1,1,blocksize,rsize)
    _gpu_w_cpu::Matrix{ComplexF64}  # CPU 작업 배열 (blocksize, rsize)
    _initialized::Bool
    _grad_prod_buf::Any             # CuArray{ComplexF64,4} for gradient elementwise product
    _grad_site_buf::Any             # CuArray for site-level sum result
    _grad_site_cpu::Vector{ComplexF64}  # CPU buffer for site inner products
    _grad_W::Matrix{ComplexF64}         # pre-allocated gradient accumulator (nblocks, n)

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

# --- BlockScalar_Layer: reset GPU caches before reuse ---
function _to_cpu(layer::BlockScalar_Layer)
    layer._gpu_w_buf = nothing
    layer._gpu_w_cpu = Matrix{ComplexF64}(undef, 0, 0)
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._initialized = false
    return layer
end

# --- BlockScalar_Layer: reset GPU caches before reuse ---
function _to_gpu(layer::BlockScalar_Layer, blocks=nothing)
    layer._gpu_w_buf = nothing
    layer._gpu_w_cpu = Matrix{ComplexF64}(undef, 0, 0)
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._block_map = Int[]
    layer._initialized = false
    return layer
end

function (layer::BlockScalar_Layer)(x::AbstractVector{<:WF3x1})
    n = length(x)
    nblocks = size(layer.W, 1)
    r = layer.r

    # GPU path for batch size 1.
    if !layer._initialized || length(layer._result) != n || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(x[1]))
        layer._result = Vector{WF3x1}(undef, n)
        for k in 1:n
            layer._result[k] = deepcopy(x[k])
        end
        if x[1] isa WF3x1GPU
            # Prepare GPU block map and CuArray weights.
            _init_gpu_block_cache!(layer, x[1])
        else
            # CPU block_map stores 1-based block ids.
            NX, NY, NZ, NT = x[1].NX, x[1].NY, x[1].NZ, x[1].NT
            NBX = div(NX, r); NBY = div(NY, r); NBZ = div(NZ, r); NBT = div(NT, r)
            nsites = NX * NY * NZ * NT
            layer._block_map = Vector{Int}(undef, nsites)
            idx = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                idx += 1
                bx = div(ix - 1, r) + 1; by = div(iy - 1, r) + 1
                bz = div(iz - 1, r) + 1; bt = div(it - 1, r) + 1
                layer._block_map[idx] = bx + NBX*((by-1) + NBY*((bz-1) + NBZ*(bt-1)))
            end
        end
        layer._initialized = true
    end

    result = layer._result

    if x[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockscalar_W_gpu!(layer; upload=false)
        for k in 1:n
            blockscalar_apply_gpu!(result[k].f, x[k].f, W_gpu, bmap_gpu, k, layer.residual)
        end
    else
        # CPU path uses block_map with blockize/unblockize.
        NC, NG = x[1].NC, x[1].NG
        NX, NY, NZ, NT = x[1].NX, x[1].NY, x[1].NZ, x[1].NT
        bmap = layer._block_map
        W = layer.W

        for k in 1:n
            rk = result[k]
            xk = x[k]
            site = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                w = W[bmap[site], k]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        rk[ic, ix, iy, iz, it, ig] = w * xk[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            if layer.residual
                axpby!(1, xk, 1, rk)
            end
        end
    end

    return result
end

# --- BlockScalar_Layer ---
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

    if false
        # GPU: preallocated buffers로 사이트별 내적 → 블록별 합산 (alloc 0)
        blockscalar_gradient_gpu!(grad_W, layer, z, d)
        return grad_W
        bmap = layer._gpu_block_map
        nblocks_total = length(bmap)
        fref = d[1].f  # CuArray{ComplexF64,4} reference

        # 1회만 버퍼 초기화
        if layer._grad_prod_buf === nothing || size(layer._grad_prod_buf) != size(fref)
            layer._grad_prod_buf = CUDA.zeros(ComplexF64, size(fref)...)
            layer._grad_site_buf = CUDA.zeros(ComplexF64, 1, 1, size(fref,3), size(fref,4))
            layer._grad_site_cpu = Vector{ComplexF64}(undef, nblocks_total)
        end
        prod_buf = layer._grad_prod_buf
        site_buf = layer._grad_site_buf

        for k in 1:n
            prod_buf .= d[k].f .* z[k].f
            _sum_spin_color!(site_buf, prod_buf)
            copyto!(layer._grad_site_cpu, vec(site_buf))

            @inbounds for i in 1:nblocks_total
                # grad_W[bmap[i], k] += conj(layer._grad_site_cpu[i])
                grad_W[bmap[i], k] += layer._grad_site_cpu[i]
            end
        end
    else
        # CPU: block_map으로 직접 내적 (blockize 제거)
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
                        # inner_product(conj(d), z) = Σ d*z, then conj → ∂L/∂W*
                        s += dk[ic, ix, iy, iz, it, ig] * zk[ic, ix, iy, iz, it, ig]
                    end
                end
                # grad_W[bid, k] += conj(s)
                grad_W[bid, k] += s
            end
        end
    end

    return grad_W
end

function DeltaRule(layer::BlockScalar_Layer, ::Identity,
                   z::Vector{<:WF3x1}, d::Vector{T}) where T<:WF3x1
    n = length(d)
    nblocks = size(layer.W, 1)
    r = layer.r

    # 출력 버퍼 캐시 (deepcopy 1회만, 또는 GPU↔CPU 전환 시 재초기화)
    if !layer._initialized || length(layer._result) != n || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(d[1]))
        layer._result = Vector{WF3x1}(undef, n)
        for k in 1:n
            layer._result[k] = deepcopy(d[k])
        end
        if d[1] isa WF3x1GPU
            _init_gpu_block_cache!(layer, d[1])
        elseif isempty(layer._block_map)
            NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
            NBX = NX÷r; NBY = NY÷r; NBZ = NZ÷r; NBT = NT÷r
            nsites = NX * NY * NZ * NT
            layer._block_map = Vector{Int}(undef, nsites)
            idx = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                idx += 1
                bx = (ix-1)÷r + 1; by = (iy-1)÷r + 1
                bz = (iz-1)÷r + 1; bt = (it-1)÷r + 1
                layer._block_map[idx] = bx + NBX*((by-1) + NBY*((bz-1) + NBZ*(bt-1)))
            end
        end
        layer._initialized = true
    end
    result = layer._result

    if d[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockscalar_W_gpu!(layer; upload=false)
        for k in 1:n
            blockscalar_apply_gpu!(result[k].f, d[k].f, W_gpu, bmap_gpu, k, layer.residual)
        end
    else
        # CPU: block_map으로 직접 스칼라 곱 (blockize/unblockize 제거)
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        for k in 1:n
            rk = result[k]
            dk = d[k]
            site = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                w = W[bmap[site], k]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        rk[ic, ix, iy, iz, it, ig] = w * dk[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            if layer.residual
                axpby!(1, dk, 1, rk)
            end
        end
    end
    return result
end

function DeltaRule!(layer::BlockScalar_Layer, ::Identity,
                    z::Vector{<:WF3x1}, d::Vector{T}, delta::Vector{<:WF3x1}) where T<:WF3x1
    n = length(d)
    nblocks = size(layer.W, 1)
    r = layer.r

    if d[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockscalar_W_gpu!(layer; upload=false)
        for k in 1:n
            blockscalar_apply_gpu!(delta[k].f, d[k].f, W_gpu, bmap_gpu, k, layer.residual)
        end
    else
        # CPU: block_map으로 직접 스칼라 곱 (blockize/unblockize 제거)
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        for k in 1:n
            dk_out = delta[k]
            dk = d[k]
            site = 0
            @inbounds for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                w = W[bmap[site], k]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        dk_out[ic, ix, iy, iz, it, ig] = w * dk[ic, ix, iy, iz, it, ig]
                    end
                end
            end
            if layer.residual
                axpby!(1, dk, 1, dk_out)
            end
        end
    end
end

function update!(layer::BlockScalar_Layer, opt::Adam, grad::Matrix{ComplexF64})
    for idx in eachindex(layer.W)
        # real part
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

        # imag part
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

"""
    _init_gpu_block_cache!(layer, x)

GPU용 block 인덱스 맵 + 버퍼를 1회만 초기화.
`layer._gpu_block_map[linear_idx]` = block_id 로, W 값만 갱신하면 됨.
"""

# load_checkpoint! can restore Adam moments before the GPU cache of a block
# layer exists. Carry them into the cache when it is built, so a resumed run
# continues with the saved optimizer state.
function _restored_adam_moment(cache, key::Symbol, W)
    cache isa Dict || return nothing
    moment = get(cache, key, nothing)
    (moment === nothing || size(moment) != size(W)) && return nothing
    return moment isa CuArray ? moment : CuArray{ComplexF64}(moment)
end

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
        # @inbounds grad[bid, k] = conj(sh[Int32(1)])
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
