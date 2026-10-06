# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl, src/gpu.jl.

"""
    BlockMatrix_Layer(x, n, r; residual=false, activation=Identity())
    BlockMatrix_Layer(x, nin, nout, r; residual=false, activation=Identity())

Block-local complex matrix mixing layer for a vector of fermion fields.

For each coarse block `B`, this applies

```julia
y[o]|_B = sum_i W[B, o, i] * x[i]|_B
```

The default square constructor initializes each block matrix to the identity.
"""
mutable struct BlockMatrix_Layer <: Layer
    W::Array{ComplexF64,3}   # (nblocks, nout, nin)
    r::Int
    residual::Bool
    activation::ActivationFunction
    _result::Vector{WF3x1}
    _block_map::Vector{Int}
    _gpu_block_map::Vector{Int}
    _gpu_w_buf::Any
    _gpu_w_cpu::Matrix{ComplexF64}
    _initialized::Bool
    _grad_W::Array{ComplexF64,3}

    function BlockMatrix_Layer(x::WF3x1, nin::Int, nout::Int, r::Int,
                               residual::Bool=false,
                               activation::ActivationFunction=Identity())
        nblocks = (x.NX ÷ r) * (x.NY ÷ r) * (x.NZ ÷ r) * (x.NT ÷ r)
        W = zeros(ComplexF64, nblocks, nout, nin)
        @inbounds for b in 1:nblocks, i in 1:min(nin, nout)
            W[b, i, i] = 1 + 0im
        end
        new(W, r, residual, activation,
            WF3x1[], Int[], Int[], nothing, Matrix{ComplexF64}(undef, 0, 0), false,
            Array{ComplexF64,3}(undef, 0, 0, 0))
    end
end

function BlockMatrix_Layer(x::WF3x1, n::Int, r::Int,
                           residual::Bool=false,
                           activation::ActivationFunction=Identity())
    return BlockMatrix_Layer(x, n, n, r, residual, activation)
end

function _to_cpu(layer::BlockMatrix_Layer)
    layer._gpu_w_buf = nothing
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._initialized = false
    return layer
end

function _to_gpu(layer::BlockMatrix_Layer, blocks=nothing)
    layer._gpu_w_buf = nothing
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._block_map = Int[]
    layer._initialized = false
    return layer
end

function (layer::BlockMatrix_Layer)(x::AbstractVector{<:WF3x1})
    nin = length(x)
    nblocks, nout, win = size(layer.W)
    win == nin || error("BlockMatrix_Layer expected $win input vectors, got $nin")
    r = layer.r

    if !layer._initialized || length(layer._result) != nout || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(x[1]))
        layer._result = Vector{WF3x1}(undef, nout)
        for o in 1:nout
            layer._result[o] = deepcopy(x[min(o, nin)])
        end
        if x[1] isa WF3x1GPU
            _init_gpu_block_cache!(layer, x[1])
        else
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
        W_gpu, bmap_gpu, _ = _sync_blockmatrix_W_gpu!(layer; upload=false)
        blockmatrix_apply_gpu!(result, x, W_gpu, bmap_gpu, layer.residual)
    else
        NC, NG = x[1].NC, x[1].NG
        NX, NY, NZ, NT = x[1].NX, x[1].NY, x[1].NZ, x[1].NT
        bmap = layer._block_map
        W = layer.W

        @inbounds for o in 1:nout
            ro = result[o]
            site = 0
            for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        s = zero(ComplexF64)
                        for i in 1:nin
                            s += W[bid, o, i] * x[i][ic, ix, iy, iz, it, ig]
                        end
                        if layer.residual && o <= nin
                            s += x[o][ic, ix, iy, iz, it, ig]
                        end
                        ro[ic, ix, iy, iz, it, ig] = s
                    end
                end
            end
        end
    end

    return result
end

# --- BlockMatrix_Layer ---
function gradient(layer::BlockMatrix_Layer, ::Identity, z::Vector{<:WF3x1}, d::Vector{T}, γ::Vector{Spin_matrix}) where T<:WF3x1
    if z[1] isa WF3x1GPU
        return [blockmatrix_gradient_gpu!(layer, z, d)]
    end

    nblocks, nout, nin = size(layer.W)
    length(z) == nin || error("BlockMatrix_Layer gradient expected $nin input vectors, got $(length(z))")
    length(d) == nout || error("BlockMatrix_Layer gradient expected $nout output deltas, got $(length(d))")
    if size(layer._grad_W) != (nblocks, nout, nin)
        layer._grad_W = zeros(ComplexF64, nblocks, nout, nin)
    else
        fill!(layer._grad_W, zero(ComplexF64))
    end
    grad_W = layer._grad_W

    NC, NG = z[1].NC, z[1].NG
    NX, NY, NZ, NT = z[1].NX, z[1].NY, z[1].NZ, z[1].NT
    bmap = layer._block_map

    @inbounds for o in 1:nout
        do_ = d[o]
        for i in 1:nin
            zi = z[i]
            site = 0
            for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                s = zero(ComplexF64)
                for ig in 1:NG
                    @simd for ic in 1:NC
                        s += do_[ic, ix, iy, iz, it, ig] * zi[ic, ix, iy, iz, it, ig]
                    end
                end
                grad_W[bid, o, i] += conj(s)
            end
        end
    end

    return grad_W
end

function DeltaRule(layer::BlockMatrix_Layer, ::Identity,
                   z::Vector{<:WF3x1}, d::Vector{T}) where T<:WF3x1
    nblocks, nout, nin = size(layer.W)
    length(d) == nout || error("BlockMatrix_Layer DeltaRule expected $nout output deltas, got $(length(d))")
    r = layer.r

    if !layer._initialized || length(layer._result) != nin || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(d[1]))
        layer._result = Vector{WF3x1}(undef, nin)
        for i in 1:nin
            layer._result[i] = deepcopy(d[min(i, nout)])
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
        W_gpu, bmap_gpu, _ = _sync_blockmatrix_W_gpu!(layer; upload=false)
        blockmatrix_delta_gpu!(result, d, W_gpu, bmap_gpu, layer.residual)
    else
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        @inbounds for i in 1:nin
            ri = result[i]
            site = 0
            for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        s = zero(ComplexF64)
                        for o in 1:nout
                            s += W[bid, o, i] * d[o][ic, ix, iy, iz, it, ig]
                        end
                        if layer.residual && i <= nout
                            s += d[i][ic, ix, iy, iz, it, ig]
                        end
                        ri[ic, ix, iy, iz, it, ig] = s
                    end
                end
            end
        end
    end
    return result
end

function DeltaRule!(layer::BlockMatrix_Layer, ::Identity,
                    z::Vector{<:WF3x1}, d::Vector{T}, delta::Vector{<:WF3x1}) where T<:WF3x1
    nblocks, nout, nin = size(layer.W)
    length(delta) == nin || error("BlockMatrix_Layer DeltaRule! expected $nin input delta buffers, got $(length(delta))")
    length(d) == nout || error("BlockMatrix_Layer DeltaRule! expected $nout output deltas, got $(length(d))")

    if d[1] isa WF3x1GPU
        W_gpu, bmap_gpu, _ = _sync_blockmatrix_W_gpu!(layer; upload=false)
        blockmatrix_delta_gpu!(delta, d, W_gpu, bmap_gpu, layer.residual)
    else
        NC, NG = d[1].NC, d[1].NG
        NX, NY, NZ, NT = d[1].NX, d[1].NY, d[1].NZ, d[1].NT
        bmap = layer._block_map
        W = layer.W

        @inbounds for i in 1:nin
            ri = delta[i]
            site = 0
            for it in 1:NT, iz in 1:NZ, iy in 1:NY, ix in 1:NX
                site += 1
                bid = bmap[site]
                for ig in 1:NG
                    @simd for ic in 1:NC
                        s = zero(ComplexF64)
                        for o in 1:nout
                            s += W[bid, o, i] * d[o][ic, ix, iy, iz, it, ig]
                        end
                        if layer.residual && i <= nout
                            s += d[i][ic, ix, iy, iz, it, ig]
                        end
                        ri[ic, ix, iy, iz, it, ig] = s
                    end
                end
            end
        end
    end
end

function update!(layer::BlockMatrix_Layer, opt::Adam, grad::Array{ComplexF64,3})
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

        layer.W[idx] -= complex(Δ_re, Δ_im)
    end
end

function update!(layer::BlockMatrix_Layer, opt::Adam, grad)
    if grad isa AbstractVector && !isempty(grad) && grad[1] isa CuArray
        cache = layer._gpu_w_buf
        if !(cache isa Dict) || !haskey(cache, :W)
            error("BlockMatrix_Layer GPU weights are not initialized. Run the GPU forward pass before update!.")
        end
        W_gpu = cache[:W]
        grad_gpu = grad[1]
        if !haskey(cache, :adam_m) || cache[:adam_m] === nothing || size(cache[:adam_m]) != size(W_gpu)
            cache[:adam_m] = CUDA.zeros(ComplexF64, size(W_gpu)...)
            cache[:adam_v] = CUDA.zeros(ComplexF64, size(W_gpu)...)
        end
        blockmatrix_adam_update!(W_gpu, cache[:adam_m], cache[:adam_v], grad_gpu,
                                 opt.α, opt.β1, opt.β2, opt.ε, opt.t)
        return layer
    end
    return update!(layer, opt, grad::Array{ComplexF64,3})
end

function _warm_blockmatrix_adam!(layer::BlockMatrix_Layer, opt::Adam)
    cache = layer._gpu_w_buf
    (!(cache isa Dict) || !haskey(cache, :W)) && return nothing
    W_gpu = cache[:W]
    if !haskey(cache, :adam_m) || cache[:adam_m] === nothing || size(cache[:adam_m]) != size(W_gpu)
        cache[:adam_m] = CUDA.zeros(ComplexF64, size(W_gpu)...)
        cache[:adam_v] = CUDA.zeros(ComplexF64, size(W_gpu)...)
    end
    if !haskey(cache, :grad) || cache[:grad] === nothing || size(cache[:grad]) != size(W_gpu)
        cache[:grad] = CUDA.zeros(ComplexF64, size(W_gpu)...)
    else
        fill!(cache[:grad], 0)
    end
    blockscalar_adam_update!(W_gpu, cache[:adam_m], cache[:adam_v], cache[:grad],
                             opt.α, opt.β1, opt.β2, opt.ε, max(opt.t, 1))
    fill!(cache[:adam_m], 0)
    fill!(cache[:adam_v], 0)
    fill!(cache[:grad], 0)
    return nothing
end

# ---- Wadj 동기화 커널 ----
function _cudakernel_adam_complex_standard!(
    W_gpu::CuDeviceArray{ComplexF64},
    m_gpu::CuDeviceArray{ComplexF64},
    v_gpu::CuDeviceArray{ComplexF64},
    grad_gpu::CuDeviceArray{ComplexF64},
    alpha::Float64, beta1::Float64, beta2::Float64, eps::Float64,
    bias1::Float64, bias2::Float64
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = length(W_gpu)
    if idx > total; return; end

    @inbounds begin
        g = grad_gpu[idx]
        g_re = real(g); g_im = imag(g)

        m = m_gpu[idx]
        mr = beta1 * real(m) + (1 - beta1) * g_re
        mi = beta1 * imag(m) + (1 - beta1) * g_im

        v = v_gpu[idx]
        vr = beta2 * real(v) + (1 - beta2) * g_re * g_re
        vi = beta2 * imag(v) + (1 - beta2) * g_im * g_im

        m_gpu[idx] = complex(mr, mi)
        v_gpu[idx] = complex(vr, vi)

        upd_re = alpha * (mr * bias1) / (sqrt(vr * bias2) + eps)
        upd_im = alpha * (mi * bias1) / (sqrt(vi * bias2) + eps)

        W_gpu[idx] -= complex(upd_re, upd_im)
    end
    return nothing
end
