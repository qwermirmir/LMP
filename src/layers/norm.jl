# NormLayer: scales each field to a fixed norm. It has no trainable weights.

"""
    NormLayer(activation=Identity(), alpha=1.0)

Normalisation layer:

    y[i] = alpha * x[i] / ||x[i]||

Backward pass, for output deltas `d`:

    delta[i] = d[i] / ||z[i]|| - Re(sum_x z[i](x) d[i](x)) * conj(z[i]) / ||z[i]||^3

The backward pass omits the factor `alpha`, so it is exact only for
`alpha = 1`, the value every current model uses.
"""
mutable struct NormLayer <: Layer
    activation::ActivationFunction
    alpha::Float64
    _result::Vector{WF3x1}
    _norm_init::Bool
    _conj_buf::Vector{WF3x1}  # conj(z[i]) for DeltaRule!

    function NormLayer(activation::ActivationFunction= Identity(),alpha::Float64=1.0)
        new(activation,alpha, WF3x1[], false, WF3x1[])
    end

end

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

function (layer::NormLayer)(x::AbstractVector{<:WF3x1})
    n = length(x)

    if !layer._norm_init || length(layer._result) != n || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(x[1]))
        layer._result = [_alloc_result_wf(x[i]) for i in 1:n]
        layer._norm_init = true
    end
    result = layer._result

    if n > 0 && x[1].f isa CuArray
        for i in 1:n
            _normlayer_fwd!((result[i]::WF3x1GPU).f, (x[i]::WF3x1GPU).f, layer.alpha)
        end
    else
        @inbounds for i in 1:n
            axpby!(layer.alpha / sqrt(norm2(x[i])), x[i], false, result[i])
        end
    end

    return result
end

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

function DeltaRule!(layer::NormLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T}, delta::Vector{WF3x1}) where T<:WF3x1

    n = length(z)
    if isempty(layer._conj_buf) || typeof(layer._conj_buf[1]) != typeof(z[1])
        layer._conj_buf = [_alloc_result_wf(z[1])]
    end
    cj_buf = layer._conj_buf[1]

    if n > 0 && z[1].f isa CuArray
        for i in 1:n
            _normlayer_bwd!(
                (delta[i]::WF3x1GPU).f,
                (z[i]::WF3x1GPU).f,
                (d[i]::WF3x1GPU).f)
        end
    else
        @inbounds for i in 1:n
            nrm   = sqrt(norm2(z[i]))
            inv_n = 1.0 / nrm

            conj_fermion!(cj_buf, z[i])
            eta1  = inner_product(cj_buf, d[i]) / 2
            coeff = -2 * real(eta1) * inv_n^3

            axpby!(inv_n, d[i], false, delta[i])
            axpby!(coeff, cj_buf, true,  delta[i])
        end
    end

end

# ---------------------------------------------------------------------------
# No weights: no gradient and no update
# ---------------------------------------------------------------------------

function gradient(layer::NormLayer, ::Identity, z::Vector{WF3x1}, d::Vector{T} ,  γ::Vector{Spin_matrix} )where T<:WF3x1
    return nothing
end

function update!(layer::NormLayer, opt::Adam, grad::Any)
   return
end

# ---------------------------------------------------------------------------
# GPU kernels: one field per launch, reduced in shared memory
# ---------------------------------------------------------------------------

# out = alpha * x / ||x||
function _normlayer_fwd_kernel!(
    out  ::CuDeviceArray{ComplexF64,4},
    x    ::CuDeviceArray{ComplexF64,4},
    alpha::Float64,
    total::Int32
)
    tid   = Int32(threadIdx().x)
    bsz   = Int32(blockDim().x)
    sdata = CUDA.@cuDynamicSharedMem(Float64, Int(bsz))
    s = 0.0
    j = tid
    while j <= total
        @inbounds v = x[j]
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
    scale = alpha / sqrt(sdata[Int32(1)])
    j = tid
    while j <= total
        @inbounds out[j] = scale * x[j]
        j += bsz
    end
    return nothing
end

function _normlayer_fwd!(out::CuArray{ComplexF64,4}, x::CuArray{ComplexF64,4}, alpha::Float64)
    total = Int32(length(x))
    @cuda threads=256 blocks=1 shmem=(256*sizeof(Float64)) _normlayer_fwd_kernel!(
        out, x, alpha, total)
end

# out = d / ||z|| - Re(sum z d) conj(z) / ||z||^3
function _normlayer_bwd_kernel!(
    out  ::CuDeviceArray{ComplexF64,4},
    z    ::CuDeviceArray{ComplexF64,4},
    d    ::CuDeviceArray{ComplexF64,4},
    total::Int32
)
    tid = Int32(threadIdx().x)
    bsz = Int32(blockDim().x)
    sn  = CUDA.@cuDynamicSharedMem(Float64,    Int(bsz))
    sd  = CUDA.@cuDynamicSharedMem(ComplexF64, Int(bsz), Int(bsz) * sizeof(Float64))
    sn_val = 0.0
    sd_val = zero(ComplexF64)
    j = tid
    while j <= total
        @inbounds zj = z[j]
        @inbounds dj = d[j]
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
    inv_n = 1.0 / sqrt(sn[Int32(1)])
    coeff = -real(sd[Int32(1)]) * inv_n^3
    j = tid
    while j <= total
        @inbounds out[j] = inv_n * d[j] + coeff * conj(z[j])
        j += bsz
    end
    return nothing
end

function _normlayer_bwd!(out::CuArray{ComplexF64,4}, z::CuArray{ComplexF64,4}, d::CuArray{ComplexF64,4})
    total = Int32(length(z))
    shmem = 256 * (sizeof(Float64) + sizeof(ComplexF64))
    @cuda threads=256 blocks=1 shmem=shmem _normlayer_bwd_kernel!(
        out, z, d, total)
end
