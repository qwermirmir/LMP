# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Optimizer.jl.

function realProcess(param::T, opt:: Adam , grad ) where {T<:AbstractSpinMatrix}
    key = objectid(param)

    if !haskey(opt.m, key)
        opt.m[key] = Random_SpinM("zero")
        opt.v[key] = Random_SpinM("zero")
    end

    m = opt.m[key]
    v = opt.v[key]

    β1 = opt.β1
    β2 = opt.β2
    α  = opt.α
    ε  = opt.ε
    t  = opt.t
    bias1 = 1 / (1 - β1^t)
    bias2 = 1 / (1 - β2^t)

    sm_g = grad.sm
    sm_m = m.sm
    sm_v = v.sm
    sm_p = param.sm

    @inbounds for j in axes(sm_g, 2), i in axes(sm_g, 1)
        g = real(sm_g[i,j])

        mr = real(sm_m[i,j])
        vr = real(sm_v[i,j])

        mr = β1 * mr + (1 - β1) * g
        vr = β2 * vr + (1 - β2) * g^2

        sm_m[i,j] = complex(mr, 0.0)
        sm_v[i,j] = complex(vr, 0.0)

        upd = α * (mr * bias1) / (sqrt(vr * bias2) + ε)
        sm_p[i,j] -= upd
    end
end

function _warm_gpu_adam!(layer::Union{DenseLinearLayer,PTLayer}, opt::Adam)
    W_gpu = layer._W_gpu
    W_gpu === nothing && return nothing
    NW = size(W_gpu, 3)
    NG = size(W_gpu, 1)
    if layer._adam_m_gpu === nothing || size(layer._adam_m_gpu, 3) != NW
        layer._adam_m_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        layer._adam_v_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
    end
    if layer._grad_gpu === nothing || size(layer._grad_gpu, 3) != NW
        layer._grad_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
    else
        fill!(layer._grad_gpu, 0)
    end
    gpu_adam_update!(W_gpu, layer._adam_m_gpu, layer._adam_v_gpu, layer._grad_gpu,
                     opt.α, opt.β1, opt.β2, opt.ε, max(opt.t, 1))
    fill!(layer._adam_m_gpu, 0)
    fill!(layer._adam_v_gpu, 0)
    fill!(layer._grad_gpu, 0)
    return nothing
end

function _warm_blockscalar_adam!(layer::BlockScalar_Layer, opt::Adam)
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
    blockmatrix_adam_update!(W_gpu, cache[:adam_m], cache[:adam_v], cache[:grad],
                             opt.α, opt.β1, opt.β2, opt.ε, max(opt.t, 1))
    fill!(cache[:adam_m], 0)
    fill!(cache[:adam_v], 0)
    fill!(cache[:grad], 0)
    return nothing
end

function _zero_wf_grad_like(x::WF3x1)
    g = similar(x)
    zero_wf!(g)
    return g
end

function _reset_wf_adam_state!(opt::Adam, xs)
    for x in xs
        key = objectid(x)
        if haskey(opt.wf_state, key)
            st = opt.wf_state[key]
            zero_wf!(st.m)
            zero_wf!(st.v)
        end
    end
    return nothing
end

function _warm_fermionfield_update!(layer::fermionfield_Layer, opt::Adam)
    isempty(layer.x) && return nothing
    grad0 = [_zero_wf_grad_like(x) for x in layer.x]
    update!(layer, opt, grad0)
    _reset_wf_adam_state!(opt, layer.x)
    return nothing
end

function _warm_gpu_update!(layer::Union{DenseLinearLayer,PTLayer}, opt::Adam)
    W_gpu = layer._W_gpu
    W_gpu === nothing && return nothing
    NW = size(W_gpu, 3)
    NG = size(W_gpu, 1)
    if layer._grad_gpu === nothing || size(layer._grad_gpu, 3) != NW
        layer._grad_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
    else
        fill!(layer._grad_gpu, 0)
    end
    update!(layer, opt, [layer._grad_gpu])
    layer._adam_m_gpu !== nothing && fill!(layer._adam_m_gpu, 0)
    layer._adam_v_gpu !== nothing && fill!(layer._adam_v_gpu, 0)
    fill!(layer._grad_gpu, 0)
    return nothing
end

function _warm_blockscalar_update!(layer::BlockScalar_Layer, opt::Adam)
    cache = layer._gpu_w_buf
    (!(cache isa Dict) || !haskey(cache, :W)) && return nothing
    W_gpu = cache[:W]
    if !haskey(cache, :grad) || cache[:grad] === nothing || size(cache[:grad]) != size(W_gpu)
        cache[:grad] = CUDA.zeros(ComplexF64, size(W_gpu)...)
    else
        fill!(cache[:grad], 0)
    end
    update!(layer, opt, [cache[:grad]])
    haskey(cache, :adam_m) && cache[:adam_m] !== nothing && fill!(cache[:adam_m], 0)
    haskey(cache, :adam_v) && cache[:adam_v] !== nothing && fill!(cache[:adam_v], 0)
    fill!(cache[:grad], 0)
    return nothing
end

function prealloc_optimizer_state!(layers, opt::Adam)
    t0 = opt.t
    advance_step!(opt)
    for layer in layers
        if layer isa fermionfield_Layer
            _warm_fermionfield_update!(layer, opt)
        elseif layer isa DenseLinearLayer && layer._W_gpu !== nothing
            _warm_gpu_adam!(layer, opt)
            _warm_gpu_update!(layer, opt)
        elseif layer isa PTLayer && layer._W_gpu !== nothing
            _warm_gpu_adam!(layer, opt)
            _warm_gpu_update!(layer, opt)
        elseif layer isa BlockScalar_Layer && layer._gpu_w_buf isa Dict && haskey(layer._gpu_w_buf, :W)
            _warm_blockscalar_adam!(layer, opt)
            _warm_blockscalar_update!(layer, opt)
        elseif layer isa BlockMatrix_Layer && layer._gpu_w_buf isa Dict && haskey(layer._gpu_w_buf, :W)
            _warm_blockmatrix_adam!(layer, opt)
        end
    end
    opt.t = t0
    CUDA.synchronize()
    return layers
end
