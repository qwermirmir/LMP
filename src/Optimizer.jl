abstract type Opt end





mutable struct AdamState{T}
    m::T
    v::T
end






mutable struct Adam <: Opt
    α::Float64
    β1::Float64
    β2::Float64
    ε::Float64
    m::Dict{UInt, Spin_matrix}
    v::Dict{UInt, Spin_matrix}
    t::Int
    scalar_state::Dict{UInt, AdamState{Float64}}   # 스칼라용
    wf_state::Dict{UInt, AdamState{WF3x1}}         # WF3x1용

    function Adam(α=0.001, β1=0.9, β2=0.999, ε=1e-8)
        new(α, β1, β2, ε,
            Dict{UInt, Spin_matrix}(),
            Dict{UInt, Spin_matrix}(),
            0,
            Dict{UInt, AdamState{Float64}}(),
            Dict{UInt, AdamState{WF3x1}}())
    end
end

"""Advance Adam's global timestep once for one optimizer sweep."""
function advance_step!(opt::Adam)
    opt.t += 1
    return opt.t
end






function complexProcess(param::T, opt:: Adam , grad ) where {T<:AbstractSpinMatrix}


    
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
        g_re = real(sm_g[i,j])
        g_im = imag(sm_g[i,j])

        mr = real(sm_m[i,j])
        mi = imag(sm_m[i,j])
        vr = real(sm_v[i,j])
        vi = imag(sm_v[i,j])

        # EMA update
        mr = β1 * mr + (1 - β1) * g_re
        vr = β2 * vr + (1 - β2) * g_re^2
        mi = β1 * mi + (1 - β1) * g_im
        vi = β2 * vi + (1 - β2) * g_im^2

        sm_m[i,j] = complex(mr, mi)
        sm_v[i,j] = complex(vr, vi)

        # bias-corrected update
        upd_re = α * (mr * bias1) / (sqrt(vr * bias2) + ε)
        upd_im = α * (mi * bias1) / (sqrt(vi * bias2) + ε)

        sm_p[i,j] -= complex(upd_re, -upd_im)
    end

end   




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








function update!(layer::CPTC_Layer, opt::Adam,  grad)
    param = layer.W

    if length(param)==1
        realProcess(param[1], opt , grad[1])
    else
        for i in 1:length(param)
            realProcess(param[i], opt , grad[i])
        end
    end


end


function update!(layer::PTC_Layer, opt::Adam,  grad)
    param = layer.W

    # if length(param) ==1
    #     complexProcess(param[1], opt , grad[1])
    
    # else 
    #     for i in 1:length(param)
    #         complexProcess(param[i], opt , grad[i] )
    #     end
    # end
    for i in 1:length(param)
        complexProcess(param[i], opt , grad[i] )
    end

    
end


function update!(layer::Attention_Layer, opt::Adam, grad )
    if layer.Q isa PTC_Layer
 

        param = layer.Q.W
        if length(param) ==1
            complexProcess(param, opt , grad[1] )
        
        else 
            for i in 1:length(param)
                complexProcess(param[i], opt , grad[1][i] )
            end
        end

        param = layer.K.W
        if length(param) ==1
            complexProcess(param, opt , grad[2] )
        
        else 
            for i in 1:length(param)
                complexProcess(param[i], opt , grad[2][i] )
            end
        end

        param = layer.V.W
        if length(param) ==1
            complexProcess(param, opt , grad[3] )
        
        else 
            for i in 1:length(param)
                complexProcess(param[i], opt , grad[3][i] )
            end
        end
    end


    if layer.Q isa CPTC_Layer
        param = layer.Q.W
        if length(param) ==1
            realProcess(param, opt , grad[1])  
        else 
            for i in 1:length(param)
                realProcess(param[i], opt , grad[1][i])
            end
        end
    
        param = layer.K.W
        if length(param) ==1
            realProcess(param, opt , grad[2])  
        else 
            for i in 1:length(param)
                realProcess(param[i], opt , grad[2][i])
            end
        end
    
        param = layer.V.W
        if length(param) ==1
            realProcess(param, opt , grad[3])  
        else 
            for i in 1:length(param)
                realProcess(param[i], opt , grad[3][i])
            end
        end
    end
    


end










function update!(layer::test_Layer, opt::Adam, grad )
    param = layer.K.W
    if length(param) ==1
        complexProcess(param, opt , grad[1] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[1][i] )
        end
    end


    param = layer.V.W
    if length(param) ==1
        realProcess(param, opt , grad[2])  
    else 
        for i in 1:length(param)
            realProcess(param[i], opt , grad[2][i])
        end
    end

    


end




function update!(layer::FullPTCAttention_Layer, opt::Adam, grad )
    param = layer.Q.W
    if length(param) ==1
        complexProcess(param, opt , grad[1] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[1][i] )
        end
    end


    param = layer.K.W
    if length(param) ==1
        complexProcess(param, opt , grad[2] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[2][i] )
        end
    end


    param = layer.V.W
    if length(param) ==1
        complexProcess(param, opt , grad[3])  
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[3][i])
        end
    end

    


end







function update!(layer::PrallelTransportAttention_Layer, opt::Adam, grad )
    param = layer.K.W
    if length(param) ==1
        complexProcess(param, opt , grad[1] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[1][i] )
        end
    end


    param = layer.V.W
    if length(param) ==1
        complexProcess(param, opt , grad[2])  
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[2][i])
        end
    end

end 

function update!(layer::siteweight_Layer, opt::Adam, grad)
    param = layer.Q.W
    if length(param) ==1
        complexProcess(param, opt , grad[1] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[1][i] )
        end
    end


    param = layer.K.W
    if length(param) ==1
        complexProcess(param, opt , grad[2] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[2][i] )
        end
    end


    param = layer.V.W
    if length(param) ==1
        complexProcess(param, opt , grad[3])  
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[3][i])
        end
    end

end

function update!(layer::DenseLinearLayer, opt::Adam, grad)
    # GPU batched path: grad[1]이 CuArray이면 GPU Adam
    if grad[1] isa CuArray
        NW = length(layer.W)
        NG = 4
        if layer._adam_m_gpu === nothing || size(layer._adam_m_gpu, 3) != NW
            layer._adam_m_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._adam_v_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        gpu_adam_update!(layer._W_gpu, layer._adam_m_gpu, layer._adam_v_gpu, grad[1],
                         opt.α, opt.β1, opt.β2, opt.ε, opt.t)
    else
        param = layer.W
        if length(param) == 1
            complexProcess(param, opt, grad[1])
        else
            for i in 1:length(param)
                complexProcess(param[i], opt, grad[i])
            end
        end
    end
end

function update!(layer::PTLayer, opt::Adam, grad)
    # GPU batched path: grad[1]이 CuArray이면 GPU Adam
    if grad[1] isa CuArray
        NW = length(layer.W)
        NG = 4
        # lazy init Adam states on GPU
        if layer._adam_m_gpu === nothing || size(layer._adam_m_gpu, 3) != NW
            layer._adam_m_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._adam_v_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
        end
        gpu_adam_update!(layer._W_gpu, layer._adam_m_gpu, layer._adam_v_gpu, grad[1],
                         opt.α, opt.β1, opt.β2, opt.ε, opt.t)
    else
        # CPU fallback
        param = layer.W
        for i in 1:length(param)
            complexProcess(param[i], opt, grad[1][i])
        end
    end
end




# NormLayer용 update! 추가
function update!(layer::NormLayer, opt::Adam, grad::Float64)
    key = objectid(layer)
    state = get!(opt.scalar_state, key) do
        AdamState{Float64}(0.0, 0.0)
    end

    state.m = opt.β1 * state.m + (1 - opt.β1) * grad
    state.v = opt.β2 * state.v + (1 - opt.β2) * grad^2

    m_hat = state.m / (1 - opt.β1^opt.t)
    v_hat = state.v / (1 - opt.β2^opt.t)
    layer.W -= opt.α * m_hat / (sqrt(v_hat) + opt.ε)
end


function update!(layer::Dirac_Layer, opt::Adam, grad)
    # Dirac_Layer에는 업데이트가 필요 없으므로 빈 함수로 둡니다.

   return
end 

function update!(layer::NormLayer, opt::Adam, grad::Any)

   return
end 

function update!(layer::PtcRayleighQuotient_Layer,opt::Adam, grad::Any)

  
end 

# function update!(layer::L2Norm_Layer, opt::Adam, grad::Any)

  
# end


function update!(layer::fermionfield_Layer, opt::Adam, grad::Any)
    param = layer.x

    if length(param) ==1
        wf_adam_step!(param[1], opt , grad[1] )
    
    else 
        for i in 1:length(param)
            wf_adam_step!(param[i], opt , grad[i] )
        end
    end
    # wf_adam_step!(layer.x, opt, grad[1][1])

    return nothing
end





# RayleighQuotient_Layer: x :: WF3x1
function update!(layer::RayleighQuotient_Layer, opt::Adam, grad::WF3x1)
    wf_adam_step!(layer.x, opt, grad)
    return nothing
end

# HoppingRayleighQuotient_Layer: x :: WF3x1
function update!(layer::HoppingRayleighQuotient_Layer, opt::Adam, grad::WF3x1)
    wf_adam_step!(layer.x, opt, grad)
    return nothing
end



# WF3x1 한 개(param, grad)에 대해 Adam 스텝 수행
function wf_adam_step!(param::WF3x1, opt::Adam, grad::WF3x1)
    key = objectid(param)

    # m, v 상태 가져오기 (없으면 similar(param)으로 초기화)
    state = get!(opt.wf_state, key) do
        m0 = similar(param)
        v0 = similar(param)
        zero_wf!(m0)
        zero_wf!(v0)
        AdamState{WF3x1}(m0, v0)
    end
    m, v = state.m, state.v

    β1, β2, α, ε = opt.β1, opt.β2, opt.α, opt.ε
    t  = max(opt.t, 1)
    b1 = 1 - β1^t
    b2 = 1 - β2^t

    NC, NX, NY, NZ, NT, NG = param.NC, param.NX, param.NY, param.NZ, param.NT, param.NG

    @inbounds for ic = 1:NC, it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX, ig = 1:NG
        g  = grad[ic, ix, iy, iz, it, ig]
        mi = m[ic, ix, iy, iz, it, ig]
        vi = v[ic, ix, iy, iz, it, ig]

        g_re, g_im = real(g), imag(g)
        m_re, m_im = real(mi), imag(mi)
        v_re, v_im = real(vi), imag(vi)

        # 1차/2차 모멘트 업데이트
        m_re = β1*m_re + (1-β1)*g_re
        m_im = β1*m_im + (1-β1)*g_im
        v_re = β2*v_re + (1-β2)*(g_re*g_re)
        v_im = β2*v_im + (1-β2)*(g_im*g_im)

        m[ic, ix, iy, iz, it, ig] = complex(m_re, m_im)
        v[ic, ix, iy, iz, it, ig] = complex(v_re, v_im)

        # bias correction
        m_hat_re = m_re / b1
        m_hat_im = m_im / b1
        v_hat_re = v_re / b2
        v_hat_im = v_im / b2

        # 실제 업데이트 양
        upd_re = α * m_hat_re / (sqrt(v_hat_re) + ε)
        upd_im = α * m_hat_im / (sqrt(v_hat_im) + ε)

        # param := param - (upd_re - i*upd_im)
        param[ic, ix, iy, iz, it, ig] -= (upd_re - im*upd_im)
    end

    return param
end



function kernel_wf_adam!(p, m, v, g, β1, β2, α, ε, b1, b2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(p)
        gi = g[i]
        mi = m[i]
        vi = v[i]

        g_re = real(gi); g_im = imag(gi)
        m_re = real(mi); m_im = imag(mi)
        v_re = real(vi); v_im = imag(vi)

        m_re = β1*m_re + (1-β1)*g_re
        m_im = β1*m_im + (1-β1)*g_im
        v_re = β2*v_re + (1-β2)*(g_re*g_re)
        v_im = β2*v_im + (1-β2)*(g_im*g_im)

        m[i] = ComplexF64(m_re, m_im)
        v[i] = ComplexF64(v_re, v_im)

        m_hat_re = m_re / b1
        m_hat_im = m_im / b1
        v_hat_re = v_re / b2
        v_hat_im = v_im / b2

        # 수치 오차 안전장치(원래는 음수면 안 나와야 정상)
        v_hat_re = ifelse(v_hat_re < 0.0, 0.0, v_hat_re)
        v_hat_im = ifelse(v_hat_im < 0.0, 0.0, v_hat_im)

        upd_re = α * m_hat_re / (sqrt(v_hat_re) + ε)
        upd_im = α * m_hat_im / (sqrt(v_hat_im) + ε)

        # param -= (upd_re - i*upd_im) = Complex(upd_re, -upd_im)
        p[i] = p[i] - ComplexF64(upd_re, -upd_im)
    end
    return
end

function wf_adam_step!(param::WF3x1GPU, opt::Adam, grad::WF3x1GPU)
    key = objectid(param)

    state = get!(opt.wf_state, key) do
        # similar(param)이 0으로 채운다 해도, 확실히 0으로 만드는 게 안전함
        m0 = similar(param)
        v0 = similar(param)
        zero_wf!(m0)
        zero_wf!(v0)
        AdamState{WF3x1}(m0, v0)
    end

    m = state.m::WF3x1GPU
    v = state.v::WF3x1GPU

    p = param.f
    g = grad.f
    mf = m.f
    vf = v.f

    t  = max(opt.t, 1)
    b1 = 1 - opt.β1^t
    b2 = 1 - opt.β2^t

    threads = 256
    blocks  = cld(length(p), threads)
    @cuda threads=threads blocks=blocks kernel_wf_adam!(p, mf, vf, g, opt.β1, opt.β2, opt.α, opt.ε, b1, b2)

    return param
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


"""
    sync_layers!(layers)

저장/substitute_Layer! 전에 호출. _W_gpu → W (CPU) 동기화.
"""
function sync_layers_to_gpu!(layers)
    for layer in layers
        if layer isa PTLayer && !isempty(layer.W) && layer.W[1] isa Spin_matrix_GPU
            NW = length(layer.W)
            NG = 4
            if layer._W_gpu === nothing || size(layer._W_gpu, 3) != NW
                layer._W_gpu     = CUDA.zeros(ComplexF64, NG, NG, NW)
                layer._W_cpu_buf = Array{ComplexF64}(undef, NG, NG, NW)
            end
            if layer._Wadj_gpu === nothing || size(layer._Wadj_gpu, 3) != NW
                layer._Wadj_gpu = CUDA.zeros(ComplexF64, NG, NG, NW)
            end
            sync_W_to_gpu!(layer._W_gpu, layer.W)
        end
    end
    return layers
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

function sync_layers!(layers)
    for layer in layers
        if layer isa PTLayer && layer._W_gpu !== nothing
            if layer._W_cpu_buf !== nothing
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            else
                layer._W_cpu_buf = Array{ComplexF64}(undef, size(layer._W_gpu)...)
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            end
        elseif layer isa DenseLinearLayer && layer._W_gpu !== nothing
            if layer._W_cpu_buf !== nothing
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            else
                layer._W_cpu_buf = Array{ComplexF64}(undef, size(layer._W_gpu)...)
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            end
        elseif layer isa BlockScalar_Layer && layer._gpu_w_buf isa Dict && haskey(layer._gpu_w_buf, :W)
            copyto!(layer.W, layer._gpu_w_buf[:W])
        elseif layer isa BlockMatrix_Layer && layer._gpu_w_buf isa Dict && haskey(layer._gpu_w_buf, :W)
            copyto!(layer.W, layer._gpu_w_buf[:W])
        end
    end
end

"""
    save_checkpoint(path, layers, opt)

Saves W + Adam GPU state + opt.t without calling substitute_Layer!.
Layers remain on GPU throughout — no memory pool growth.
"""
function save_checkpoint(path::AbstractString, layers, opt::Adam)
    sync_layers!(layers)  # _W_gpu → _W_cpu_buf + layer.W (for BlockScalar)
    data = Dict{String,Any}("opt_t" => opt.t)

    for (i, layer) in enumerate(layers)
        if (layer isa PTLayer || layer isa DenseLinearLayer) && layer._W_gpu !== nothing
            # _W_cpu_buf is the (NG,NG,NW) staging buffer populated by sync_layers!
            data["layer_$(i)_W"] = layer._W_cpu_buf !== nothing ?
                                    copy(layer._W_cpu_buf) : Array(layer._W_gpu)
            if layer._adam_m_gpu !== nothing
                data["layer_$(i)_adam_m"] = Array(layer._adam_m_gpu)
                data["layer_$(i)_adam_v"] = Array(layer._adam_v_gpu)
            end
        elseif layer isa BlockScalar_Layer
            # layer.W already synced from GPU by sync_layers!
            data["layer_$(i)_W"] = copy(layer.W)
            cache = layer._gpu_w_buf
            if cache isa Dict && haskey(cache, :adam_m) && cache[:adam_m] !== nothing
                data["layer_$(i)_adam_m"] = Array(cache[:adam_m])
                data["layer_$(i)_adam_v"] = Array(cache[:adam_v])
            end
        elseif layer isa BlockMatrix_Layer
            data["layer_$(i)_W"] = copy(layer.W)
            cache = layer._gpu_w_buf
            if cache isa Dict && haskey(cache, :adam_m) && cache[:adam_m] !== nothing
                data["layer_$(i)_adam_m"] = Array(cache[:adam_m])
                data["layer_$(i)_adam_v"] = Array(cache[:adam_v])
            end
        elseif layer isa fermionfield_Layer
            for j in eachindex(layer.x)
                x = layer.x[j]
                data["layer_$(i)_fermion_x_$(j)"] = Array(x.f)
                if x isa WF3x1GPU
                    data["layer_$(i)_fermion_blocks_$(j)"] = Tuple(Int.(x.blockinfo.blocks))
                end
                state_key = objectid(x)
                if haskey(opt.wf_state, state_key)
                    state = opt.wf_state[state_key]
                    data["layer_$(i)_fermion_m_$(j)"] = Array(state.m.f)
                    data["layer_$(i)_fermion_v_$(j)"] = Array(state.v.f)
                end
            end
        end
    end

    JLD2.save(path, data)
    return path
end

_blocks_tuple(x) = (Int(x[1]), Int(x[2]), Int(x[3]), Int(x[4]))

function _checkpoint_source_blocks(data, key::String)
    meta_key = replace(key, "_x_" => "_blocks_")
    haskey(data, meta_key) || return nothing
    return _blocks_tuple(data[meta_key])
end

@inline _ckpt_cvt(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)

function _checkpoint_layout_copy_kernel!(
    dest, src, total::Int,
    sb1::Int, sb2::Int, sb3::Int, sb4::Int,
    db1::Int, db2::Int, db3::Int, db4::Int,
    NX::Int, NY::Int, NZ::Int, NT::Int,
    sb_s1::Int, sb_s2::Int, sb_s3::Int, sb_s4::Int,
    sr_s1::Int, sr_s2::Int, sr_s3::Int, sr_s4::Int,
    db_s1::Int, db_s2::Int, db_s3::Int, db_s4::Int,
    dr_s1::Int, dr_s2::Int, dr_s3::Int, dr_s4::Int,
    drn1::Int, drn2::Int, drn3::Int, drn4::Int,
)
    idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if idx <= total
        t = idx - 1
        c = (t % 3) + 1
        t = div(t, 3)
        a = (t % 4) + 1
        t = div(t, 4)
        bd = (t % (db1 * db2 * db3 * db4)) + 1
        rd = div(t, db1 * db2 * db3 * db4) + 1

        ix0 = _ckpt_cvt(bd, db1, db_s1) + _ckpt_cvt(rd, drn1, dr_s1) * db1
        iy0 = _ckpt_cvt(bd, db2, db_s2) + _ckpt_cvt(rd, drn2, dr_s2) * db2
        iz0 = _ckpt_cvt(bd, db3, db_s3) + _ckpt_cvt(rd, drn3, dr_s3) * db3
        it0 = _ckpt_cvt(bd, db4, db_s4) + _ckpt_cvt(rd, drn4, dr_s4) * db4

        bs = 1 +
             (ix0 % sb1) * sb_s1 +
             (iy0 % sb2) * sb_s2 +
             (iz0 % sb3) * sb_s3 +
             (it0 % sb4) * sb_s4
        rs = 1 +
             div(ix0, sb1) * sr_s1 +
             div(iy0, sb2) * sr_s2 +
             div(iz0, sb3) * sr_s3 +
             div(it0, sb4) * sr_s4

        @inbounds dest[c, a, bd, rd] = src[c, a, bs, rs]
    end
    return nothing
end

function _checkpoint_linear_canonical_copy_kernel!(
    dest, src, total::Int,
    db1::Int, db2::Int, db3::Int, db4::Int,
    NX::Int, NY::Int, NZ::Int, NT::Int,
    db_s1::Int, db_s2::Int, db_s3::Int, db_s4::Int,
    dr_s1::Int, dr_s2::Int, dr_s3::Int, dr_s4::Int,
    drn1::Int, drn2::Int, drn3::Int, drn4::Int,
)
    idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if idx <= total
        t = idx - 1
        c = (t % 3) + 1
        t = div(t, 3)
        a = (t % 4) + 1
        t = div(t, 4)
        bd = (t % (db1 * db2 * db3 * db4)) + 1
        rd = div(t, db1 * db2 * db3 * db4) + 1

        ix0 = _ckpt_cvt(bd, db1, db_s1) + _ckpt_cvt(rd, drn1, dr_s1) * db1
        iy0 = _ckpt_cvt(bd, db2, db_s2) + _ckpt_cvt(rd, drn2, dr_s2) * db2
        iz0 = _ckpt_cvt(bd, db3, db_s3) + _ckpt_cvt(rd, drn3, dr_s3) * db3
        it0 = _ckpt_cvt(bd, db4, db_s4) + _ckpt_cvt(rd, drn4, dr_s4) * db4
        site = 1 + ix0 + iy0 * NX + iz0 * NX * NY + it0 * NX * NY * NZ
        src_idx = c + 3 * (a - 1) + 12 * (site - 1)

        @inbounds dest[c, a, bd, rd] = src[src_idx]
    end
    return nothing
end

function _copy_checkpoint_fermion_linear_canonical!(x::WF3x1GPU, arr)
    length(arr) == length(x.f) || error("Checkpoint fermion length mismatch: checkpoint has $(length(arr)), field has $(length(x.f)).")
    src = CuArray{ComplexF64}(reshape(arr, :))
    dst_blocks = _blocks_tuple(x.blockinfo.blocks)
    db1, db2, db3, db4 = dst_blocks
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    drn1, drn2, drn3, drn4 = NX ÷ db1, NY ÷ db2, NZ ÷ db3, NT ÷ db4
    total = length(x.f)
    threads = 256
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _checkpoint_linear_canonical_copy_kernel!(
        x.f, src, total,
        db1, db2, db3, db4,
        NX, NY, NZ, NT,
        1, db1, db1 * db2, db1 * db2 * db3,
        1, drn1, drn1 * drn2, drn1 * drn2 * drn3,
        drn1, drn2, drn3, drn4,
    )
    return x
end

function _copy_checkpoint_fermion_array!(x::WF3x1GPU, arr, src_blocks::NTuple{4,Int})
    dst_blocks = _blocks_tuple(x.blockinfo.blocks)
    if src_blocks == dst_blocks && size(arr) == size(x.f)
        copyto!(x.f, arr)
        return x
    end

    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    src_blocksize = prod(src_blocks)
    src_rsize = (NX * NY * NZ * NT) ÷ src_blocksize
    if size(arr, 3) != src_blocksize || size(arr, 4) != src_rsize
        error("Checkpoint fermion layout mismatch: source blocks=$src_blocks imply f size " *
              "(3,4,$src_blocksize,$src_rsize), but checkpoint has $(size(arr)).")
    end

    sb1, sb2, sb3, sb4 = src_blocks
    db1, db2, db3, db4 = dst_blocks
    srn1, srn2, srn3, srn4 = NX ÷ sb1, NY ÷ sb2, NZ ÷ sb3, NT ÷ sb4
    drn1, drn2, drn3, drn4 = NX ÷ db1, NY ÷ db2, NZ ÷ db3, NT ÷ db4
    src = CuArray{ComplexF64,4}(arr)
    total = length(x.f)
    threads = 256
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _checkpoint_layout_copy_kernel!(
        x.f, src, total,
        sb1, sb2, sb3, sb4,
        db1, db2, db3, db4,
        NX, NY, NZ, NT,
        1, sb1, sb1 * sb2, sb1 * sb2 * sb3,
        1, srn1, srn1 * srn2, srn1 * srn2 * srn3,
        1, db1, db1 * db2, db1 * db2 * db3,
        1, drn1, drn1 * drn2, drn1 * drn2 * drn3,
        drn1, drn2, drn3, drn4,
    )
    return x
end

function _copy_checkpoint_fermion_array!(x::WF3x1CPU, arr, src_blocks)
    size(arr) == size(x.f) || error("Cannot layout-convert fermion checkpoint into CPU field without CPU canonical conversion.")
    copyto!(x.f, arr)
    return x
end

function _load_checkpoint_fermion_field!(x::WF3x1GPU, data, key::String)
    arr = data[key]
    src_blocks = _checkpoint_source_blocks(data, key)
    if src_blocks === nothing
        return _copy_checkpoint_fermion_linear_canonical!(x, arr)
    end
    return _copy_checkpoint_fermion_array!(x, arr, src_blocks)
end

function _load_checkpoint_fermion_field!(x::WF3x1CPU, data, key::String)
    return _copy_checkpoint_fermion_array!(x, data[key], nothing)
end

function _load_checkpoint_fermion_adam_field!(x::WF3x1GPU, arr, data, key_x::String)
    src_blocks = _checkpoint_source_blocks(data, key_x)
    if src_blocks === nothing
        return _copy_checkpoint_fermion_linear_canonical!(x, arr)
    end
    return _copy_checkpoint_fermion_array!(x, arr, src_blocks)
end

function _load_checkpoint_fermion_adam_field!(x::WF3x1CPU, arr, data, key_x::String)
    return _copy_checkpoint_fermion_array!(x, arr, nothing)
end

function _lattice_dims_from_layers(layers)
    for layer in layers
        if layer isa fermionfield_Layer && !isempty(layer.x)
            x = layer.x[1]
            return (Int(x.NX), Int(x.NY), Int(x.NZ), Int(x.NT))
        end
    end
    return nothing
end

function _infer_block_r(nblocks::Int, dims::NTuple{4,Int})
    volume = prod(dims)
    volume % nblocks == 0 || return nothing
    block_volume = volume ÷ nblocks
    r = round(Int, block_volume^(1 / 4))
    return r^4 == block_volume ? r : nothing
end

@inline function _block_id(bx::Int, by::Int, bz::Int, bt::Int, NBX::Int, NBY::Int, NBZ::Int)
    return bx + NBX * ((by - 1) + NBY * ((bz - 1) + NBZ * (bt - 1)))
end

function _convert_blockscalar_weights(arr::AbstractMatrix{ComplexF64},
                                      dims::NTuple{4,Int},
                                      src_r::Int,
                                      dst_r::Int)
    src_r == dst_r && return copy(arr)
    NX, NY, NZ, NT = dims
    src_NBX, src_NBY, src_NBZ, src_NBT = NX ÷ src_r, NY ÷ src_r, NZ ÷ src_r, NT ÷ src_r
    dst_NBX, dst_NBY, dst_NBZ, dst_NBT = NX ÷ dst_r, NY ÷ dst_r, NZ ÷ dst_r, NT ÷ dst_r
    src_nblocks = src_NBX * src_NBY * src_NBZ * src_NBT
    dst_nblocks = dst_NBX * dst_NBY * dst_NBZ * dst_NBT
    size(arr, 1) == src_nblocks || error("source block count does not match inferred source r")

    out = zeros(ComplexF64, dst_nblocks, size(arr, 2))
    counts = zeros(Int, dst_nblocks)

    if src_r < dst_r
        dst_r % src_r == 0 || error("source r=$src_r does not divide destination r=$dst_r")
        ratio = dst_r ÷ src_r
        @inbounds for sbt in 1:src_NBT, sbz in 1:src_NBZ, sby in 1:src_NBY, sbx in 1:src_NBX
            dbx = (sbx - 1) ÷ ratio + 1
            dby = (sby - 1) ÷ ratio + 1
            dbz = (sbz - 1) ÷ ratio + 1
            dbt = (sbt - 1) ÷ ratio + 1
            src_bid = _block_id(sbx, sby, sbz, sbt, src_NBX, src_NBY, src_NBZ)
            dst_bid = _block_id(dbx, dby, dbz, dbt, dst_NBX, dst_NBY, dst_NBZ)
            out[dst_bid, :] .+= arr[src_bid, :]
            counts[dst_bid] += 1
        end
        @inbounds for bid in eachindex(counts)
            out[bid, :] ./= counts[bid]
        end
    else
        src_r % dst_r == 0 || error("destination r=$dst_r does not divide source r=$src_r")
        ratio = src_r ÷ dst_r
        @inbounds for dbt in 1:dst_NBT, dbz in 1:dst_NBZ, dby in 1:dst_NBY, dbx in 1:dst_NBX
            sbx = (dbx - 1) ÷ ratio + 1
            sby = (dby - 1) ÷ ratio + 1
            sbz = (dbz - 1) ÷ ratio + 1
            sbt = (dbt - 1) ÷ ratio + 1
            src_bid = _block_id(sbx, sby, sbz, sbt, src_NBX, src_NBY, src_NBZ)
            dst_bid = _block_id(dbx, dby, dbz, dbt, dst_NBX, dst_NBY, dst_NBZ)
            out[dst_bid, :] .= arr[src_bid, :]
        end
    end

    return out
end

function _maybe_convert_blockscalar_checkpoint(arr::AbstractMatrix{ComplexF64},
                                               layer::BlockScalar_Layer,
                                               dims)
    size(arr) == size(layer.W) && return copy(arr)
    dims === nothing && return nothing
    src_r = _infer_block_r(size(arr, 1), dims)
    src_r === nothing && return nothing
    try
        return _convert_blockscalar_weights(arr, dims, src_r, layer.r)
    catch
        return nothing
    end
end

"""
    load_checkpoint!(path, layers, opt)

Restores W + Adam state into live GPU layers from a W-only checkpoint.
No substitute_Layer! needed — layers stay on GPU.
Falls back silently if a layer's key is absent (partial checkpoint).
"""
function load_checkpoint!(path::AbstractString, layers, opt::Adam)
    data = JLD2.load(path)
    opt.t = Int(get(data, "opt_t", 0))
    lattice_dims = _lattice_dims_from_layers(layers)

    for (i, layer) in enumerate(layers)
        if layer isa PTLayer || layer isa DenseLinearLayer
            key_W = "layer_$(i)_W"
            if haskey(data, key_W)
                w_arr = data[key_W]  # Array{ComplexF64,3} (NG,NG,NW)
                expected = layer._W_gpu === nothing ? size(w_arr) : size(layer._W_gpu)
                if expected != size(w_arr)
                    @warn "Skipping checkpoint weight with incompatible shape" layer=i key=key_W expected got=size(w_arr)
                    continue
                end
                if layer._W_gpu === nothing
                    layer._W_gpu = CuArray{ComplexF64,3}(w_arr)
                else
                    copyto!(layer._W_gpu, w_arr)
                end
                if layer._W_cpu_buf === nothing
                    layer._W_cpu_buf = copy(w_arr)
                else
                    copyto!(layer._W_cpu_buf, w_arr)
                end
                sync_W_from_gpu!(layer.W, layer._W_gpu, layer._W_cpu_buf)
            end
            key_m = "layer_$(i)_adam_m"
            if haskey(data, key_m) && size(data[key_m]) == size(layer._W_gpu)
                layer._adam_m_gpu = CuArray{ComplexF64,3}(data[key_m])
                layer._adam_v_gpu = CuArray{ComplexF64,3}(data["layer_$(i)_adam_v"])
            end
        elseif layer isa BlockScalar_Layer
            key_W = "layer_$(i)_W"
            if haskey(data, key_W)
                w_arr = _maybe_convert_blockscalar_checkpoint(data[key_W], layer, lattice_dims)
                if w_arr === nothing
                    @warn "Skipping checkpoint BlockScalar weight with incompatible shape" layer=i key=key_W expected=size(layer.W) got=size(data[key_W])
                    continue
                end
                copyto!(layer.W, w_arr)
                cache = layer._gpu_w_buf
                if cache isa Dict && haskey(cache, :W) && cache[:W] !== nothing
                    copyto!(cache[:W], layer.W)
                end
            end
            key_m = "layer_$(i)_adam_m"
            if haskey(data, key_m)
                m_arr = _maybe_convert_blockscalar_checkpoint(data[key_m], layer, lattice_dims)
                v_arr = _maybe_convert_blockscalar_checkpoint(data["layer_$(i)_adam_v"], layer, lattice_dims)
                cache = layer._gpu_w_buf
                if cache isa Dict && haskey(cache, :W) && cache[:W] !== nothing
                    m_arr !== nothing && (cache[:adam_m] = CuArray{ComplexF64}(m_arr))
                    v_arr !== nothing && (cache[:adam_v] = CuArray{ComplexF64}(v_arr))
                elseif m_arr !== nothing && v_arr !== nothing
                    layer._gpu_w_buf = Dict{Symbol,Any}(
                        :adam_m => m_arr,
                        :adam_v => v_arr,
                    )
                end
            end
        elseif layer isa BlockMatrix_Layer
            key_W = "layer_$(i)_W"
            if haskey(data, key_W)
                if size(layer.W) != size(data[key_W])
                    @warn "Skipping checkpoint BlockMatrix weight with incompatible shape" layer=i key=key_W expected=size(layer.W) got=size(data[key_W])
                    continue
                end
                copyto!(layer.W, data[key_W])
                cache = layer._gpu_w_buf
                if cache isa Dict && haskey(cache, :W) && cache[:W] !== nothing
                    copyto!(cache[:W], layer.W)
                end
            end
            key_m = "layer_$(i)_adam_m"
            if haskey(data, key_m) && size(data[key_m]) == size(layer.W)
                cache = layer._gpu_w_buf
                if cache isa Dict
                    cache[:adam_m] = CuArray{ComplexF64}(data[key_m])
                    cache[:adam_v] = CuArray{ComplexF64}(data["layer_$(i)_adam_v"])
                else
                    # The GPU cache is built on the first GPU pass and picks these up.
                    layer._gpu_w_buf = Dict{Symbol,Any}(
                        :adam_m => data[key_m],
                        :adam_v => data["layer_$(i)_adam_v"],
                    )
                end
            end
        elseif layer isa fermionfield_Layer
            for j in eachindex(layer.x)
                key_x = "layer_$(i)_fermion_x_$(j)"
                haskey(data, key_x) || continue
                x = layer.x[j]
                _load_checkpoint_fermion_field!(x, data, key_x)

                key_m = "layer_$(i)_fermion_m_$(j)"
                if haskey(data, key_m)
                    m = similar(x)
                    v = similar(x)
                    _load_checkpoint_fermion_adam_field!(m, data[key_m], data, key_x)
                    _load_checkpoint_fermion_adam_field!(v, data["layer_$(i)_fermion_v_$(j)"], data, key_x)
                    opt.wf_state[objectid(x)] = AdamState{WF3x1}(m, v)
                end
            end
        end
    end

    CUDA.synchronize()
    return layers
end
