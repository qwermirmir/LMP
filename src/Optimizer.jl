module Optimizer

using LatticeDiracOperators
using CUDA
using JLD2

import ..elementry_base: Hopping_operator, Spin_matrix, inner_product, Random_SpinM
import ..Layer_types: CPTC_Layer, PTC_Layer, Layer, ActivationFunction, Identity,Attention_Layer,NormLayer,test_Layer,FullPTCAttention_Layer,PtcRayleighQuotient_Layer
import ..elementry_base: AbstractSpinMatrix
import ..Layer_types: PrallelTransportAttention_Layer,siteweight_Layer,Dirac_Layer,RayleighQuotient_Layer,HoppingRayleighQuotient_Layer,DenseLinearLayer,PTLayer
import ..Layer_types: fermionfield_Layer
import ..Layer_types: BlockScalar_Layer, BlockMatrix_Layer

import ..gpu: gpu_adam_update!, blockscalar_adam_update!, sync_W_from_gpu!, sync_W_to_gpu!, Spin_matrix_GPU, zero_wf!

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

import LatticeDiracOperators: Dirac_Operator

abstract type Opt end

export Adam, update!, sync_layers!, sync_layers_to_gpu!, prealloc_optimizer_state!
export save_checkpoint, load_checkpoint!
export Opt




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
    opt.t += 1

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
    
    opt.t += 1
    
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
    opt.t += 1

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
    opt.t += 1






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
    opt.t += 1





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
    opt.t += 1






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
    opt.t += 1

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
    opt.t += 1

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
    opt.t += 1

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
    opt.t += 1
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

    opt.t += 1

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
    opt.t += 1          # 타임스텝 증가 (CPTC_Layer와 동일 패턴)
    wf_adam_step!(layer.x, opt, grad)
    return nothing
end

# HoppingRayleighQuotient_Layer: x :: WF3x1
function update!(layer::HoppingRayleighQuotient_Layer, opt::Adam, grad::WF3x1)
    opt.t += 1
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


using CUDA

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
    opt.t += 1
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

        layer.W[idx] -= complex(Δ_re, Δ_im)
    end
end

function update!(layer::BlockScalar_Layer, opt::Adam, grad)
    if grad isa AbstractVector && !isempty(grad) && grad[1] isa CuArray
        opt.t += 1
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
    opt.t += 1
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
        opt.t += 1
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
        blockscalar_adam_update!(W_gpu, cache[:adam_m], cache[:adam_v], grad_gpu,
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
    blockscalar_adam_update!(W_gpu, cache[:adam_m], cache[:adam_v], cache[:grad],
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

"""
    load_checkpoint!(path, layers, opt)

Restores W + Adam state into live GPU layers from a W-only checkpoint.
No substitute_Layer! needed — layers stay on GPU.
Falls back silently if a layer's key is absent (partial checkpoint).
"""
function load_checkpoint!(path::AbstractString, layers, opt::Adam)
    data = JLD2.load(path)
    opt.t = Int(get(data, "opt_t", 0))

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
                if size(layer.W) != size(data[key_W])
                    @warn "Skipping checkpoint BlockScalar weight with incompatible shape" layer=i key=key_W expected=size(layer.W) got=size(data[key_W])
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
                end
            end
        elseif layer isa fermionfield_Layer
            for j in eachindex(layer.x)
                key_x = "layer_$(i)_fermion_x_$(j)"
                haskey(data, key_x) || continue
                x = layer.x[j]
                copyto!(x.f, data[key_x])

                key_m = "layer_$(i)_fermion_m_$(j)"
                if haskey(data, key_m)
                    m = similar(x)
                    v = similar(x)
                    copyto!(m.f, data[key_m])
                    copyto!(v.f, data["layer_$(i)_fermion_v_$(j)"])
                    opt.wf_state[objectid(x)] = AdamState{WF3x1}(m, v)
                end
            end
        end
    end

    CUDA.synchronize()
    return layers
end

end # module Optimizer
