module Optimizer

using LatticeDiracOperators
using CUDA

import ..elementry_base: Hopping_operator, Spin_matrix, inner_product, Random_SpinM
import ..Layer_types: CPTC_Layer, PTC_Layer, Layer, ActivationFunction, Identity,Attention_Layer,NormLayer,test_Layer,FullPTCAttention_Layer,PtcRayleighQuotient_Layer
import ..elementry_base: AbstractSpinMatrix
import ..Layer_types: PrallelTransportAttention_Layer,siteweight_Layer,Dirac_Layer,RayleighQuotient_Layer,HoppingRayleighQuotient_Layer,DenseLinearLayer,PTLayer
import ..Layer_types: fermionfield_Layer

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

import LatticeDiracOperators: Dirac_Operator

abstract type Opt end

export Adam, update!
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

    g_re = real(grad.sm)
    g_im = imag(grad.sm)

    m_re = real(m.sm)
    m_im = imag(m.sm)

    v_re = real(v.sm)
    v_im = imag(v.sm)

    m_re .= opt.β1 .* m_re .+ (1 - opt.β1) .* g_re
    v_re .= opt.β2 .* v_re .+ (1 - opt.β2) .* (g_re .^ 2)

    m_im .= opt.β1 .* m_im .+ (1 - opt.β1) .* g_im
    v_im .= opt.β2 .* v_im .+ (1 - opt.β2) .* (g_im .^ 2)

    m.sm .= complex.(m_re, m_im)
    v.sm .= complex.(v_re, v_im)

    m_hat_re = m_re ./ (1 - opt.β1^opt.t)
    v_hat_re = v_re ./ (1 - opt.β2^opt.t)

    m_hat_im = m_im ./ (1 - opt.β1^opt.t)
    v_hat_im = v_im ./ (1 - opt.β2^opt.t)

    update_real = opt.α .* m_hat_re ./ (sqrt.(v_hat_re) .+ opt.ε)
    update_imag = opt.α .* m_hat_im ./ (sqrt.(v_hat_im) .+ opt.ε)

    param.sm .-= update_real .- im .* update_imag


end   




function realProcess(param::T, opt:: Adam , grad ) where {T<:AbstractSpinMatrix}
    key = objectid(param)

    if !haskey(opt.m, key)
        opt.m[key] = Random_SpinM("zero")
        opt.v[key] = Random_SpinM("zero")
    end

    m = opt.m[key]
    v = opt.v[key]

    g = real(grad.sm)  # 실수영자 계산

    m.sm .= opt.β1 .* m.sm .+ (1 - opt.β1) .* g
    v.sm .= opt.β2 .* v.sm .+ (1 - opt.β2) .* (g .^ 2)

    m_hat = m.sm ./ (1 - opt.β1^opt.t)
    v_hat = v.sm ./ (1 - opt.β2^opt.t)

    param.sm .-= opt.α .* m_hat ./ (sqrt.(v_hat) .+ opt.ε)
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


<<<<<<< HEAD

=======
>>>>>>> origin/fix/bigfiles
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

    param = layer.W
    if length(param) ==1
        complexProcess(param, opt , grad[1] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[i] )
        end
    end

end

function update!(layer::PTLayer, opt::Adam, grad)
    opt.t += 1

    param = layer.W
    if length(param) ==1
        complexProcess(param[1], opt , grad[1][1] )
    
    else 
        for i in 1:length(param)
            complexProcess(param[i], opt , grad[1][i] )
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
        AdamState{WF3x1}(similar(param), similar(param))
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





end # module Optimizer


