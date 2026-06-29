module Layer_types

abstract type Layer end
abstract type ActivationFunction end

import ..elementry_base: Hopping_operator, Spin_matrix,AbstractSpinMatrix,Paths,Random_SpinM
import ..gpu: Spin_matrix_GPU, Random_SpinM_GPU, Generate_weight


using Gaugefields
using LatticeDiracOperators
using CUDA

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}
const GPUgauge=Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}

export Layer, PTC_Layer, CPTC_Layer,ActivationFunction,Identity,Attention_Layer,Softmax,NormLayer,test_Layer,complexNormLayer
export minmaxNormLayer, zscoreNormLayer
export FullPTCAttention_Layer,Attention_Layer
export xavier_init!
export Dirac_Layer
export PrallelTransportAttention_Layer
export ModRelu
export siteweight_Layer
export RayleighQuotient_Layer
export PtcRayleighQuotient_Layer
# export L2Norm_Layer
export HoppingRayleighQuotient_Layer
export DenseLinearLayer
export PTLayer
export fermionfield_Layer
export Orthogonal_Layer
export Ritz_Layer
export ZeroMode_Layer
export GramSchmidt_Layer
export BlockScalar_Layer
export BlockMatrix_Layer

struct Identity <: ActivationFunction end

struct Softmax <: ActivationFunction end

struct RealSoftmax <: ActivationFunction end

struct ModRelu <: ActivationFunction end





function (Identity)(x::WF3x1)
    return x
end # 항등함수
# Identity를 함수처럼 동작하게 만들기
function (activation::Identity)(x)
    return x  # 항등 함수
end




# mutable struct L2Norm_Layer <: Layer
#     target::WF3x1
#     activation::ActivationFunction
#     function L2Norm_Layer(target::WF3x1,activation::ActivationFunction= Identity())
#         new(target,activation)
#     end
# end





mutable struct PTC_Layer{T<:AbstractSpinMatrix} <: Layer
    W::Vector{T}  
    hop::Hopping_operator   
    residual::Bool
    activation::ActivationFunction

    function PTC_Layer(hop::Hopping_operator{G}, W::Vector{T}, residual::Bool=false, activation::ActivationFunction=Identity()) where {G<:AbstractGaugefields, T<:AbstractArray{ComplexF64,2}}
        new{T}(W, hop, residual, activation)
    end

    function PTC_Layer(hop::Hopping_operator{G}, W::Union{Nothing, Vector{T}}=nothing, residual::Bool=false, activation::ActivationFunction=Identity()) where {G<:AbstractGaugefields, T<:AbstractArray{ComplexF64,2}}
        if W === nothing
            W = Generate_weight(hop)
        end
        return PTC_Layer(hop, W, residual, activation)
    end
end

mutable struct CPTC_Layer{T<:AbstractSpinMatrix} <: Layer
    W::Vector{T}  
    hop::Hopping_operator   
    residual::Bool   
    activation::ActivationFunction

    function CPTC_Layer(hop::Hopping_operator{G}, W::Union{Nothing, Vector{T}}=nothing, residual::Bool=false, activation::ActivationFunction=Identity()) where {G<:AbstractGaugefields, T<:AbstractArray{ComplexF64,2}}
        
        if W === nothing
            W = Generate_weight(hop, "Init")
        end
        
        new{T}(W, hop, residual, activation)
    end
end


struct Attention_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    
    residual::Bool

    function Attention_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number , activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator,activation, residual)
    end

    function Attention_Layer(Q::Layer, nominator::Number , activation::ActivationFunction=Identity(), residual::Bool=false)
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator, activation, residual)
    end
end


mutable struct test_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Float64
    test::WF3x1
    activation::ActivationFunction
    residual::Bool

    function test_Layer(Q::Layer, K::Layer, V::Layer, nominator::Float64 ,  test::WF3x1,  activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator, test , activation,  residual)
    end

    function test_Layer(Q::Layer, nominator::Float64 ,test::WF3x1, activation::ActivationFunction=Identity(), residual::Bool=false )
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator, test, activation,  residual)
    end
end




struct FullPTCAttention_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    residual::Bool

    function FullPTCAttention_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number ,    activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator,  activation,  residual)
    end

    function FullPTCAttention_Layer(Q::Layer, nominator::Number ,activation::ActivationFunction=Identity(), residual::Bool=false )
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator,  activation,  residual)
    end
end


struct PrallelTransportAttention_Layer <: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    residual::Bool
    bias::Float64

    function PrallelTransportAttention_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number ,  activation::ActivationFunction=Identity(), residual::Bool=false , bias::Float64=0.1)
        new(Q, K, V, nominator,  activation,  residual,bias)
    end

    function PrallelTransportAttention_Layer(Q::Layer, nominator::Number ,activation::ActivationFunction=Identity(), residual::Bool=false , bias::Float64=0.1)
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV

        new(Q, K, V, nominator,  activation,  residual, bias)
    end
end




struct siteweight_Layer<: Layer
    Q::Layer
    K::Layer
    V::Layer
    nominator::Number
    activation::ActivationFunction
    residual::Bool

    function siteweight_Layer(Q::Layer, K::Layer, V::Layer, nominator::Number ,  activation::ActivationFunction=Identity(), residual::Bool=false)
        new(Q, K, V, nominator,  activation,  residual)
    end

    function siteweight_Layer(Q::Layer, nominator::Number ,activation::ActivationFunction=Identity(), residual::Bool=false )
        K = deepcopy(Q)
        V = deepcopy(Q)
        WK = Generate_weight(K.hop)
        WV = Generate_weight(V.hop)
        K.W = WK
        V.W = WV
        new(Q, K, V, nominator,  activation,  residual)
    end
end



struct RayleighQuotient_Layer <: Layer
    x::WF3x1
    activation::ActivationFunction
    function RayleighQuotient_Layer(x::WF3x1, activation::ActivationFunction= Identity())
        new(x, activation)
    end


end 


mutable struct PtcRayleighQuotient_Layer <: Layer
    D::Any
    activation::ActivationFunction
    Out::Int
    function PtcRayleighQuotient_Layer(D::Any, activation::ActivationFunction= Identity(),Out =1)
        new(D, activation , Out)
    end
end 

mutable struct HoppingRayleighQuotient_Layer <: Layer
    hop::Hopping_operator
    x::WF3x1
    activation::ActivationFunction
    function HoppingRayleighQuotient_Layer(hop::Hopping_operator, x::WF3x1, activation::ActivationFunction= Identity())
        new(hop, x, activation)
    end
end

mutable struct DenseLinearLayer{T<:AbstractSpinMatrix} <: Layer
    W::Vector{T}
    In::Int
    Out::Int
    cuda::Bool
    activation::ActivationFunction
    # --- 캐시 버퍼 ---
    _result::Vector{WF3x1}
    _yi::Vector{WF3x1}          # yi 버퍼 (1-element)
    _dl_init::Bool
    _conj_buf::Vector{WF3x1}   # conj 작업용 버퍼 (DeltaRule!에서 재사용)
    _delta_init_dl::Bool
    # --- GPU batched 버퍼 ---
    _W_gpu::Any           # CuArray{ComplexF64,3} (NG,NG,NW)
    _Wadj_gpu::Any        # CuArray{ComplexF64,3}
    _input_all::Any       # CuArray{ComplexF64,5} (NC,NG,bs,rs,In)
    _result_all::Any      # CuArray{ComplexF64,5} (NC,NG,bs,rs,Out)
    _d_buf::Any           # CuArray{ComplexF64,5} (NC,NG,bs,rs,Out)
    _ci_all::Any          # CuArray{ComplexF64,5} (NC,NG,bs,rs,In)
    _grad_gpu::Any        # CuArray{ComplexF64,3} (NG,NG,NW)
    _adam_m_gpu::Any
    _adam_v_gpu::Any
    _W_cpu_buf::Any          # Array{ComplexF64,3} (NG,NG,NW) — sync_W_from_gpu! staging

    function DenseLinearLayer(In::Int, Out::Int, cuda::Bool, activation::ActivationFunction=Identity() )
        if cuda == true
            W = Vector{Spin_matrix_GPU}(undef, In*Out)
            for i in 1:In*Out
                W[i] = Random_SpinM_GPU("complex")
            end
        else
            W = Vector{Spin_matrix}(undef, In*Out)
            for i in 1:In*Out
                W[i] = Random_SpinM("complex")
            end
        end
        return new{eltype(W)}(W, In, Out, cuda, activation, WF3x1[], WF3x1[], false, WF3x1[], false,
                              nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end



mutable struct PTLayer{G<:AbstractGaugefields, T<:AbstractSpinMatrix} <: Layer
    paths::Vector{Paths}
    U::Vector{G}
    Out::Int
    W::Vector{T}
    activation::ActivationFunction
    # --- 캐시 버퍼 (매 epoch deepcopy 방지) ---
    _hopped::Vector{WF3x1}      # forward에서 hop 결과 저장 (gradient에서 재사용)
    _result::Vector{WF3x1}      # forward 출력 버퍼
    _initialized::Bool           # 버퍼 초기화 여부
    # --- DeltaRule 캐시 버퍼 ---
    _invpaths::Vector{Paths}     # reverse_paths 캐시 (paths 불변이므로 1회만 계산)
    _d_conj_buf::Vector{WF3x1}  # conj(d) 작업용 버퍼
    _ci_buffers::Vector{WF3x1}  # 스레드별 ci 버퍼
    _delta_init::Bool            # DeltaRule 버퍼 초기화 여부
    _Wadj::Vector{T}             # adjoint(W) 캐시 벡터
    _invhops::Vector{Any}        # 스레드별 Hopping_operator 캐시
    _hop_out::Vector{Vector{WF3x1}} # 스레드별 hop 출력 버퍼 (1-element)
    # --- GPU-resident batched buffers ---
    _W_gpu::Any                 # CuArray{ComplexF64,3} (4,4,NW)
    _Wadj_gpu::Any              # CuArray{ComplexF64,3} (4,4,NW) adjoint(W)
    _hopped_all::Any            # CuArray{ComplexF64,5} (NC,NG,bs,rs,npaths) — forward hop cache for gradient
    _grad_gpu::Any              # CuArray{ComplexF64,3} (4,4,NW) gradient on GPU
    _adam_m_gpu::Any            # CuArray{ComplexF64,3} Adam 1st moment
    _adam_v_gpu::Any            # CuArray{ComplexF64,3} Adam 2nd moment
    # --- Batched hopping cache ---
    _hop_dirs_gpu::Any           # CuArray{Int8,2} (max_steps, npaths) direction table
    _hop_nsteps_gpu::Any         # CuArray{Int32,1} (npaths,) step counts
    _hop_max_steps::Int          # max steps across all paths
    _inv_hop_dirs_gpu::Any       # CuArray{Int8,2} (max_steps, npaths) inverse direction table
    _W_cpu_buf::Any              # Array{ComplexF64,3} (4,4,NW) — sync_W_from_gpu! staging

    function PTLayer(paths::Vector{Paths}, U::Vector{G}, Out::Int,
                     activation::ActivationFunction=Identity()) where {G<:AbstractGaugefields}

        if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
            W = Vector{Spin_matrix_GPU}(undef, length(paths)*Out)
            for i in eachindex(W)
                W[i] = Random_SpinM_GPU("complex")
            end
            return new{G, Spin_matrix_GPU}(paths, U, Out, W, activation,
                       WF3x1[], WF3x1[], false,
                       Paths[], WF3x1[], WF3x1[], false,
                       Spin_matrix_GPU[], Any[], Vector{WF3x1}[],
                       nothing, nothing, nothing, nothing, nothing, nothing,
                       nothing, nothing, 0, nothing, nothing)
        else
            W = Vector{Spin_matrix}(undef, length(paths)*Out)
            for i in eachindex(W)
                W[i] = Random_SpinM("complex")
            end
            return new{G, Spin_matrix}(paths, U, Out, W, activation,
                       WF3x1[], WF3x1[], false,
                       Paths[], WF3x1[], WF3x1[], false,
                       Spin_matrix[], Any[], Vector{WF3x1}[],
                       nothing, nothing, nothing, nothing, nothing, nothing,
                       nothing, nothing, 0, nothing, nothing)
        end
    end
end




mutable struct fermionfield_Layer <: Layer
    x::Vector{WF3x1}
    In::Int
    activation::ActivationFunction
    function fermionfield_Layer(x::WF3x1, activation::ActivationFunction= Identity())
        i=1
        new([x], i, activation)
    end
    function fermionfield_Layer(x::WF3x1, In::Int, activation::ActivationFunction= Identity())
        xs=Vector{WF3x1}(undef, In)
        for i in 1:In
            xs[i]=deepcopy(x)
            gauss_distribution_fermion!(xs[i])
        end
        new(xs, In, activation)
    end
end


struct Orthogonal_Layer <: Layer
    coeff::Float64
    activation::ActivationFunction
    function Orthogonal_Layer(coeff::Float64, activation::ActivationFunction= Identity())
        new(coeff, activation)
    end 
end


mutable struct Ritz_Layer <: Layer
    D::Any
    activation::ActivationFunction
    Out::Int
    # --- 캐시 버퍼 (Dirac 재계산 방지) ---
    _Dz::Vector{Any}        # forward에서 D*z 캐시 → DeltaRule!에서 재사용
    _DdagDz::Vector{Any}    # D†Dz 캐시 (DeltaRule!에서 Dirac ops 절반)
    _VD::Vector{Any}         # DeltaRule! VD 버퍼 (CPU path only)
    _work::Vector{Any}       # DeltaRule! workspace [ψ, r] (CPU path only)
    _ritz_init::Bool
    # Gram matrix + eigen 캐시 — CPU path
    _A::Matrix{ComplexF64}
    _B::Matrix{ComplexF64}
    _C::Matrix{ComplexF64}
    _Bfac::Matrix{ComplexF64}
    _evals::Vector{Float64}
    _evecs::Matrix{ComplexF64}
    _m::Vector{ComplexF64}
    function Ritz_Layer(D::Any, activation::ActivationFunction= Identity(),Out =1)
        new(D, activation, Out,
            Any[], Any[], Any[], Any[], false,
            Matrix{ComplexF64}(undef,0,0), Matrix{ComplexF64}(undef,0,0),
            Matrix{ComplexF64}(undef,0,0), Matrix{ComplexF64}(undef,0,0),
            Float64[], Matrix{ComplexF64}(undef,0,0), ComplexF64[])
    end
end 


mutable struct ZeroMode_Layer <: Layer
    D::Any
    activation::ActivationFunction
    Out::Int
    _Dz::Vector{Any}
    _DdagDz::Vector{Any}
    _delta::Vector{Any}
    _losses::Vector{Float64}
    _denoms::Vector{Float64}
    _zeromode_init::Bool
    function ZeroMode_Layer(D::Any, activation::ActivationFunction=Identity(), Out=1)
        new(D, activation, Out, Any[], Any[], Any[], Float64[], Float64[], false)
    end
end


mutable struct GramSchmidt_Layer <: Layer
    x::Vector{WF3x1}           # forward에서 직교화 결과 캐시 (backward에서 재사용)
    activation::ActivationFunction
    function GramSchmidt_Layer(activation::ActivationFunction= Identity())
        new(WF3x1[], activation)
    end 
end





function (s::Softmax)(x::Spin_matrix)
    A = x.sm
    maxA = maximum(abs.(A), dims=2)      # 각 행의 복소수 절댓값 최대
    expA = exp.(A .- maxA)               # 안정화된 지수
    row_sum = sum(expA, dims=2)
    return Spin_matrix(expA ./ row_sum)
end


function (s::Softmax)(x::Vector{Spin_matrix})
    y = similar(x)

    for i in 1:length(x)
        A = x[i].sm
        maxA = maximum(abs.(A), dims=2)
        expA = exp.(A .- maxA)
        row_sum = sum(expA, dims=2)
        y[i] = Spin_matrix(expA ./ row_sum)
    end

    return y
end

function (s::Softmax)(x::Spin_matrix_GPU)
    A= x.sm
    maxA = maximum(abs.(A), dims=2)      # 각 행의 복소수 절댓값 최대
    expA = exp.(A .- maxA)               # 안정화된 지수
    row_sum = sum(expA, dims=2)
    return Spin_matrix_GPU(expA ./ row_sum)
end

function (s::Softmax)(x::Vector{Spin_matrix_GPU})
    y = similar(x)

    for i in 1:length(x)
        A = x[i].sm
        maxA = maximum(abs.(A), dims=2)
        expA = exp.(A .- maxA)
        row_sum = sum(expA, dims=2)
        y[i] = Spin_matrix_GPU(expA ./ row_sum)
    end

    return y
end

function (r::ModRelu)(x::Spin_matrix)
    A = x.sm
    modA = abs.(A)
    phaseA = angle.(A)
    activated_modA = max.(modA .-0.1, 0.0)  # ReLU 적용
    activated_A = activated_modA .* exp.(im .* phaseA)
    return Spin_matrix(activated_A)
end

# function (r::ModRelu)(x::Spin_matrix)
#     A = x.sm
#     for i in 1:4
#         if abs(A[i,i])>0.1
#             A[i,i]= (abs(A[i,i])-50.0)*A[i,i]/abs(A[i,i])
#         else
#             A[i,i]=0
#         end
#     end

#     return Spin_matrix(A)
# end







function (r::ModRelu)(xs::Vector{Spin_matrix})
    ys = similar(xs)
    @inbounds for i in eachindex(xs)
        ys[i] = r(xs[i])
    end
    return ys
end











struct Decoder <: Layer
    activation::ActivationFunction
    function Decoder(activation::ActivationFunction )
        new(activation)
    end 
end 

struct Dirac_Layer <: Layer
    activation
    W::Nothing
    function Dirac_Layer(activation)
        new(activation)
    end
end



mutable struct NormLayer <: Layer
    activation::ActivationFunction
    alpha::Float64
    _result::Vector{WF3x1}
    _norm_init::Bool
    _conj_buf::Vector{WF3x1}   # conj 작업용 버퍼 (DeltaRule!에서 재사용)

    function NormLayer(activation::ActivationFunction= Identity(),alpha::Float64=1.0)
        new(activation,alpha, WF3x1[], false, WF3x1[])
    end

end 

mutable struct complexNormLayer <: Layer
    activation::ActivationFunction

    function complexNormLayer(activation::ActivationFunction= Identity())
        new(activation)
    end

end


mutable struct zscoreNormLayer <: Layer
    activation::ActivationFunction

    function zscoreNormLayer(activation::ActivationFunction= Identity())
        new(activation)
    end
end 

mutable struct minmaxNormLayer <: Layer
    activation::ActivationFunction

    function minmaxNormLayer(activation::ActivationFunction= Identity())
        new(activation)
    end
end

function xavier_init!(W::Vector{T}, fan_in::Int, fan_out::Int) where {T<:AbstractArray{ComplexF64,2}}
    scale = sqrt(6 / (fan_in + fan_out))
    for M in W
        for i in 1:size(M, 1), j in 1:size(M, 2)
            real_part = rand() * 2 * scale - scale
            imag_part = rand() * 2 * scale - scale
            M[i, j] = complex(real_part, imag_part)
        end
    end
end



mutable struct LossSumLayer{L1<:Layer,L2<:Layer} <: Layer
    l1::L1
    l2::L2
    w1::Float64
    w2::Float64
    activation::ActivationFunction

    function LossSumLayer(l1::L1, l2::L2,
                          w1::Real=1.0, w2::Real=1.0,
                          activation::ActivationFunction=Identity()) where {L1<:Layer,L2<:Layer}
        new{L1,L2}(l1, l2, Float64(w1), Float64(w2), activation)
    end
end


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












# --- CuArray 표시 시 CUDA context 에러 방지 ---
function Base.show(io::IO, layer::Ritz_Layer)
    print(io, "Ritz_Layer(Out=$(layer.Out))")
end

function Base.show(io::IO, layer::ZeroMode_Layer)
    print(io, "ZeroMode_Layer(Out=$(layer.Out))")
end

function Base.show(io::IO, layer::PTLayer)
    print(io, "PTLayer(paths=$(length(layer.paths)), Out=$(layer.Out))")
end

end # module Layer_types
