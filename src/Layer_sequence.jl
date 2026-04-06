module Layer_sequence





import ..Layer_types: Layer, CPTC_Layer, PTC_Layer,ActivationFunction,Identity,Attention_Layer,Softmax,NormLayer,test_Layer,complexNormLayer,minmaxNormLayer,zscoreNormLayer,RealAttention_Layer
import ..Layer_types: FullPTCAttention_Layer,Dirac_Layer,PrallelTransportAttention_Layer,siteweight_Layer,RayleighQuotient_Layer,PtcRayleighQuotient_Layer,HoppingRayleighQuotient_Layer
import ..Layer_types: Orthogonal_Layer,LossSumLayer,Ritz_Layer
import ..elementry_base: Hopping_operator, Spin_matrix, inner_product,g2vector_generator!,addmul!
import ..elementry_base: Clifford, Random_SpinM,norm2,mean,standardization,minmax,spincolortrace
import ..Optimizer: Opt, update!, Adam
import ..activation:  gradient, DeltaRule,DeltaRule!
import ..gpu: Matmul!,gpuaddmul!,zero_wf!,Convertsm,Spin_matrix_GPU
import ..Layer_types: DenseLinearLayer, PTLayer,fermionfield_Layer,GramSchmidt_Layer
import ..Layer_types: BlockScalar_Layer
import ..elementry_base: blockize, unblockize
abstract type Operator end







# using LatticeQCD
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Plots
using CUDA

# import LatticeDiracOperators: Dirac_Operators

const WF3x1CPU = LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}
const WF3x1GPU =LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_accelerator{3, CuArray{ComplexF64, 4, CUDA.DeviceMemory}, 4, :cuda, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, CuArray{ComplexF64, 4, CUDA.DeviceMemory}}

const WF3x1 = Union{WF3x1CPU, WF3x1GPU}
# const Dirac_operator =LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}
# const DagDirac_operator =LatticeDiracOperators.Dirac_operators.Adjoint_Wilson_operator{LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator{4, Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing{3}, LatticeDiracOperators.Dirac_operators.WilsonFermion_4D_nowing{3}}}


export forward_propagation, Lossfunction, train,  preD,Operator ,doubletrain,backward_propagation!
export refresh_fermionfield!, refresh_model!
export substitute_Layer!
export _to_cpu, _to_gpu,_to_cpu_gauge



function substitute_Layer!(layers::Vector{<:Layer})
    # 첫 번째 layer의 GPU/CPU 여부를 판별하여 반대 방향으로 변환
    if _is_gpu_layer(layers)
        # GPU → CPU
        for i in eachindex(layers)
            layers[i] = _to_cpu(layers[i])
        end
    else
        # CPU → GPU
        for i in eachindex(layers)
            layers[i] = _to_gpu(layers[i])
        end
    end
    return layers
end

# --- Layer 벡터가 GPU인지 판별 ---
function _is_gpu_layer(layers::Vector{<:Layer})
    for layer in layers
        if _has_gpu(layer)
            return true
        end
    end
    return false
end

_has_gpu(::Layer) = false
_has_gpu(layer::PTC_Layer{Spin_matrix_GPU}) = true
_has_gpu(layer::PTC_Layer) = eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
_has_gpu(layer::CPTC_Layer{Spin_matrix_GPU}) = true
_has_gpu(layer::CPTC_Layer) = eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
_has_gpu(layer::DenseLinearLayer{Spin_matrix_GPU}) = true
_has_gpu(layer::PTLayer) = eltype(layer.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
_has_gpu(layer::Attention_Layer) = _has_gpu(layer.Q)
_has_gpu(layer::FullPTCAttention_Layer) = _has_gpu(layer.Q)
_has_gpu(layer::PrallelTransportAttention_Layer) = _has_gpu(layer.Q)
_has_gpu(layer::siteweight_Layer) = _has_gpu(layer.Q)
_has_gpu(layer::test_Layer) = _has_gpu(layer.Q)
_has_gpu(layer::LossSumLayer) = _has_gpu(layer.l1) || _has_gpu(layer.l2)
_has_gpu(layer::fermionfield_Layer) = !isempty(layer.x) && layer.x[1] isa WF3x1GPU
_has_gpu(layer::GramSchmidt_Layer) = !isempty(layer.x) && layer.x[1] isa WF3x1GPU
function _has_gpu(layer::Ritz_Layer)
    try; return eltype(layer.D.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator; catch; return false; end
end
function _has_gpu(layer::PtcRayleighQuotient_Layer)
    try; return eltype(layer.D.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator; catch; return false; end
end
function _has_gpu(layer::HoppingRayleighQuotient_Layer)
    return eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator || layer.x isa WF3x1GPU
end

# --- GPU→CPU gauge field 변환 ---
function _to_cpu_gauge(U::Vector{G}) where {G<:AbstractGaugefields}

        NC = U[1].NC
        NX = U[1].NX; NY = U[1].NY; NZ = U[1].NZ; NT = U[1].NT
        Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
        try
            substitute_U!(Ucpu, U)
        catch
            @warn "GPU gauge 데이터 복구 불가 → identity gauge로 대체. layer.U를 수동으로 설정하세요."
        end
        return Ucpu

end



# --- GPU→CPU Hopping_operator 변환 ---
function _to_cpu_hop(hop::Hopping_operator)
    Ucpu = _to_cpu_gauge(hop.Gauge)
    if Ucpu === hop.Gauge
        return hop
    end
    return Hopping_operator(Ucpu, hop.P)
end

# --- GPU→CPU Spin matrix 벡터 변환 ---
_to_cpu_W(W::Vector{Spin_matrix_GPU}) = Convertsm(W)
_to_cpu_W(W::Vector{Spin_matrix}) = W
_to_cpu_W(W::Vector{T}) where {T<:AbstractArray{ComplexF64,2}} = W

# --- GPU→CPU fermion field 변환 ---
# GPU 메모리가 해제된 경우 (JLD2 로드 등) 빈 CPU fermion으로 대체
function _to_cpu_fermion(x::WF3x1GPU)
    NC = x.NC
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    xcpu = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
    try
        substitute_fermion!(xcpu, x)
    catch
        @warn "GPU fermion 데이터 복구 불가 → 0으로 초기화된 CPU fermion으로 대체"
    end
    return xcpu
end
_to_cpu_fermion(x::WF3x1CPU) = x

function _to_cpu_fermions(xs::Vector{<:WF3x1})
    if isempty(xs) || xs[1] isa WF3x1CPU
        return xs
    end
    NC = xs[1].NC
    NX = xs[1].NX; NY = xs[1].NY; NZ = xs[1].NZ; NT = xs[1].NT
    Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold")
    result = Vector{WF3x1CPU}(undef, length(xs))
    warned = false
    for i in eachindex(xs)
        result[i] = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
        if xs[i] isa WF3x1GPU
            try
                substitute_fermion!(result[i], xs[i])
            catch
                if !warned
                    @warn "GPU fermion 데이터 복구 불가 → 0으로 초기화된 CPU fermion으로 대체"
                    warned = true
                end
            end
        elseif xs[i] isa WF3x1CPU
            substitute_fermion!(result[i], xs[i])
        end
    end
    return result
end

# --- 기본: GPU 성분이 없는 Layer ---
_to_cpu(layer::Layer) = layer

# --- PTC_Layer ---
function _to_cpu(layer::PTC_Layer)
    hop_cpu = _to_cpu_hop(layer.hop)
    W_cpu = _to_cpu_W(layer.W)
    return PTC_Layer(hop_cpu, W_cpu, layer.residual, layer.activation)
end

# --- CPTC_Layer ---
function _to_cpu(layer::CPTC_Layer)
    hop_cpu = _to_cpu_hop(layer.hop)
    W_cpu = _to_cpu_W(layer.W)
    return CPTC_Layer(hop_cpu, W_cpu, layer.residual, layer.activation)
end

# --- Attention_Layer ---
function _to_cpu(layer::Attention_Layer)
    return Attention_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                           layer.nominator, layer.activation, layer.residual)
end

# --- test_Layer ---
function _to_cpu(layer::test_Layer)
    layer.Q = _to_cpu(layer.Q)
    layer.K = _to_cpu(layer.K)
    layer.V = _to_cpu(layer.V)
    if layer.test isa WF3x1GPU
        layer.test = _to_cpu_fermion(layer.test)
    end
    return layer
end

# --- FullPTCAttention_Layer ---
function _to_cpu(layer::FullPTCAttention_Layer)
    return FullPTCAttention_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                                  layer.nominator, layer.activation, layer.residual)
end

# --- PrallelTransportAttention_Layer ---
function _to_cpu(layer::PrallelTransportAttention_Layer)
    return PrallelTransportAttention_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                                            layer.nominator, layer.activation, layer.residual, layer.bias)
end

# --- siteweight_Layer ---
function _to_cpu(layer::siteweight_Layer)
    return siteweight_Layer(_to_cpu(layer.Q), _to_cpu(layer.K), _to_cpu(layer.V),
                            layer.nominator, layer.activation, layer.residual)
end

# --- DenseLinearLayer ---
function _to_cpu(layer::DenseLinearLayer{Spin_matrix_GPU})
    new_layer = DenseLinearLayer(layer.In, layer.Out, false, layer.activation)
    W_cpu = Convertsm(layer.W)
    for j in eachindex(W_cpu)
        new_layer.W[j] = W_cpu[j]
    end
    return new_layer
end

# --- PTLayer ---
function _to_cpu(layer::PTLayer)
    U_cpu = _to_cpu_gauge(layer.U)
    new_layer = PTLayer(layer.paths, U_cpu, layer.Out, layer.activation)
    W_cpu = _to_cpu_W(layer.W)
    new_layer.W = W_cpu
    return new_layer
end

# --- fermionfield_Layer ---
function _to_cpu(layer::fermionfield_Layer)
    layer.x = _to_cpu_fermions(layer.x)
    return layer
end

# --- GramSchmidt_Layer ---
function _to_cpu(layer::GramSchmidt_Layer)
    layer.x = _to_cpu_fermions(layer.x)
    return layer
end

# --- HoppingRayleighQuotient_Layer ---
function _to_cpu(layer::HoppingRayleighQuotient_Layer)
    layer.hop = _to_cpu_hop(layer.hop)
    if layer.x isa WF3x1GPU
        layer.x = _to_cpu_fermion(layer.x)
    end
    return layer
end

# --- LossSumLayer ---
function _to_cpu(layer::LossSumLayer)
    l1_cpu = _to_cpu(layer.l1)
    l2_cpu = _to_cpu(layer.l2)
    return LossSumLayer(l1_cpu, l2_cpu, layer.w1, layer.w2, layer.activation)
end

# --- Dirac operator GPU→CPU 변환 (D.U로 gauge field 접근) ---
function _to_cpu_dirac(D)
    U = D.U
    if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        Ucpu = _to_cpu_gauge(U)
        xcpu = Initialize_pseudofermion_fields(Ucpu[1], "Wilson", nowing=true)
        params = Dict{String,Any}()
        params["Dirac_operator"] = "Wilson"
        params["κ"] = D.κ
        params["eps_CG"] = D.eps_CG
        params["faster version"] = isa(D, LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator_faster)
        return Dirac_operator(Ucpu, xcpu, params)
    else
        return D
    end
end

# --- Ritz_Layer ---
function _to_cpu(layer::Ritz_Layer)
    layer.D = _to_cpu_dirac(layer.D)
    return layer
end

# --- PtcRayleighQuotient_Layer ---
function _to_cpu(layer::PtcRayleighQuotient_Layer)
    layer.D = _to_cpu_dirac(layer.D)
    return layer
end


# ============================================================
#  CPU → GPU 변환
# ============================================================

# --- CPU→GPU gauge field 변환 ---
function _to_gpu_gauge(U::Vector{G}) where {G<:AbstractGaugefields}
    if G <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return U  # 이미 GPU
    end
    NC = U[1].NC
    NX = U[1].NX; NY = U[1].NY; NZ = U[1].NZ; NT = U[1].NT
    Ugpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    substitute_U!(Ugpu, U)
    return Ugpu
end

# --- CPU→GPU Hopping_operator 변환 ---
function _to_gpu_hop(hop::Hopping_operator)
    Ugpu = _to_gpu_gauge(hop.Gauge)
    if Ugpu === hop.Gauge
        return hop
    end
    return Hopping_operator(Ugpu, hop.P)
end

# --- CPU→GPU Spin matrix 벡터 변환 ---
_to_gpu_W(W::Vector{Spin_matrix_GPU}) = W
_to_gpu_W(W::Vector{Spin_matrix}) = [Spin_matrix_GPU(w) for w in W]

# --- CPU→GPU fermion field 변환 ---
function _to_gpu_fermion(x::WF3x1CPU)
    NC = x.NC
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    xgpu = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
    substitute_fermion!(xgpu, x)
    return xgpu
end
_to_gpu_fermion(x::WF3x1GPU) = x

function _to_gpu_fermions(xs::Vector{<:WF3x1})
    if isempty(xs) || xs[1] isa WF3x1GPU
        return xs
    end
    NC = xs[1].NC
    NX = xs[1].NX; NY = xs[1].NY; NZ = xs[1].NZ; NT = xs[1].NT
    Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    result = Vector{WF3x1GPU}(undef, length(xs))
    for i in eachindex(xs)
        result[i] = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
        substitute_fermion!(result[i], xs[i])
    end
    return result
end

# --- Dirac operator CPU→GPU 변환 ---
function _to_gpu_dirac(D)
    U = D.U
    if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return D  # 이미 GPU
    end
    Ugpu = _to_gpu_gauge(U)
    xgpu = Initialize_pseudofermion_fields(Ugpu[1], "Wilson", nowing=true)
    params = Dict{String,Any}()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = D.κ
    params["eps_CG"] = D.eps_CG
    params["faster version"] = isa(D, LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator_faster)
    return Dirac_operator(Ugpu, xgpu, params)
end

# --- 기본: CPU 전용 성분만 있는 Layer ---
_to_gpu(layer::Layer) = layer

# --- PTC_Layer ---
function _to_gpu(layer::PTC_Layer)
    hop_gpu = _to_gpu_hop(layer.hop)
    W_gpu = _to_gpu_W(layer.W)
    return PTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

# --- CPTC_Layer ---
function _to_gpu(layer::CPTC_Layer)
    hop_gpu = _to_gpu_hop(layer.hop)
    W_gpu = _to_gpu_W(layer.W)
    return CPTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

# --- Attention_Layer ---
function _to_gpu(layer::Attention_Layer)
    return Attention_Layer(_to_gpu(layer.Q), _to_gpu(layer.K), _to_gpu(layer.V),
                           layer.nominator, layer.activation, layer.residual)
end

# --- test_Layer ---
function _to_gpu(layer::test_Layer)
    layer.Q = _to_gpu(layer.Q)
    layer.K = _to_gpu(layer.K)
    layer.V = _to_gpu(layer.V)
    if layer.test isa WF3x1CPU
        layer.test = _to_gpu_fermion(layer.test)
    end
    return layer
end

# --- FullPTCAttention_Layer ---
function _to_gpu(layer::FullPTCAttention_Layer)
    return FullPTCAttention_Layer(_to_gpu(layer.Q), _to_gpu(layer.K), _to_gpu(layer.V),
                                  layer.nominator, layer.activation, layer.residual)
end

# --- PrallelTransportAttention_Layer ---
function _to_gpu(layer::PrallelTransportAttention_Layer)
    return PrallelTransportAttention_Layer(_to_gpu(layer.Q), _to_gpu(layer.K), _to_gpu(layer.V),
                                            layer.nominator, layer.activation, layer.residual, layer.bias)
end

# --- siteweight_Layer ---
function _to_gpu(layer::siteweight_Layer)
    return siteweight_Layer(_to_gpu(layer.Q), _to_gpu(layer.K), _to_gpu(layer.V),
                            layer.nominator, layer.activation, layer.residual)
end

# --- DenseLinearLayer ---
function _to_gpu(layer::DenseLinearLayer{Spin_matrix})
    new_layer = DenseLinearLayer(layer.In, layer.Out, true, layer.activation)
    W_gpu = [Spin_matrix_GPU(w) for w in layer.W]
    for j in eachindex(W_gpu)
        new_layer.W[j] = W_gpu[j]
    end
    return new_layer
end

# --- PTLayer ---
function _to_gpu(layer::PTLayer)
    U_gpu = _to_gpu_gauge(layer.U)
    new_layer = PTLayer(layer.paths, U_gpu, layer.Out, layer.activation)
    W_gpu = _to_gpu_W(layer.W)
    new_layer.W = W_gpu
    return new_layer
end

# --- fermionfield_Layer ---
function _to_gpu(layer::fermionfield_Layer)
    layer.x = _to_gpu_fermions(layer.x)
    return layer
end

# --- GramSchmidt_Layer ---
function _to_gpu(layer::GramSchmidt_Layer)
    layer.x = _to_gpu_fermions(layer.x)
    return layer
end

# --- HoppingRayleighQuotient_Layer ---
function _to_gpu(layer::HoppingRayleighQuotient_Layer)
    layer.hop = _to_gpu_hop(layer.hop)
    if layer.x isa WF3x1CPU
        layer.x = _to_gpu_fermion(layer.x)
    end
    return layer
end

# --- Ritz_Layer ---
function _to_gpu(layer::Ritz_Layer)
    layer.D = _to_gpu_dirac(layer.D)
    return layer
end

# --- PtcRayleighQuotient_Layer ---
function _to_gpu(layer::PtcRayleighQuotient_Layer)
    layer.D = _to_gpu_dirac(layer.D)
    return layer
end

# --- LossSumLayer ---
function _to_gpu(layer::LossSumLayer)
    l1_gpu = _to_gpu(layer.l1)
    l2_gpu = _to_gpu(layer.l2)
    return LossSumLayer(l1_gpu, l2_gpu, layer.w1, layer.w2, layer.activation)
end


function Lossfunction(f::WF3x1,t::WF3x1)

    loss=f-t
    
    return inner_product(loss,loss)

end




function Lossfunction(f::Vector{WF3x1}, t::Vector{WF3x1})
    @assert length(f) == length(t)
    loss = 0.0
    for i in 1:length(f)
        diff = f[i] - t[i]
        loss += inner_product(diff, diff)
    end
    return loss / length(f)
end




function (layer::PTC_Layer)(x::WF3x1)
    result = similar(x)
    fields = layer.hop(x)  # Hopping operator를 적용한 field들
    c=similar(x)

    if length(layer.W) == 1
        # 가중치가 하나뿐인 경우 (모든 field에 같은 가중치를 곱함)
        for i in 1:length(fields)
            mul!(c, layer.W[1], fields[i])
            axpby!(1,c,1,result)
        end
    elseif length(layer.W) == length(fields)
        # 가중치 벡터와 field 벡터 크기가 같은 경우
        for i in 1:length(fields)
            mul!(c, layer.W[i], fields[i])
            axpby!(1,c,1,result)
        end
    else
        error("W의 길이가 1이거나 fields와 같아야 합니다.")
    end


    result = layer.activation(result)

    if layer.residual == true
        # result = result + x
        axpby!(1,x,1,result)
    end


    return result
end

function (layer::CPTC_Layer)(x::WF3x1)

    fields = layer.hop(x)  # Hopping operator를 적용한 field들
    cW=Clifford(layer.W)
    y=similar(x)
    c=similar(x)



    if length(layer.W) == 1
        # 가중치가 하나뿐인 경우 (모든 field에 같은 가중치를 곱함)
        for i in 1:length(fields)
            mul!(c, cW[1], fields[i])
            axpby!(1,c,1,y)
        end
    elseif length(layer.W) == length(fields)
        # 가중치 벡터와 field 벡터 크기가 같은 경우
        for i in 1:length(fields)
            mul!(c, cW[i], fields[i])
            axpby!(1,c,1,y)
        end
    else
        error("W의 길이가 1이거나 fields와 같아야 합니다.")
    end


    result =layer.activation(y)

    if layer.residual == true
        # result = result + x
        axpby!(1,x,1,result)
    end

    return result 
end


function  (layer::Attention_Layer)(x::WF3x1)


    QW=Clifford(layer.Q.W)
    KW=Clifford(layer.K.W)
    VW=Clifford(layer.V.W)

    lW=length(QW)


    m=layer.nominator


    Hx=layer.V.hop(x)



    Q=similar(x)
    K=similar(x)
    M=similar(QW)
    S=similar(QW)
    for i in 1:lW
        mul!(Q,QW[i],Hx[i])
        mul!(K,KW[i],Hx[i])

        S[i]=layer.activation(Matmul!(conj(Q),K)/m)
        # M[i]=Matmul!(conj(Q),K)/m
        # println("M[$(i)]: ", M[i])
        # println("S[$(i)]: ", S[i])

    
    end 

    for i in 1:lW
        mul!(S[i],S[i],VW[i])
    end




    z=similar(x)
    

    temp =similar(x)
    for i in 1:length(KW)
        mul!(temp,S[i],Hx[i])
        axpby!(1,temp,1,z)
    end



    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end 
    return z




end




function  (layer::test_Layer)(x::WF3x1)


    # QW=Clifford(layer.Q.W)
    KW=Clifford(layer.K.W)
    VW=Clifford(layer.V.W)
    t=layer.test


    m=layer.nominator


    Hx=layer.V.hop(x)
    Ht=layer.V.hop(t)

    # if Q.W === nothing || K.W === nothing || V.W === nothing
    #     error("gradient() 호출 시 Q, K, V의 W가 정의되지 않았습니다.")
    # end

    Kx=KW*layer.K.hop(x)
    # Qx=KW*layer.Q.hop(x)
    # Vx=VW*layer.V.hop(x)

    M=similar(KW)
    S=similar(KW)
    for i in 1:length(KW)
        M[i]=Matmul!(conj(Kx[i]-t),Kx[i]-t)/m
        S[i]=layer.activation(M[i])
    end 

    Hx=layer.V.hop(x)
    # W=S*VW


    for i in 1:length(KW)
        mul!(S[i],S[i],VW[i])
    end


    z=similar(x)
    temp= similar(x)
    
    # for i in 1:length(KW)
    #     @inbounds z+=W[i]*Hx[i]
    # end

    for i in 1:length(KW)
        mul!(temp,S[i],Hx[i])
        axpby!(1,temp,1,z)
    end



    # if layer.residual == true 
    #     z=z+x
    # end 

    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end

    return z

end


function (layer::FullPTCAttention_Layer)(x::WF3x1)


    QW = layer.Q.W
    KW = layer.K.W
    VW = layer.V.W




    lW=length(KW)


    m=layer.nominator


    Hx=layer.V.hop(x)


    # if Q.W === nothing || K.W === nothing || V.W === nothing
    #     error("gradient() 호출 시 Q, K, V의 W가 정의되지 않았습니다.")
    # end

    Kx=KW*layer.K.hop(x)
    Qx=QW*layer.Q.hop(x)


    M=similar(KW)
    S=similar(KW)
    for i in 1:lW
        M[i] = Matmul!(conj(Qx[i]), Kx[i]) / m
        S[i] = layer.activation(M[i])
        
    end
    

    Hx=layer.V.hop(x)
    W=S*VW
    z=similar(x)


    temp= similar(x)

    for i in 1:length(KW)
        mul!(temp,W[i],Hx[i])
        axpby!(1,temp,1,z)
    end

    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end




    return z
end 

function (D::Dirac_Layer)(x::WF3x1)

    Dirac=D.activation
    y=Dirac*x
    return y
end 

function (layer::PrallelTransportAttention_Layer)(x::WF3x1)

    # QW=layer.Q.W
    KW=layer.K.W
    VW=layer.V.W

    lW=length(KW)
    m=layer.nominator
    b=layer.bias

    Kx=KW*layer.K.hop(x)
    # Qx=QW*layer.Q.hop(x)
    Hx=layer.V.hop(x)
    M=similar(KW)
    S=similar(KW)
    R=similar(KW)


    for i in 1:lW
        M[i]=Matmul!(conj(Hx[1]),Kx[i])/m
        for a in 1:4
            R[i][a,a]=abs(M[i][a,a])
        end

    end 

    ratio = zeros(Float64, 4)           # ✅



    for u in 1:lW
        for a in 1:4
            ratio[a]+=R[u][a,a]
        end
    end

    for i in 1:lW
        for a in 1:4
            val = max(real(R[i][a,a] / ratio[a] - b),0)
            S[i][a,a]=val*M[i][a,a]
        end
    end

 

    # println("ratio: ", ratio[2])

    W=S*VW
    z=similar(x)




    temp= similar(x)

    for i in 1:length(KW)
        mul!(temp,W[i],Hx[i])
        axpby!(1,temp,1,z)
    end

    if layer.residual == true 
        # z=z+x
        axpby!(1,x,1,z)
    end



    return z
end 

function (layer::siteweight_Layer)(x::WF3x1)

    QW=layer.Q.W
    KW=layer.K.W
    VW=layer.V.W

    lW=length(KW)
    m=layer.nominator


    Hx=layer.V.hop(x)

    Kx=KW*layer.K.hop(x)
    Qx=QW*layer.Q.hop(x)
    
    Hx=layer.V.hop(x)
    Vx=VW*layer.V.hop(x)

    result = similar(x)
    for p in 1:lW
        a=similar(x)
        S=Vector{Array{ComplexF64, 4}}(undef, lW)
        for q in 1:lW
            S[q] = spincolortrace(conj(Qx[p]), Kx[q]) / m
            # n=norm(vec(S[q]))
            # S[q]=S[q]/n
            a += S[q] * Vx[q]

        end
        result += a
    end 




    if layer.residual == true 
        # result=result+x
        axpby!(1,x,1,result)
    end

    return result
end


function (layer::HoppingRayleighQuotient_Layer)(D::Dirac_operator)


    x=layer.x

    Hx=layer.hop(x)

    y=similar(x)
    for i in 1:length(Hx)
        axpby!(1,Hx[i],1,y)
    end
    u=D*y
    
    rq=inner_product(u,u)/inner_product(y,y)
    
    return rq
end


# function (layer::DenseLinearLayer)(x::Vector{WF3x1CPU}) 
#     Out=layer.Out
#     W=layer.W


#     @assert length(W) == length(x)*Out "W의 길이는 Out이어야 합니다."

#     result = Vector{WF3x1}(undef, Out)

#     for k in 1:length(x)
#         @inbounds for i in 1:Out
#             result[i]=W[i+(k-1)*Out]*x[k]
#             result[i] = layer.activation(result[i])
#         end
#     end

#     return result
    
# end


function (layer::DenseLinearLayer)(x::Vector{WF3x1GPU}) 
    Out = layer.Out
    W   = layer.W

    @assert length(W) == length(x)*Out "W의 길이는 Out이어야 합니다."
    result = Vector{WF3x1}(undef, Out)
    yi = deepcopy(x[1])
    substitute_fermion!(yi, x[1])

    for k in 1:length(x)
        @inbounds for i in 1:Out
                # yi용 버퍼 1개 생성
            mul!(yi, W[i+(k-1)*Out], x[k])            # yi = W[i]*x 를 "할당 없이" 계산
            result[i] += layer.activation(yi)  # activation이 새로 만들면 여기서 또 alloc
        end
    end 
    return result
end


function (layer::DenseLinearLayer)(x::AbstractVector{T}) where {T<:WF3x1}
    Out = layer.Out
    W   = layer.W

    @assert length(W) == length(x)*Out "W의 길이는 Out이어야 합니다."
    result = Vector{WF3x1}(undef, Out)
    for i in 1:Out
        result[i] = deepcopy(x[1])
        zero_wf!(result[i])
    end
    yi = deepcopy(x[1])

    for k in 1:length(x)
        @inbounds for i in 1:Out
            mul!(yi, W[i+(k-1)*Out], x[k])
            result[i] += layer.activation(yi)
        end
    end 
    return result
    
end





# function (layer::DenseLinearLayer)(x::WF3x1) 
#     Out = layer.Out
#     W   = layer.W

#     result = Vector{WF3x1}(undef, Out)
#     yi = deepcopy(x)   

    
#     @inbounds for i in 1:Out
#            # yi용 버퍼 1개 생성
#         mul!(yi, W[i], x)            # yi = W[i]*x 를 "할당 없이" 계산
#         result[i] = layer.activation(yi)  # activation이 새로 만들면 여기서 또 alloc
#     end

#     return result
# end


using Base.Threads

function (layer::PTLayer)(x::AbstractVector{T}) where {T<:WF3x1}
    Out   = layer.Out
    W     = layer.W
    paths = layer.paths

    npaths = length(paths)

    @assert npaths == length(x) "입력 x의 길이는 paths의 길이와 같아야 합니다."
    @assert length(W) == Out * npaths "W의 길이는 Out*paths이어야 합니다."

    # 1) 캐시 버퍼를 한 번만 할당, 이후 재사용
    if !layer._initialized || length(layer._hopped) != npaths || length(layer._result) != Out
        layer._hopped = Vector{WF3x1}(undef, npaths)
        layer._result = Vector{WF3x1}(undef, Out)
        for i in 1:npaths
            layer._hopped[i] = deepcopy(x[1])
        end
        for j in 1:Out
            layer._result[j] = deepcopy(x[1])
        end
        layer._initialized = true
    end

    hopped = layer._hopped
    result = layer._result

    # zero 초기화 (deepcopy 대신)
    @inbounds for j in 1:Out
        zero_wf!(result[j])
    end

    # 2) hop 연산
    if x[1] isa WF3x1CPU
        hop = Hopping_operator(layer.U, paths[1])
        hop_buf = WF3x1CPU[hopped[1]]  # 1-element buffer 재사용
        @inbounds for i in 1:npaths
            hop.P = paths[i]
            hop_buf[1] = hopped[i]
            hop(x[i], hop_buf)
        end
    else
        hop = Hopping_operator(layer.U, paths[1])
        hop_buf = WF3x1GPU[hopped[1]]  # GPU도 preallocated 버전 사용
        @inbounds for i in 1:npaths
            hop.P = paths[i]
            hop_buf[1] = hopped[i]
            hop(x[i], hop_buf)
        end
    end

    # 3) Out 방향으로 병렬화
    if x[1] isa WF3x1CPU
        @threads for j in 1:Out
            rj = result[j]
            @inbounds for i in 1:npaths
                w = W[(i-1)*Out + j]
                addmul!(rj, w, hopped[i])
            end
            result[j] = rj
        end
    else
        for j in 1:Out
            rj = result[j]
            @inbounds for i in 1:npaths
                gpuaddmul!(rj, W[(i-1)*Out + j], hopped[i])
            end
            result[j] = rj
        end
    end

    return result
end

function (layer::fermionfield_Layer)(x::Any)
    return layer.x
end

"""
    refresh_fermionfield!(layer::fermionfield_Layer, U)

JLD2 로드 후 stale CuArray를 가진 fermion field를 현재 GPU context로 재생성.
substitute_fermion!으로 데이터를 복사합니다.
"""
function refresh_fermionfield!(layer::fermionfield_Layer, U)
    for i in eachindex(layer.x)
        # stale CuArray는 읽을 수 없으므로 새로 생성 + 초기화
        new_x = Initialize_pseudofermion_fields(U[1], "Wilson", nowing=true)
        substitute_fermion!(new_x, layer.x[i])
        layer.x[i] = new_x
    end
    return layer
end

"""
    refresh_model!(layers, D_new, U_new)

JLD2로 로드한 모델의 stale CuArray를 현재 GPU context의 새 객체로 교체.
- Ritz_Layer, PtcRayleighQuotient_Layer: D 교체
- PTLayer: U 교체
- fermionfield_Layer: x 재생성
- GramSchmidt_Layer: x 재생성
- LossSumLayer: 재귀적으로 처리
"""
function refresh_model!(layers::Vector{<:Layer}, D_new, U_new)
    for layer in layers
        _refresh_layer!(layer, D_new, U_new)
    end
    return layers
end

_refresh_layer!(layer::Ritz_Layer, D_new, U_new) = (layer.D = D_new)
_refresh_layer!(layer::PtcRayleighQuotient_Layer, D_new, U_new) = (layer.D = D_new)
_refresh_layer!(layer::PTLayer, D_new, U_new) = (layer.U = U_new)
_refresh_layer!(layer::fermionfield_Layer, D_new, U_new) = refresh_fermionfield!(layer, U_new)

function _refresh_layer!(layer::GramSchmidt_Layer, D_new, U_new)
    for i in eachindex(layer.x)
        new_x = Initialize_pseudofermion_fields(U_new[1], "Wilson", nowing=true)
        gauss_distribution_fermion!(new_x)
        layer.x[i] = new_x
    end
end

function _refresh_layer!(layer::LossSumLayer, D_new, U_new)
    _refresh_layer!(layer.l1, D_new, U_new)
    _refresh_layer!(layer.l2, D_new, U_new)
end

_refresh_layer!(layer::Layer, D_new, U_new) = nothing









function (layer::NormLayer)(x::Vector{WF3x1})

    result = Vector{WF3x1}(undef, length(x))
    for i in 1:length(x)
        result[i] = layer.alpha*(x[i]/sqrt(norm2(x[i])))
    end 

    return result
end

function (layer::complexNormLayer)(x::WF3x1)


    m=mean(x)
    x=x-m

    return x
end

function (layer::zscoreNormLayer)(x::WF3x1)
    x=standardization(x)

    return x
end 

function (layer::minmaxNormLayer)(x::WF3x1)
    
    
    min,max=minmax(x)

    x=x-min
    x=x/(max-min)

    return x
end












function (layer::RayleighQuotient_Layer)(D::Dirac_operator)
    x=layer.x
    y=D*x
    rq=norm2(y)/norm2(x)

    return rq
end


function (layer::PtcRayleighQuotient_Layer)(x::Vector{WF3x1})

    D=layer.D
    rq=0.0
    Dx=similar(x[1])

    for i in 1:length(x)

    

        mul!(Dx,D,x[i])

    # Dx=D*x[1]
        rq+=inner_product(Dx,Dx)/inner_product(x[i],x[i])

    end 
    return rq
end

# function (layer::Ritz_Layer)(x::AbstractVector{<:WF3x1})

#     l=length(x)
#     # V=x
#     A= Matrix{ComplexF64}(undef, l, l)
#     B= Matrix{ComplexF64}(undef, l, l)
#     D=layer.D
#     # DDAGD를 쓸수가 없음 아직 LatticeDiracOperators에 구현이 안되어있음


#     for i in 1:l
#         for j in 1:l
#             y=similar(x[j])
#             mul!(y,D,x[j])
#             u=similar(y)
#             mul!(u,D,x[i])
#             A[i,j]=inner_product(u,y)
#             B[i,j]=inner_product(x[i],x[j])
#         end
#     end
#     # println(A)
#     # println(B)

#     evals, evecs = eigen(A, B)

#     lambda=0
#     for i in 1:length(evals)
#         lambda+=real(evals[i])
#     end

#     return lambda
#     # return A,B
# end
using LinearAlgebra
using Base.Threads

function (layer::Ritz_Layer)(x::AbstractVector{<:WF3x1})
    l = length(x)


    # 1) Dx[i] = D * x[i] 를 i당 1번만 계산 + 캐시 (DeltaRule!에서 재사용)
    need_reinit = !layer._ritz_init || length(layer._Dz) != l
    if !need_reinit
        # 타입이 바뀌었으면 (GPU↔CPU 전환 등) 캐시 무효화
        need_reinit = typeof(layer._Dz[1]) != typeof(x[1])
    end
    if need_reinit
        layer._Dz = Vector{Any}(undef, l)
        for i in 1:l
            layer._Dz[i] = deepcopy(x[1])
        end
        layer._ritz_init = true
    end

    Dx = layer._Dz
    @inbounds for i in 1:l
        mul!(Dx[i], layer.D, x[i])   # mul! overwrites Dx[i]
    end

    # 2) Gram matrices: A_ij=<Dx_i, Dx_j>, B_ij=<x_i, x_j>
    A = Matrix{ComplexF64}(undef, l, l)
    B = Matrix{ComplexF64}(undef, l, l)

    # CPU면 threads로 outer i를 병렬화 가능 (GPU x일 때는 inner_product가 sync를 많이 걸 수 있어 비추)
    @inbounds for i in 1:l
        # diagonal
        bii = inner_product(x[i],  x[i])
        aii = inner_product(Dx[i], Dx[i])
        B[i,i] = bii
        A[i,i] = aii

        # upper triangle만 계산하고 Hermitian으로 채우기
        for j in i+1:l
            bij = inner_product(x[i],  x[j])
            aij = inner_product(Dx[i], Dx[j])

            B[i,j] = bij
            B[j,i] = conj(bij)

            A[i,j] = aij
            A[j,i] = conj(aij)
        end
    end

    # 3) generalized eigen (Hermitian으로 감싸면 내부 알고리즘이 더 맞고 조금 더 빠름)
    F = eigen(Hermitian(A), Hermitian(B))

    return real(sum(F.values))
end




function (layer::Orthogonal_Layer)(x::Vector{WF3x1})

    l=length(x)
    cl=(l-1)*l/2
    a=layer.coeff

    result=0.0
    
    for i in 2:l
        for j in 1:i-1
            c=inner_product(x[j],x[i])
            result+=c'*c
        end
    end 


    return a*result
end 


function (layer::GramSchmidt_Layer)(x::Vector{WF3x1})
    V = layer.x                 # V[j] 들은 서로 직교라고 가정
    l = length(x)

    result = Vector{WF3x1}(undef, l)

    @inbounds for i in 1:l
        y = deepcopy(x[i])      # y = x_i
        for j in 1:length(V)
            denom = inner_product(V[j], V[j])
            coeff = inner_product(V[j], y) / denom
            axpby!(-coeff, V[j], 1, y)   # y -= coeff * V[j]
        end
        result[i] = y
    end

    return result
end




# function (layer::L2Norm_Layer)(x::Any)

#     if x isa Vector{WF3x1}

#         return norm2(x[1]-layer.target)
#     else 
#         return norm2(x - layer.target)
#     end
# end


function (L::LossSumLayer)(x::Vector{WF3x1})
    return L.w1 * L.l1(x) + L.w2 * L.l2(x)
end


function Base.:+(a::A, b::B) where {A<:Layer, B<:Layer}
    return LossSumLayer(a, b)
end


function (layers::Vector{T})(x::Any) where T<:Layer
    result = x
    for layer in layers
        result = layer(result)
    end
    return result
end


function (layer::BlockScalar_Layer)(x::Vector{WF3x1})
    return [_blockscalar_forward(layer, xi) for xi in x]
end

function (layer::BlockScalar_Layer)(x::WF3x1)
    return _blockscalar_forward(layer, x)
end

function _blockscalar_forward(layer::BlockScalar_Layer, x::WF3x1)
    blocks = blockize(x, layer.r)
    @assert length(blocks) == length(layer.W) "블록 수($(length(blocks)))와 가중치 수($(length(layer.W)))가 일치해야 합니다."

    for i in eachindex(blocks)
        axpby!(layer.W[i], blocks[i], 0, blocks[i])   # blocks[i] = W[i] * blocks[i]
    end

    result = unblockize(blocks)

    result = layer.activation(result)

    if layer.residual
        axpby!(1, x, 1, result)   # result += x
    end

    return result
end











function forward_propagation(layers::Vector{T}, input::Any, target::Any) where {T<:Layer}
    L = length(layers)

    # u는 "마지막 레이어 입력까지" 담습니다. 즉 길이 = L
    # u[1] = 입력, u[2] = 1번째 레이어 출력, ..., u[L] = (L-1)번째 레이어 출력


    
    let last = layers[L]
        if hasfield(typeof(last), :target)   # 또는: hasproperty(last, :target)
            # last가 mutable이면 아래 대입이 됩니다.
            last.target = target             # (= setproperty!(last, :target, target))
        end
    end

    u = Vector{Any}(undef, L)
    u[1] = [input]

    # 1..L-1까지는 WF3x1 → WF3x1
    for i in 1:(L-1)
        u[i+1] = layers[i](u[i])
    end

    # 마지막 레이어는 WF3x1 → Float64
    f = layers[L](u[L])  # 스칼라

    return u, f
end

# Inplace 버전 - 미리 할당된 u 재사용
function forward_propagation!(layers::Vector{T}, input::Any, target::Any, u::Vector{Any}) where {T<:Layer}
    L = length(layers)
    @assert length(u) == L "u의 길이는 layers 길이와 같아야 합니다."

    let last = layers[L]
        if hasfield(typeof(last), :target)
            last.target = target
        end
    end

    u[1] = [input]

    for i in 1:(L-1)
        u[i+1] = layers[i](u[i])
    end

    f = layers[L](u[L])

    return f
end


export backward_propagation




function backward_propagation(layers::Vector{T},  u::Vector{Any}, batch::Int) where T<:Layer
    @assert length(u) == length(layers) 
    @assert all(isassigned(u, j) for j in 1:length(u))  # u가 전부 채워졌는지 안전 확인(선택)

    delta = Vector{Any}(undef, length(layers)-1)


    # 출력층 델타: u[end] = u[L+1] (마지막 레이어 출력)
    delta[1] = DeltaRule(layers[end], layers[end].activation, u[end], batch)

    if length(layers) > 2
        for i in 1:length(layers)-2
            # l = L - i  에 해당. 활성화는 반드시 u[l+1]을 써야 함!
            delta[1+i] = DeltaRule(
                layers[length(layers) - i],
                layers[length(layers) - i].activation,
                u[length(layers) - i ],   # ← 여기 +1 이 핵심 수정
                delta[i]
            )
        end
    end

    @assert length(delta) == length(layers) - 1

    return delta
end

function backward_propagation!(layers::Vector{T}, u::Vector{Any}, batch::Int, delta::Vector{Any}) where T<:Layer
    L = length(layers)

    @assert length(u) == L
    @assert length(delta) == L-1

    # delta[k] = ∂f/∂u[k+1]

    # 마지막 레이어: u[L] -> scalar
    DeltaRule!(layers[L], layers[L].activation, u[L], delta[L-1], batch)

    # hidden layers: j = L-1, ..., 2
    for j in (L-1):-1:2
        DeltaRule!(
            layers[j],
            layers[j].activation,
            u[j],          # <- 여기 중요
            delta[j],      # = ∂f/∂u[j+1]
            delta[j-1]     # = ∂f/∂u[j]
        )
    end

end

using Plots
using Base.Iterators
using DisplayAs
using PyPlot




function train(opt::Opt, layers::Vector{T}, input::WF3x1, target::WF3x1 , Dirac_operator ; epochs=100, batch=1,precond=nothing , def=false) where {T<:Layer}
    depth = length(layers)
    loss_history = Float64[]  # 손실 값 저장할 벡터

    # Jupyter에서 그래프를 실시간으로 업데이트하기 위한 설정
    fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    line, = ax.plot([], [], "b-", linewidth=2)  # 초기 그래프

    # display(fig)  # Jupyter에서 그래프 창 한 번만 표시


    function cosine_annealing_lr(epoch, α_max, α_min, T_max)
        t = epoch % T_max
        return α_min + 0.5 * (α_max - α_min) * (1 + cos(π * t / T_max))
    end


    M=precond


    γ=Clifford()


    D = Dirac_operator

    # 재사용 버퍼 (similar 대신 deepcopy)
    y = deepcopy(input)
    x = deepcopy(input)

    # 메모리 최적화: epoch 밖에서 버퍼 미리 할당
    u = Vector{Any}(undef, depth)
    delta = Vector{Any}(undef, depth-1)
    d = Vector{Any}(undef, depth)
    temp = Vector{ComplexF64}(undef, batch)

    for epoch in 1:epochs

        current_loss = 0.0
        # for i in 1:depth
        #     if layers[i].W === nothing
        #         d[i]=nothing
        #         continue
        #     end
        #     d[i]= similar(layers[i].W)
        # end 

        for b in 1:batch

            # g2vector_generator!(input)
            gauss_distribution_fermion!(input)

            # target=deepcopy(input)/norm(input)

            
            #deflation
            if def== true 
                k=similar(input)
                gmres(input,D,k,maxsteps=1,restart=5)
            end

            if D isa Vector{T}
                y=D(input)
            else 
                mul!(y,D,input)
            end
            # mul!(x,D,y)


            #left
            den=norm(y) 

            #Right
            # den=norm(input)

            # target=input
            target=(input)/den
            if M != nothing
                target=target - M(y/den)
            end
            #left
            input=y/den 
            # input=(y)/den
            # input=x

            # # #Right
            # input=input/den
            
            # input=y-input
            # target=similar(input)


            f = forward_propagation!(layers, input, target, u)
            temp[b] = f

            backward_propagation!(layers, u, batch, delta)

            for i in 1:depth-1
                if b == 1
                    d[i] = gradient(layers[i], layers[i].activation::Any, u[i], delta[i], γ)
                else
                    # batch > 1일 때 누적 (향후 구현)
                    # d[i] = add_gradient(d[i], gradient(...))
                end
            end





      

        end

        # Gradient 계산, grad에 모든 층의 gradient 저장
         @time for i in 1:depth-1
            # d = gradient(layers[i], layers[i].activation::Any, u[i], delta[depth - i + 1],γ)
            
            # if d === nothing
            #     continue  # NormLayer 등은 업데이트 생략
            # end
        

        # Optimizer로 가중치 업데이트
            update!(layers[i], opt, d[i])

        end

        # 현재 손실 계산 후 저장
        current_loss = 0.0
        for b in 1:batch
            current_loss += temp[b]
        end
        push!(real(loss_history), real(current_loss))



        # 기존 그래프를 지우고 다시 그리기
        
        println("Epoch: $epoch, Loss: $current_loss")  # 로그 출력



    end
    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end







function train(opt::Opt, layers::Vector{T} , config ,params ; epochs=100, batch=1 ) where {T<:Layer}

    depth = length(layers)

    loss_history = Float64[]  # 손실 값 저장할 벡터


        fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    line, = ax.plot([], [], "b-", linewidth=2)  # 초기 그래프
    γ=Clifford()

    x = Initialize_pseudofermion_fields(config[1][1],"Wilson")

    # config 수만큼 Dirac operator를 미리 생성 (매 epoch 재생성 방지)
    Ds = Vector{Any}(undef, length(config))
    for k in 1:length(config)
        Ds[k] = Dirac_operator(config[k], x, params)
    end

    D = Ds[1]
    dummy ,f = forward_propagation(layers, D , D)

    delta=deepcopy(dummy[2:end])  # delta는 레이어 수보다 하나 적음에 주의
    u = dummy  # u 버퍼 재사용 (매 epoch 재할당 방지)

    for epoch in 1:epochs
        k=mod(epoch-1,length(config))+1
        D = Ds[k]
        for i in 1:depth
            if layers[i] isa PTLayer
                layers[i].U= config[k]
            end 
        end
        if layers[end] isa PtcRayleighQuotient_Layer
            layers[end].D=D
        elseif  layers[end] isa LossSumLayer
            if layers[end].l1 isa PtcRayleighQuotient_Layer
                layers[end].l1.D=D
            end
            if layers[end].l2 isa PtcRayleighQuotient_Layer
                layers[end].l2.D=D
            end
        end
        if layers[end] isa Ritz_Layer
            layers[end].D=D
        end
        temp=Vector{ComplexF64}(undef, batch)
        d=Vector{Any}(undef, depth)
        fill!(d, nothing)
        for b in 1:batch
             f = forward_propagation!(layers, D , D, u)
            temp[b]=f
            backward_propagation!(layers, u, batch, delta)



            for i in 1:depth-1
                gi = gradient(layers[i], layers[i].activation, u[i], delta[i], γ)

                gi === nothing && continue

                if d[i] === nothing
                    d[i] = gi
                else
                    d[i] += gi
                end
            end
        end


        # Gradient 계산, grad에 모든 층의 gradient 저장
         for i in 1:depth-1
        # Optimizer로 가중치 업데이트
            d[i] === nothing && continue
           update!(layers[i], opt, d[i])
        end

        # 현재 손실 계산 후 저장
        current_loss = 0.0
        for b in 1:batch
            current_loss += temp[b]
        end
        push!(real(loss_history), real(current_loss))



        # 기존 그래프를 지우고 다시 그리기

        println("Epoch: $(epoch), Loss: $current_loss")  # 로그 출력
    end


    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end








end # module Layer_sequence