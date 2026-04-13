module Layer_sequence





import ..Layer_types: Layer, CPTC_Layer, PTC_Layer,ActivationFunction,Identity,Attention_Layer,Softmax,NormLayer,test_Layer,complexNormLayer,minmaxNormLayer,zscoreNormLayer,RealAttention_Layer
import ..Layer_types: FullPTCAttention_Layer,Dirac_Layer,PrallelTransportAttention_Layer,siteweight_Layer,RayleighQuotient_Layer,PtcRayleighQuotient_Layer,HoppingRayleighQuotient_Layer
import ..Layer_types: Orthogonal_Layer,LossSumLayer,Ritz_Layer
import ..elementry_base: Hopping_operator, Spin_matrix, inner_product,g2vector_generator!,addmul!
import ..elementry_base: Clifford, Random_SpinM,norm2,mean,standardization,minmax,spincolortrace
import ..Optimizer: Opt, update!, Adam
import ..activation:  gradient, DeltaRule,DeltaRule!
import ..gpu: Matmul!,gpuaddmul!,zero_wf!,Convertsm,Spin_matrix_GPU,_build_block_weight_gpu,_init_gpu_block_cache!,_update_block_weight_gpu!,Matmul_batch!
import ..gpu: sync_W_to_gpu!,pack_fermions!,unpack_fermions!,batched_forward!,fused_hopping_gpu!,fused_wilson_mul_gpu!,batched_hopping_steps_gpu!,get_shared_buf!
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
        # GPU → CPU 전 blocks 정보 추출 (CPU→GPU 재변환 시 필요)
        _gpu_blocks = _extract_gpu_blocks(layers)
        for i in eachindex(layers)
            layers[i] = _to_cpu(layers[i])
        end
        # blocks 정보를 _saved_blocks에 저장
        if _gpu_blocks !== nothing
            _saved_blocks[] = _gpu_blocks
        end
    else
        # CPU → GPU: 이전에 저장한 blocks 정보 사용
        blks = _saved_blocks[]
        for i in eachindex(layers)
            layers[i] = _to_gpu(layers[i], blks)
        end
    end
    return layers
end

# 모듈 레벨에서 blocks 정보 저장
const _saved_blocks = Ref{Any}(nothing)

# GPU gauge에서 blocks 정보 추출
function _extract_gpu_blocks(layers::Vector{<:Layer})
    for layer in layers
        blks = _get_blocks(layer)
        if blks !== nothing
            return blks
        end
    end
    return nothing
end

_get_blocks(::Layer) = nothing
function _get_blocks(layer::PTLayer)
    if eltype(layer.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        bi = layer.U[1].blockinfo
        return bi.blocks
    end
    return nothing
end
function _get_blocks(layer::Ritz_Layer)
    try
        U = layer.D.U
        if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
            return U[1].blockinfo.blocks
        end
    catch; end
    return nothing
end
function _get_blocks(layer::HoppingRayleighQuotient_Layer)
    if eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return layer.hop.Gauge[1].blockinfo.blocks
    end
    return nothing
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

# --- BlockScalar_Layer: GPU 캐시 클리어 ---
function _to_cpu(layer::BlockScalar_Layer)
    layer._gpu_w_buf = nothing
    layer._gpu_w_cpu = Matrix{ComplexF64}(undef, 0, 0)
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._initialized = false
    return layer
end

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
    # GPU↔CPU 전환 시 캐시 무효화
    layer._ritz_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._VD = Any[]
    layer._work = Any[]
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
function _to_gpu_gauge(U::Vector{G}; blocks=nothing) where {G<:AbstractGaugefields}
    if G <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return U  # 이미 GPU
    end
    NC = U[1].NC
    NX = U[1].NX; NY = U[1].NY; NZ = U[1].NZ; NT = U[1].NT
    if blocks !== nothing
        Ugpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true, blocks=blocks)
    else
        Ugpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    end
    substitute_U!(Ugpu, U)
    return Ugpu
end

# --- CPU→GPU Hopping_operator 변환 ---
function _to_gpu_hop(hop::Hopping_operator; blocks=nothing)
    Ugpu = _to_gpu_gauge(hop.Gauge; blocks=blocks)
    if Ugpu === hop.Gauge
        return hop
    end
    return Hopping_operator(Ugpu, hop.P)
end

# --- CPU→GPU Spin matrix 벡터 변환 ---
_to_gpu_W(W::Vector{Spin_matrix_GPU}) = W
_to_gpu_W(W::Vector{Spin_matrix}) = [Spin_matrix_GPU(w) for w in W]

# --- CPU→GPU fermion field 변환 ---
function _to_gpu_fermion(x::WF3x1CPU; blocks=nothing)
    NC = x.NC
    NX = x.NX; NY = x.NY; NZ = x.NZ; NT = x.NT
    if blocks !== nothing
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true, blocks=blocks)
    else
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    end
    xgpu = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
    substitute_fermion!(xgpu, x)
    return xgpu
end
_to_gpu_fermion(x::WF3x1GPU; blocks=nothing) = x

function _to_gpu_fermions(xs::Vector{<:WF3x1}; blocks=nothing)
    if isempty(xs) || xs[1] isa WF3x1GPU
        return xs
    end
    NC = xs[1].NC
    NX = xs[1].NX; NY = xs[1].NY; NZ = xs[1].NZ; NT = xs[1].NT
    if blocks !== nothing
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true, blocks=blocks)
    else
        Utmp = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT, condition="cold", cuda=true)
    end
    result = Vector{WF3x1GPU}(undef, length(xs))
    for i in eachindex(xs)
        result[i] = Initialize_pseudofermion_fields(Utmp[1], "Wilson", nowing=true)
        substitute_fermion!(result[i], xs[i])
    end
    return result
end

# --- Dirac operator CPU→GPU 변환 ---
function _to_gpu_dirac(D; blocks=nothing)
    U = D.U
    if eltype(U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator
        return D  # 이미 GPU
    end
    Ugpu = _to_gpu_gauge(U; blocks=blocks)
    xgpu = Initialize_pseudofermion_fields(Ugpu[1], "Wilson", nowing=true)
    params = Dict{String,Any}()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = D.κ
    params["eps_CG"] = D.eps_CG
    params["faster version"] = isa(D, LatticeDiracOperators.Dirac_operators.Wilson_Dirac_operator_faster)
    return Dirac_operator(Ugpu, xgpu, params)
end

# --- 기본: CPU 전용 성분만 있는 Layer ---
_to_gpu(layer::Layer, blocks=nothing) = layer

# --- BlockScalar_Layer: 캐시 리셋 (GPU 초기화는 forward에서 자동) ---
function _to_gpu(layer::BlockScalar_Layer, blocks=nothing)
    layer._gpu_w_buf = nothing
    layer._gpu_w_cpu = Matrix{ComplexF64}(undef, 0, 0)
    layer._gpu_block_map = Int[]
    layer._result = WF3x1[]
    layer._block_map = Int[]
    layer._initialized = false
    return layer
end

# --- PTC_Layer ---
function _to_gpu(layer::PTC_Layer, blocks=nothing)
    hop_gpu = _to_gpu_hop(layer.hop; blocks=blocks)
    W_gpu = _to_gpu_W(layer.W)
    return PTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

# --- CPTC_Layer ---
function _to_gpu(layer::CPTC_Layer, blocks=nothing)
    hop_gpu = _to_gpu_hop(layer.hop; blocks=blocks)
    W_gpu = _to_gpu_W(layer.W)
    return CPTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

# --- Attention_Layer ---
function _to_gpu(layer::Attention_Layer, blocks=nothing)
    return Attention_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                           layer.nominator, layer.activation, layer.residual)
end

# --- test_Layer ---
function _to_gpu(layer::test_Layer, blocks=nothing)
    layer.Q = _to_gpu(layer.Q, blocks)
    layer.K = _to_gpu(layer.K, blocks)
    layer.V = _to_gpu(layer.V, blocks)
    if layer.test isa WF3x1CPU
        layer.test = _to_gpu_fermion(layer.test; blocks=blocks)
    end
    return layer
end

# --- FullPTCAttention_Layer ---
function _to_gpu(layer::FullPTCAttention_Layer, blocks=nothing)
    return FullPTCAttention_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                                  layer.nominator, layer.activation, layer.residual)
end

# --- PrallelTransportAttention_Layer ---
function _to_gpu(layer::PrallelTransportAttention_Layer, blocks=nothing)
    return PrallelTransportAttention_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                                            layer.nominator, layer.activation, layer.residual, layer.bias)
end

# --- siteweight_Layer ---
function _to_gpu(layer::siteweight_Layer, blocks=nothing)
    return siteweight_Layer(_to_gpu(layer.Q, blocks), _to_gpu(layer.K, blocks), _to_gpu(layer.V, blocks),
                            layer.nominator, layer.activation, layer.residual)
end

# --- DenseLinearLayer ---
function _to_gpu(layer::DenseLinearLayer{Spin_matrix}, blocks=nothing)
    new_layer = DenseLinearLayer(layer.In, layer.Out, true, layer.activation)
    W_gpu = [Spin_matrix_GPU(w) for w in layer.W]
    for j in eachindex(W_gpu)
        new_layer.W[j] = W_gpu[j]
    end
    return new_layer
end

# --- PTLayer ---
function _to_gpu(layer::PTLayer, blocks=nothing)
    U_gpu = _to_gpu_gauge(layer.U; blocks=blocks)
    new_layer = PTLayer(layer.paths, U_gpu, layer.Out, layer.activation)
    W_gpu = _to_gpu_W(layer.W)
    new_layer.W = W_gpu
    return new_layer
end

# --- fermionfield_Layer ---
function _to_gpu(layer::fermionfield_Layer, blocks=nothing)
    layer.x = _to_gpu_fermions(layer.x; blocks=blocks)
    return layer
end

# --- GramSchmidt_Layer ---
function _to_gpu(layer::GramSchmidt_Layer, blocks=nothing)
    layer.x = _to_gpu_fermions(layer.x; blocks=blocks)
    return layer
end

# --- HoppingRayleighQuotient_Layer ---
function _to_gpu(layer::HoppingRayleighQuotient_Layer, blocks=nothing)
    layer.hop = _to_gpu_hop(layer.hop; blocks=blocks)
    if layer.x isa WF3x1CPU
        layer.x = _to_gpu_fermion(layer.x; blocks=blocks)
    end
    return layer
end

# --- Ritz_Layer ---
function _to_gpu(layer::Ritz_Layer, blocks=nothing)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks)
    # GPU↔CPU 전환 시 캐시 무효화
    layer._ritz_init = false
    layer._Dz = Any[]
    layer._DdagDz = Any[]
    layer._VD = Any[]
    layer._work = Any[]
    return layer
end

# --- PtcRayleighQuotient_Layer ---
function _to_gpu(layer::PtcRayleighQuotient_Layer, blocks=nothing)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks)
    return layer
end

# --- LossSumLayer ---
function _to_gpu(layer::LossSumLayer, blocks=nothing)
    l1_gpu = _to_gpu(layer.l1, blocks)
    l2_gpu = _to_gpu(layer.l2, blocks)
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


function (layer::DenseLinearLayer)(x::AbstractVector{T}) where {T<:WF3x1}
    Out = layer.Out
    In  = layer.In
    W   = layer.W
    NW  = length(W)

    @assert NW == In*Out "W의 길이는 In*Out이어야 합니다."

    # 캐시 버퍼 초기화 (최초 1회만)
    if !layer._dl_init || length(layer._result) != Out
        layer._result = Vector{WF3x1}(undef, Out)
        for i in 1:Out
            layer._result[i] = deepcopy(x[1])
        end
        layer._yi = [deepcopy(x[1])]
        layer._dl_init = true
    end

    result = layer._result

    if x[1] isa WF3x1GPU
        # GPU batched: 50 kernel launches → 1
        NC = 3; NG = 4
        blocksize = x[1].blockinfo.blocksize
        rsize     = x[1].blockinfo.rsize

        if layer._W_gpu === nothing
            layer._W_gpu      = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._input_all  = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, In)
            layer._result_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, Out)
            sync_W_to_gpu!(layer._W_gpu, W)
        end

        pack_fermions!(layer._input_all, x)
        batched_forward!(layer._result_all, layer._W_gpu, layer._input_all, blocksize, rsize, In, Out)
        unpack_fermions!(result, layer._result_all)
    else
        # CPU path
        yi = layer._yi[1]
        @inbounds for j in 1:Out
            zero_wf!(result[j])
        end
        for k in 1:In
            @inbounds for i in 1:Out
                mul!(yi, W[i+(k-1)*Out], x[k])
                axpby!(1, yi, 1, result[i])
            end
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
    _is_gpu_pt = x[1] isa WF3x1GPU
    if !layer._initialized || length(layer._result) != Out
        if !_is_gpu_pt
            # CPU path에서만 _hopped 필요
            layer._hopped = Vector{WF3x1}(undef, npaths)
            for i in 1:npaths
                layer._hopped[i] = deepcopy(x[1])
            end
        end
        layer._result = Vector{WF3x1}(undef, Out)
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

        # 3) CPU: Out 방향 선형결합
        @threads for j in 1:Out
            rj = result[j]
            @inbounds for i in 1:npaths
                w = W[(i-1)*Out + j]
                addmul!(rj, w, hopped[i])
            end
            result[j] = rj
        end
    else
        # GPU batched hop: 57 kernels → max_steps kernels
        NC = 3; NG = 4
        blocksize = x[1].blockinfo.blocksize
        rsize     = x[1].blockinfo.rsize

        # lazy init GPU buffers (W, hopped_all, etc.)
        NW = npaths * Out
        if layer._W_gpu === nothing
            layer._W_gpu      = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._Wadj_gpu   = CUDA.zeros(ComplexF64, NG, NG, NW)
            layer._hopped_all = CUDA.zeros(ComplexF64, NC, NG, blocksize, rsize, npaths)
            sync_W_to_gpu!(layer._W_gpu, W)
        end
        # 공유 임시 버퍼 (layers 간 재사용 — forward 직후 unpack되므로 안전)
        _result_buf = get_shared_buf!(:result_all, NC, NG, blocksize, rsize, Out)
        _hop_temp   = get_shared_buf!(:hop_temp, NC, NG, blocksize, rsize, npaths)

        # lazy init batched hop tables
        if layer._hop_dirs_gpu === nothing
            max_steps = 0
            for i in 1:npaths
                steps_i = paths[i].various[1].steps
                ns = length(steps_i)
                # identity path (single step with direction 0) → 0 effective steps
                if ns == 1 && steps_i[1].f == 0
                    ns = 0
                end
                max_steps = max(max_steps, ns)
            end
            layer._hop_max_steps = max_steps

            dirs_cpu = zeros(Int8, max_steps, npaths)
            nsteps_cpu = zeros(Int32, npaths)
            for i in 1:npaths
                steps_i = paths[i].various[1].steps
                ns = length(steps_i)
                if ns == 1 && steps_i[1].f == 0
                    nsteps_cpu[i] = Int32(0)
                else
                    nsteps_cpu[i] = Int32(ns)
                    for s in 1:ns
                        dirs_cpu[s, i] = Int8(steps_i[s].f)
                    end
                end
            end
            layer._hop_dirs_gpu   = CuArray(dirs_cpu)
            layer._hop_nsteps_gpu = CuArray(nsteps_cpu)
        end

        # Batched hop: pack x → hopped_all, then max_steps batched kernels
        if layer._hop_max_steps > 0
            batched_hopping_steps_gpu!(
                layer._hopped_all, _hop_temp, x,
                layer.U, layer._hop_dirs_gpu, layer._hop_nsteps_gpu,
                layer._hop_max_steps, npaths, blocksize, rsize)
        else
            pack_fermions!(layer._hopped_all, x)
        end

        # 1회 batched forward kernel (공유 버퍼 사용)
        batched_forward!(_result_buf, layer._W_gpu, layer._hopped_all, blocksize, rsize, npaths, Out)

        # unpack result_all → result
        unpack_fermions!(result, _result_buf)
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









function (layer::NormLayer)(x::AbstractVector{<:WF3x1})
    n = length(x)
    # 캐시 존재하면 재사용 (GPU↔CPU 전환 시 재초기화)
    if !layer._norm_init || length(layer._result) != n || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(x[1]))
        layer._result = [deepcopy(x[i]) for i in 1:n]
        layer._norm_init = true
    end
    result = layer._result
    @inbounds for i in 1:n
        s = layer.alpha / sqrt(norm2(x[i]))
        axpby!(s, x[i], 0, result[i])   # result[i] = s * x[i]
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


function (layer::PtcRayleighQuotient_Layer)(x::AbstractVector{<:WF3x1})

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
        layer._DdagDz = Vector{Any}(undef, l)
        for i in 1:l
            layer._Dz[i] = deepcopy(x[1])
            layer._DdagDz[i] = deepcopy(x[1])
        end
        layer._ritz_init = true
    end

    Dx = layer._Dz
    DdagDx = layer._DdagDz

    # GPU fused path: zero-alloc Wilson kernel
    _is_gpu_ritz = x[1] isa WF3x1GPU
    if _is_gpu_ritz
        _κ = layer.D.κ
        _U = layer.D.U
        @inbounds for i in 1:l
            fused_wilson_mul_gpu!(Dx[i], x[i], _U, _κ, false)       # Dx = D * x
            fused_wilson_mul_gpu!(DdagDx[i], Dx[i], _U, _κ, true)   # DdagDx = D† D x
        end
    else
        @inbounds for i in 1:l
            mul!(Dx[i], layer.D, x[i])       # Dx = D * x
            mul!(DdagDx[i], layer.D', Dx[i])  # DdagDx = D† D x
        end
    end

    # 2) Gram matrices: A_ij=<Dx_i, Dx_j>, B_ij=<x_i, x_j>
    if size(layer._A) != (l, l)
        layer._A = Matrix{ComplexF64}(undef, l, l)
        layer._B = Matrix{ComplexF64}(undef, l, l)
        layer._m = Vector{ComplexF64}(undef, l)
    end
    A = layer._A
    B = layer._B

    @inbounds for i in 1:l
        bii = inner_product(x[i],  x[i])
        aii = inner_product(Dx[i], Dx[i])
        B[i,i] = bii
        A[i,i] = aii

        for j in i+1:l
            bij = inner_product(x[i],  x[j])
            aij = inner_product(Dx[i], Dx[j])

            B[i,j] = bij
            B[j,i] = conj(bij)

            A[i,j] = aij
            A[j,i] = conj(aij)
        end
    end

    # 3) generalized eigen + 캐시 (DeltaRule!에서 재사용)
    F = eigen(Hermitian(A), Hermitian(B))
    layer._evals = F.values
    layer._evecs = F.vectors

    # m_k = u_k† B u_k 미리 계산 (DeltaRule!에서 재사용)
    m = layer._m
    @inbounds for k in 1:l
        s = zero(ComplexF64)
        for i in 1:l, j in 1:l
            s += conj(F.vectors[i,k]) * B[i,j] * F.vectors[j,k]
        end
        m[k] = s
    end

    return real(sum(F.values))
end




function (layer::Orthogonal_Layer)(x::AbstractVector{<:WF3x1})

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


function (layer::GramSchmidt_Layer)(x::AbstractVector{<:WF3x1})
    l = length(x)

    # 버퍼 초기화 (최초 1회 또는 크기 변경 시)
    if length(layer.x) != l
        layer.x = [deepcopy(x[i]) for i in 1:l]
    end
    q = layer.x

    # Modified Gram-Schmidt: 입력 벡터들을 서로 직교화
    copyto!(q[1].f, x[1].f)
    @inbounds for i in 2:l
        copyto!(q[i].f, x[i].f)
        for j in 1:i-1
            denom = inner_product(q[j], q[j])
            coeff = inner_product(q[j], q[i]) / denom
            axpby!(-coeff, q[j], 1, q[i])   # q[i] -= proj_{q[j]}(q[i])
        end
    end

    return q
end




# function (layer::L2Norm_Layer)(x::Any)

#     if x isa Vector{WF3x1}

#         return norm2(x[1]-layer.target)
#     else 
#         return norm2(x - layer.target)
#     end
# end


function (L::LossSumLayer)(x::AbstractVector{<:WF3x1})
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


function (layer::BlockScalar_Layer)(x::AbstractVector{<:WF3x1})
    n = length(x)
    nblocks = size(layer.W, 1)
    r = layer.r

    # ── 캐시 초기화 (1회만, 또는 GPU↔CPU 전환 시 재초기화) ──
    if !layer._initialized || length(layer._result) != n || (!isempty(layer._result) && typeof(layer._result[1]) != typeof(x[1]))
        layer._result = Vector{WF3x1}(undef, n)
        for k in 1:n
            layer._result[k] = deepcopy(x[k])
        end
        if x[1] isa WF3x1GPU
            # GPU용 block map + CuArray 버퍼 캐시
            _init_gpu_block_cache!(layer, x[1])
        else
            # CPU용 block_map: 사이트 좌표 → block_id 매핑 (1회 계산)
            NX, NY, NZ, NT = x[1].NX, x[1].NY, x[1].NZ, x[1].NT
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

    if x[1] isa WF3x1GPU
        for k in 1:n
            w_gpu = _update_block_weight_gpu!(layer, view(layer.W, :, k))
            result[k].f .= w_gpu .* x[k].f
            if layer.residual
                axpby!(1, x[k], 1, result[k])
            end
        end
    else
        # CPU: block_map으로 직접 스칼라 곱 (blockize/unblockize 제거)
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




function train(opt::Opt, layers::Vector{T}, input::WF3x1, target::WF3x1 , Dirac_operator ; epochs=100, batch=1,precond=nothing , def=false, logfile=nothing) where {T<:Layer}
    depth = length(layers)
    loss_history = Float64[]  # 손실 값 저장할 벡터

    # 로그 파일 설정 (logfile 미지정 시 자동 생성)
    if logfile === nothing
        logfile = "train_log_$(Libc.strftime("%Y-%m-%d_%H%M%S", time())).txt"
    end
    log_io = open(logfile, "a")
    println(log_io, "=== Training started: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    println(log_io, "epochs=$epochs, batch=$batch, depth=$depth")
    flush(log_io)

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
        println(log_io, "Epoch: $epoch, Loss: $current_loss")
        flush(log_io)



    end
    println(log_io, "=== Training finished: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    close(log_io)
    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end







function train(opt::Opt, layers::Vector{T} , config ,params ; epochs=100, batch=1, logfile=nothing, test_config=nothing, test_interval=10 ) where {T<:Layer}

    depth = length(layers)

    loss_history = Float64[]  # 손실 값 저장할 벡터

    # 로그 파일 설정 (logfile 미지정 시 자동 생성)
    if logfile === nothing
        logfile = "train_log_$(Libc.strftime("%Y-%m-%d_%H%M%S", time())).txt"
    end
    log_io = open(logfile, "a")
    println(log_io, "=== Training started: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    println(log_io, "epochs=$epochs, batch=$batch, depth=$depth, configs=$(length(config))")
    flush(log_io)

    # test config용 Dirac operator 미리 생성
    test_Ds = nothing
    test_loss_history = Float64[]
    if test_config !== nothing
        test_Ds = Vector{Any}(undef, length(test_config))
        x_test = Initialize_pseudofermion_fields(test_config[1][1], "Wilson")
        for k in 1:length(test_config)
            test_Ds[k] = Dirac_operator(test_config[k], x_test, params)
        end
    end


        fig, ax = plt.subplots()
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    line, = ax.plot([], [], "b-", linewidth=2)  # 초기 그래프
    test_line = nothing
    if test_config !== nothing
        test_line, = ax.plot([], [], "r--", linewidth=2, label="test")
        line.set_label("train")
        ax.legend()
    end
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

    temp=Vector{ComplexF64}(undef, batch)
    d=Vector{Any}(undef, depth)

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
        println(log_io, "Epoch: $(epoch), Loss: $current_loss")
        flush(log_io)

        # Test loss 평가
        if test_config !== nothing && mod(epoch, test_interval) == 0
            test_loss = 0.0
            for tc in 1:length(test_config)
                D_test = test_Ds[tc]
                # test용 config를 layer에 설정
                for i in 1:depth
                    if layers[i] isa PTLayer
                        layers[i].U = test_config[tc]
                    end
                end
                if layers[end] isa PtcRayleighQuotient_Layer
                    layers[end].D = D_test
                elseif layers[end] isa LossSumLayer
                    if layers[end].l1 isa PtcRayleighQuotient_Layer
                        layers[end].l1.D = D_test
                    end
                    if layers[end].l2 isa PtcRayleighQuotient_Layer
                        layers[end].l2.D = D_test
                    end
                end
                if layers[end] isa Ritz_Layer
                    layers[end].D = D_test
                end
                for b in 1:batch
                    test_loss += real(forward_propagation!(layers, D_test, D_test, u))
                end
            end
            test_loss /= length(test_config)
            push!(test_loss_history, real(test_loss))
            println("  [Test] Epoch: $(epoch), Test Loss: $test_loss")
            println(log_io, "  [Test] Epoch: $(epoch), Test Loss: $test_loss")
            flush(log_io)

            # test 그래프 업데이트
            test_line.set_xdata(test_interval:test_interval:(test_interval*length(test_loss_history)))
            test_line.set_ydata(test_loss_history)

            # train config으로 복원
            k_restore = mod(epoch-1, length(config)) + 1
            for i in 1:depth
                if layers[i] isa PTLayer
                    layers[i].U = config[k_restore]
                end
            end
            D_restore = Ds[k_restore]
            if layers[end] isa PtcRayleighQuotient_Layer
                layers[end].D = D_restore
            elseif layers[end] isa LossSumLayer
                if layers[end].l1 isa PtcRayleighQuotient_Layer
                    layers[end].l1.D = D_restore
                end
                if layers[end].l2 isa PtcRayleighQuotient_Layer
                    layers[end].l2.D = D_restore
                end
            end
            if layers[end] isa Ritz_Layer
                layers[end].D = D_restore
            end
        end
    end

    println(log_io, "=== Training finished: $(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ===")
    close(log_io)

    line.set_xdata(1:length(loss_history))
    line.set_ydata(loss_history)
    ax.set_yscale("log")  # 손실이 크게 변할 수 있으므로 로그 스케일로 설정
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


end








end # module Layer_sequence