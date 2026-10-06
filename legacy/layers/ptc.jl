# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

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

_has_gpu(layer::PTC_Layer{Spin_matrix_GPU}) = true

_has_gpu(layer::PTC_Layer) = eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator

# --- PTC_Layer ---
function _to_cpu(layer::PTC_Layer)
    hop_cpu = _to_cpu_hop(layer.hop)
    W_cpu = _to_cpu_W(layer.W)
    return PTC_Layer(hop_cpu, W_cpu, layer.residual, layer.activation)
end

# --- PTC_Layer ---
function _to_gpu(layer::PTC_Layer, blocks=nothing)
    hop_gpu = _to_gpu_hop(layer.hop; blocks=blocks)
    W_gpu = _to_gpu_W(layer.W)
    return PTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

function _to_gpu(layer::PTC_Layer, blocks, shared_U)
    hop_gpu = _to_gpu_hop(layer.hop; blocks=blocks, shared_U=shared_U)
    W_gpu = _to_gpu_W(layer.W)
    return PTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

function (layer::PTC_Layer)(x::WF3x1)
    result = similar(x)
    fields = layer.hop(x) # Hopping operator field
    c=similar(x)

    if length(layer.W) == 1
        # Use a shared weight for all hopped fields.
        for i in 1:length(fields)
            mul!(c, layer.W[1], fields[i])
            axpby!(1,c,1,result)
        end
    elseif length(layer.W) == length(fields)
        # Use one weight per hopped field.
        for i in 1:length(fields)
            mul!(c, layer.W[i], fields[i])
            axpby!(1,c,1,result)
        end
    else
        error("W length must be 1 or match the number of hopped fields.")
    end


    result = layer.activation(result)

    if layer.residual == true
        # result = result + x
        axpby!(1,x,1,result)
    end


    return result
end

function gradient(layer::PTC_Layer,::Identity, z::WF3x1, d::WF3x1,γ::Vector{Spin_matrix} )

    if layer.W === nothing
        error("gradient() 호출 시 layer.W가 정의되지 않았습니다.")
    end

    hop = layer.hop
    if hop === nothing
        error("gradient() 호출 시 layer.hop이 정의되지 않았습니다.")
    end

    W = layer.W
    lW = length(W)

    s = hop(z)
    grad = deepcopy(W)

    for i in 1:lW
        grad[i]=Matmul!(d,s[i])
    end 

    return grad



end

function DeltaRule(layer::PTC_Layer, ::Identity,z::WF3x1, d::WF3x1)


    hop = layer.hop
    cW = adjoint(layer.W)
    lW = length(cW)


    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)
    cd = conj(d) # 여긴 문제가 없음
    cL = PTC_Layer(invhop,cW)

    result = cL(cd) # 여긴 문제가 없음
    result = conj(result) # 여긴 문제가 없음 

    
    if layer.residual == true
        # result=result+d
        axpby!(1,d,1,result)
    end


    return result
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
