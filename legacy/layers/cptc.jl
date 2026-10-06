# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

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

_has_gpu(layer::CPTC_Layer{Spin_matrix_GPU}) = true

_has_gpu(layer::CPTC_Layer) = eltype(layer.hop.Gauge) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator

# --- CPTC_Layer ---
function _to_cpu(layer::CPTC_Layer)
    hop_cpu = _to_cpu_hop(layer.hop)
    W_cpu = _to_cpu_W(layer.W)
    return CPTC_Layer(hop_cpu, W_cpu, layer.residual, layer.activation)
end

# --- CPTC_Layer ---
function _to_gpu(layer::CPTC_Layer, blocks=nothing)
    hop_gpu = _to_gpu_hop(layer.hop; blocks=blocks)
    W_gpu = _to_gpu_W(layer.W)
    return CPTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

function _to_gpu(layer::CPTC_Layer, blocks, shared_U)
    hop_gpu = _to_gpu_hop(layer.hop; blocks=blocks, shared_U=shared_U)
    W_gpu = _to_gpu_W(layer.W)
    return CPTC_Layer(hop_gpu, W_gpu, layer.residual, layer.activation)
end

function (layer::CPTC_Layer)(x::WF3x1)

    fields = layer.hop(x) # Hopping operator field
    cW=Clifford(layer.W)
    y=similar(x)
    c=similar(x)



    if length(layer.W) == 1
        # Use a shared Clifford weight for all hopped fields.
        for i in 1:length(fields)
            mul!(c, cW[1], fields[i])
            axpby!(1,c,1,y)
        end
    elseif length(layer.W) == length(fields)
        # Use one Clifford weight per hopped field.
        for i in 1:length(fields)
            mul!(c, cW[i], fields[i])
            axpby!(1,c,1,y)
        end
    else
        error("W length must be 1 or match the number of hopped fields.")
    end


    result =layer.activation(y)

    if layer.residual == true
        # result = result + x
        axpby!(1,x,1,result)
    end

    return result 
end

# using LatticeQCD
  # 활성화 함수 추상 타입











function gradient(layer::CPTC_Layer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})
    if layer.W === nothing
        error("gradient() 호출 시 layer.W가 정의되지 않았습니다.")
    end


    # cd = conj(d)

    lW = length(layer.W) 

    s = layer.hop(z)
    grad = deepcopy(layer.W)

    for i in 1:lW
        for u in 1:4
            for v in 1:4
                grad[i][u, v] = inner_product(conj(d),γ[4*(u-1)+v]*s[i])
            end
        end
    end 

    # println(inner_product(d,z))
    # println(inner_product(cd,z))

    # 이 부분에선 복소수 함수이므로, 라이프니츠법칙에 따라야한다. 

    grad=grad+conj(grad)

    return grad/2
end

function DeltaRule(layer::CPTC_Layer, ::Identity,z::WF3x1, d::WF3x1)
    hop = layer.hop
    W = Clifford(layer.W)
    invP = reverse_paths(hop.P)
    invhop = Hopping_operator(hop.Gauge, invP)

    dH=invhop(conj(d)) 

    
    zW=similar(W)
    for i in 1:length(W)
        for a in 1:4
            for b in 1:4
                zW[i][a,b]=W[i][b,a]
            end
        end
    end

    result=similar(d)
    for i in 1:length(W)
        result+=zW[i]*conj(dH[i])
    end

    if layer.residual == true
        result=result+d
    end


    return result
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
