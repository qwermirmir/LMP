# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

mutable struct PtcRayleighQuotient_Layer <: Layer
    D::Any
    activation::ActivationFunction
    Out::Int
    function PtcRayleighQuotient_Layer(D::Any, activation::ActivationFunction= Identity(),Out =1)
        new(D, activation , Out)
    end
end 

function _has_gpu(layer::PtcRayleighQuotient_Layer)
    try; return eltype(layer.D.U) <: Gaugefields.AbstractGaugefields_module.Gaugefields_4D_accelerator; catch; return false; end
end

# --- PtcRayleighQuotient_Layer ---
function _to_cpu(layer::PtcRayleighQuotient_Layer)
    layer.D = _to_cpu_dirac(layer.D)
    return layer
end

function _shared_layer_gauge(layer::PtcRayleighQuotient_Layer)
    try
        return layer.D.U
    catch
        return nothing
    end
end

# --- PtcRayleighQuotient_Layer ---
function _to_gpu(layer::PtcRayleighQuotient_Layer, blocks=nothing)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks)
    return layer
end

function _to_gpu(layer::PtcRayleighQuotient_Layer, blocks, shared_U)
    layer.D = _to_gpu_dirac(layer.D; blocks=blocks, shared_U=shared_U)
    return layer
end

_refresh_layer!(layer::PtcRayleighQuotient_Layer, D_new, U_new) = (layer.D = D_new)

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

# function gradient(layer::L2Norm_Layer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})

#     return nothing
# end 



# function DeltaRule(layer::L2Norm_Layer, ::Identity, z::Vector{WF3x1},batch::Int)

#     @assert length(z) == 1 "L2Norm_Layer DeltaRule expects a single WF3x1 input."

#     result=2*(z[1]-layer.target)/batch

#     return [conj(result)]

# end 






function gradient(layer::PtcRayleighQuotient_Layer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix})

    return nothing
end 

function DeltaRule(layer::PtcRayleighQuotient_Layer, ::Identity, z::Vector{WF3x1}, batch::Int)

    result = Vector{WF3x1}(undef, length(z))


    for i in 1:length(z)

        n=norm2(z[i])

        Dirac=layer.D
        y=Dirac*z[i]
        u=Dirac'*y
        result[i] = similar(z[i])
        result[i]=2*(conj(u)/n- norm2(y)/n^2 * conj(z[i]))
    end 




    return result
end

function update!(layer::PtcRayleighQuotient_Layer,opt::Adam, grad::Any)

  
end 
