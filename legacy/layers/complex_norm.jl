# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/activation.jl.

mutable struct complexNormLayer <: Layer
    activation::ActivationFunction

    function complexNormLayer(activation::ActivationFunction= Identity())
        new(activation)
    end

end

function (layer::complexNormLayer)(x::WF3x1)


    m=mean(x)
    x=x-m

    return x
end

function gradient(layer::complexNormLayer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix} )

    return nothing
end

function DeltaRule(layer::complexNormLayer, ::Identity,z::WF3x1, d::WF3x1)

    m=mean(d)

    

    result=d-m


    return result

end 
