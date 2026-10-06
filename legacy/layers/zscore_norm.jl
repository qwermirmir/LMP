# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/activation.jl.

mutable struct zscoreNormLayer <: Layer
    activation::ActivationFunction

    function zscoreNormLayer(activation::ActivationFunction= Identity())
        new(activation)
    end
end 

function (layer::zscoreNormLayer)(x::WF3x1)
    x=standardization(x)

    return x
end 

function gradient(layer::zscoreNormLayer, ::Identity, z::WF3x1, d::WF3x1 ,  γ::Vector{Spin_matrix} )

    return nothing
end

function DeltaRule(layer::zscoreNormLayer, ::Identity,z::WF3x1, d::WF3x1)


    real=(z+conj(z))/2
    imag=(z-conj(z))/2im

    mu=mean(z)

    rd=similar(d)
    id=similar(d)
    N=length(z)

    rσ=sqrt(norm2(real-mu.real)/N)
    iσ=sqrt(norm2(imag-mu.imag)/N)


    #---------------------------------------------------------------------------------
    # real 미분


end
