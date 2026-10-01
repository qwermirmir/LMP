# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

struct RayleighQuotient_Layer <: Layer
    x::WF3x1
    activation::ActivationFunction
    function RayleighQuotient_Layer(x::WF3x1, activation::ActivationFunction= Identity())
        new(x, activation)
    end


end 

function (layer::RayleighQuotient_Layer)(D::Dirac_operator)
    x=layer.x
    y=D*x
    rq=norm2(y)/norm2(x)

    return rq
end

function gradient(layer::RayleighQuotient_Layer, ::Identity, Dirac::Dirac_operator)


    x=layer.x
    n=norm2(x)


    
    y=Dirac*x
    u=Dirac'*y
    grad = similar(x)
    grad= conj(u)/n- norm2(y)/n^2 * conj(x)




    return [grad]
end

# RayleighQuotient_Layer: x :: WF3x1
function update!(layer::RayleighQuotient_Layer, opt::Adam, grad::WF3x1)
    wf_adam_step!(layer.x, opt, grad)
    return nothing
end
