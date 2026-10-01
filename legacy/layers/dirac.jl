# Legacy code kept for reference. It is not included by src/LMP.jl and has not
# been kept up to date with the package; it will not run as is.
# Original location: src/Layer_sequence.jl, src/Layer_types.jl, src/Optimizer.jl, src/activation.jl.

struct Dirac_Layer <: Layer
    activation
    W::Nothing
    function Dirac_Layer(activation)
        new(activation)
    end
end

function (D::Dirac_Layer)(x::WF3x1)

    Dirac=D.activation
    y=Dirac*x
    return y
end 

function gradient(layer::Dirac_Layer,::Any,z::WF3x1,d::WF3x1 ,  γ::Vector{Spin_matrix} )


    return nothing
end 

function DeltaRule(layer::Dirac_Layer, ::Any,  z::WF3x1, d::WF3x1)

    D=layer.activation
    cd=conj(d)
    y=similar(d)
    mul!(y,D',cd)
    return conj(y)
end 

function update!(layer::Dirac_Layer, opt::Adam, grad)
    # Dirac_Layer에는 업데이트가 필요 없으므로 빈 함수로 둡니다.

   return
end 
